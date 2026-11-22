"""Bounded raw-parquet audit. No watermark pixels or competition test data read."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


KEY = ('target_img_dataset', 'img_id', 'ann_id')


def key(row):
    return tuple(str(row[k]) for k in KEY)


def source_key(row):
    family, name, _ = key(row)
    if 'coco' in family.lower() or 'lvis' in family.lower():
        return f'coco2017:{int(name)}'
    return f'{family.lower()}:{name}'


def decode(value):
    assert value['bytes'], 'Use embedded raw bytes, not viewer thumbnails'
    with Image.open(io.BytesIO(value['bytes'])) as im:
        return np.asarray(im.convert('RGB')).copy(), im.format


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--inspect', type=int, default=64)
    a = p.parse_args()
    folder = a.root / 'data-expansion'
    manifest = json.loads(Path(__file__).with_name('download_manifest.json').read_text())
    images = folder / 'pipe' / manifest['file']
    masks = folder / 'pipe-masks' / manifest['mask_file']
    hashes = {str(images): sha256(images), str(masks): sha256(masks)}
    assert hashes[str(images)] == manifest['sha256']
    assert hashes[str(masks)] == manifest['mask_sha256']
    meta = pq.read_table(images, columns=list(KEY)).to_pylist()
    mmeta = pq.read_table(masks, columns=list(KEY)).to_pylist()
    indexes = {key(row): i for i, row in enumerate(mmeta)}
    key_counts = Counter(key(row) for row in mmeta)
    selected = []
    for batch in pq.ParquetFile(images).iter_batches(batch_size=16):
        selected.extend(batch.to_pylist())
        if len(selected) >= a.inspect:
            break
    selected = selected[:a.inspect]
    wanted = {key(row) for row in selected}
    mask_rows = {}
    for batch in pq.ParquetFile(masks).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            if key(row) in wanted:
                mask_rows.setdefault(key(row), []).append(row)
        if all(len(mask_rows.get(k, [])) == key_counts[k] for k in wanted):
            break
    inspected = []
    for row in selected:
        k = key(row)
        if k not in mask_rows:
            inspected.append({'key': k, 'error': 'No mask in downloaded shard'})
            continue
        variants = mask_rows[k]
        mask_hashes = {hashlib.sha256(v['mask']['bytes']).hexdigest() for v in variants}
        if len(mask_hashes) != 1:
            inspected.append({'key': k, 'error': 'Conflicting masks under one join key', 'variants': len(variants)})
            continue
        original, original_format = decode(row['target_img'])
        edited, edited_format = decode(row['source_img'])
        mask_rgb, mask_format = decode(variants[0]['mask'])
        mask = mask_rgb[..., 0]
        record = {'key': k, 'source_id': source_key(row), 'original_shape': list(original.shape),
                  'edited_shape': list(edited.shape), 'mask_shape': list(mask.shape),
                  'original_format': original_format, 'edited_format': edited_format,
                  'mask_format': mask_format, 'mask_unique_count': len(np.unique(mask)),
                  'original_sha256': hashlib.sha256(row['target_img']['bytes']).hexdigest()}
        if original.shape == edited.shape and mask.shape == original.shape[:2]:
            changed = np.any(original != edited, axis=2)
            record['mask_area_nonzero'] = float((mask > 0).mean())
            record['mask_area_half'] = float((mask >= 128).mean())
            record['outside_nonzero_mask_changed_pixels'] = int((changed & (mask == 0)).sum())
            record['outside_half_mask_changed_pixels'] = int((changed & (mask < 128)).sum())
            record['changed_pixels'] = int(changed.sum())
        inspected.append(record)
    partitions = json.loads((a.root / 'sources/coco2017/source_partitions.json').read_text())
    existing = {source: split for split, sources in partitions.items() for source in sources}
    unique = sorted({source_key(row) for row in meta})
    report = {'hashes': hashes, 'image_rows': len(meta), 'mask_rows': len(mmeta),
              'duplicated_mask_keys': sum(v > 1 for v in key_counts.values()),
              'distinct_sources': len(unique), 'sources_by_dataset': dict(Counter(s.split(':')[0] for s in unique)),
              'matched_mask_rows': sum(key(r) in indexes for r in meta),
              'existing_source_partitions': dict(Counter(existing.get(s, 'new') for s in unique)),
              'inspected': inspected, 'pyarrow_version': __import__('pyarrow').__version__,
              'policy': 'Group COCO and LVIS by shared photo ID; preserve existing splits. Embed original then hard-paste edited pixels only within declared support; never infer provenance from RGB differences.'}
    (folder / 'raw_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('inspected', 'hashes')}, indent=2))
    print('RAW_EXAMPLES', json.dumps(inspected[:2]))


if __name__ == '__main__':
    main()
