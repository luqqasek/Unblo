"""Prepare a source-disjoint native512 clean/edit pool before exact embedding."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from scipy.fft import dctn

from audit_pipe import KEY, key, source_key, decode, sha256


def phash(image):
    x = np.asarray(Image.fromarray(image).convert('L').resize((32, 32)), np.float32)
    v = dctn(x, norm='ortho')[:8, :8].ravel()[1:]
    return np.packbits(v > np.median(v)).tobytes().hex()


def near(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count() <= 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--train', type=int, default=512)
    p.add_argument('--dev', type=int, default=64)
    p.add_argument('--locked', type=int, default=64)
    a = p.parse_args()
    folder = a.root / 'data-expansion'
    out = folder / 'pipe-prepared-v2'
    out.mkdir(exist_ok=True)
    assert not (out / 'completed.json').exists(), 'Preserve completed pool'
    manifest = json.loads((folder / 'download_manifest.json').read_text())
    image_file = folder / 'pipe' / manifest['file']
    mask_file = folder / 'pipe-masks' / manifest['mask_file']
    assert sha256(image_file) == manifest['sha256'] and sha256(mask_file) == manifest['mask_sha256']
    old = json.loads((a.root / 'sources/coco2017/source_partitions.json').read_text())
    old_ids = {s for ids in old.values() for s in ids}
    meta = pq.read_table(image_file, columns=list(KEY)).to_pylist()
    sources = sorted({source_key(r) for r in meta} - old_ids)
    def partition(s):
        v = int(hashlib.sha256(('pipe42:' + s).encode()).hexdigest(), 16) % 100
        return 'locked' if v < 10 else ('dev' if v < 20 else 'train')
    assignments = {s: partition(s) for s in sources}
    (out / 'source_partitions.json').write_text(json.dumps(assignments, indent=2) + '\n')
    # Only source metadata determines partitions; no real GT is consulted.
    wanted = {key(r) for r in meta if source_key(r) in assignments}
    mask_rows = {}
    for batch in pq.ParquetFile(mask_file).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            if key(row) in wanted:
                mask_rows.setdefault(key(row), []).append(row['mask'])
    train_images = sorted((a.root / 'task-1/data/train/images').glob('*.png'))
    assert len(train_images) == 510
    fingerprints = []
    for f in train_images:
        x = np.asarray(Image.open(f).convert('RGB'))
        fingerprints.append((f.stem, hashlib.sha256(x.tobytes()).hexdigest(), phash(x)))
    limits = {'train': a.train, 'dev': a.dev, 'locked': a.locked}
    accepted = {k: [] for k in limits}
    seen_sources, seen_hashes = set(), []
    rejected = Counter()
    for batch in pq.ParquetFile(image_file).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            sid, k = source_key(row), key(row)
            if sid not in assignments or sid in seen_sources:
                continue
            split = assignments[sid]
            if len(accepted[split]) >= limits[split]:
                continue
            variants = mask_rows.get(k, [])
            if not variants:
                rejected['missing_mask'] += 1
                continue
            if len({hashlib.sha256(v['bytes']).hexdigest() for v in variants}) != 1:
                rejected['ambiguous_mask'] += 1
                continue
            original, _ = decode(row['target_img'])
            edited, _ = decode(row['source_img'])
            mask, _ = decode(variants[0])
            if original.shape != (512, 512, 3) or edited.shape != original.shape:
                rejected['non_native_or_unaligned'] += 1
                continue
            original_mask_shape = list(mask.shape[:2])
            # Author confirms torchvision.Resize((512,512)), a stretch, not a crop:
            # https://huggingface.co/datasets/paint-by-inpaint/PIPE_Masks/discussions/4
            # This operates on clean edit metadata BEFORE any watermark embedding.
            if mask.shape[:2] != (512, 512):
                mask = np.asarray(Image.fromarray(mask).resize((512,512), Image.Resampling.BILINEAR))
            support = mask[..., 0] >= 128
            area = float(support.mean())
            if not .003 <= area <= .45:
                rejected['area_outside_declared_range'] += 1
                continue
            h = hashlib.sha256(original.tobytes()).hexdigest()
            ph = phash(original)
            hits = [n for n, hh, pp in fingerprints if hh == h or near(ph, pp)]
            if hits:
                rejected['competition_train_duplicate_candidate'] += 1
                continue
            if any(h == hh or near(ph, pp) for hh, pp in seen_hashes):
                rejected['new_pool_duplicate_candidate'] += 1
                continue
            sid_hash = hashlib.sha256(sid.encode()).hexdigest()[:16]
            target = out / split / sid_hash
            target.mkdir(parents=True, exist_ok=True)
            for name, x in [('clean', original), ('edited', edited), ('edit_mask', support.astype(np.uint8)*255)]:
                f = target / f'{name}.png'
                Image.fromarray(x).save(f, compress_level=1)
                assert np.array_equal(np.asarray(Image.open(f)), x)
            record = {'source_id': sid, 'partition': split, 'join_key': k, 'path': str(target),
                      'edit_fraction': area, 'original_pixel_sha256': h, 'phash': ph,
                      'original_bytes_sha256': hashlib.sha256(row['target_img']['bytes']).hexdigest(),
                      'edited_bytes_sha256': hashlib.sha256(row['source_img']['bytes']).hexdigest(),
                      'mask_bytes_sha256': hashlib.sha256(variants[0]['bytes']).hexdigest(),
                      'mask_original_shape': original_mask_shape,
                      'mask_rule': 'PIL bilinear stretch to512 per author; channel0>=128 before embedding',
                      'screened_competition_train_n': 510, 'post_embedding_transforms': False}
            accepted[split].append(record)
            seen_sources.add(sid)
            seen_hashes.append((h, ph))
        if all(len(accepted[k]) >= v for k, v in limits.items()):
            break
    for split, rows in accepted.items():
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = {'counts': {k: len(v) for k, v in accepted.items()}, 'requested': limits,
              'source_counts': dict(Counter(r['source_id'].split(':')[0] for rows in accepted.values() for r in rows)),
              'rejections': dict(rejected), 'assigned_new_sources': len(assignments),
              'competition_test_read': False, 'exact_png_roundtrip': True,
              'duplicate_screen': 'decoded-pixel SHA256 and pHash Hamming<=4 vs510train and retained new photos; not exhaustive crop deduplication',
              'embedding_calls': 0, 'training_started': False, 'source_manifest': manifest}
    assert all(result['counts'][k] == v for k, v in limits.items()), result
    (out / 'completed.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
