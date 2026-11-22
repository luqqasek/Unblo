"""Exact original PixelSeal encoding, source-disjoint prepared PIPE images."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from experiments.native_localizer.common import atomic_json, sha256
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--part', type=int, choices=(0,1), required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    prepared = a.root / 'data-expansion/pipe-prepared-v2'
    assert (prepared / 'completed.json').exists()
    assert json.loads((prepared/'old_source_duplicate_audit.json').read_text())['passed']
    out = a.root / 'data-expansion/encoded/pixelseal'
    out.mkdir(parents=True, exist_ok=True)
    marker = out / f'completed{a.part}.json'
    if marker.exists():
        return
    start = time.monotonic()
    wm = load_watermark('pixelseal', models_dir='checkpoints', device='cuda')
    assert wm.nbits == 256
    counts = {}
    for split in ('train','dev'):
        rows = [json.loads(s) for s in (prepared / f'{split}.jsonl').read_text().splitlines()]
        records = []
        for i in range(a.part,len(rows),2):
            row = rows[i]
            dest = out / split / Path(row['path']).name
            dest.mkdir(parents=True, exist_ok=True)
            clean = np.asarray(Image.open(Path(row['path'])/'clean.png').convert('RGB'))
            edited = np.asarray(Image.open(Path(row['path'])/'edited.png').convert('RGB'))
            mask = np.asarray(Image.open(Path(row['path'])/'edit_mask.png')) >= 128
            rng = np.random.default_rng(901042 + i + (100000 if split == 'dev' else 0))
            bits = rng.integers(0,2,256).tolist()
            with torch.inference_mode():
                marked = wm.embed(clean.copy(), bits)
            assert marked.dtype == np.uint8 and marked.shape == clean.shape == (512,512,3)
            path = dest / 'marked.png'
            Image.fromarray(marked).save(path, compress_level=1)
            assert np.array_equal(np.asarray(Image.open(path)), marked)
            # Provenance invariants for both new constructions, before training.
            hole = marked.copy(); hole[mask] = edited[mask]
            fragment = edited.copy(); fragment[mask] = marked[mask]
            assert np.array_equal(hole[~mask],marked[~mask])
            assert np.array_equal(fragment[mask],marked[mask])
            y = downsample_mask(mask)
            residual = marked.astype(np.float32)-clean.astype(np.float32)
            records.append(dict(row,message=bits,marked_path=str(path),marked_sha256=sha256(path),
                                embed_rectangle=[0,0,512,512],embedding_calls=1,
                                foreground_fragment_fraction=float(y.mean()),
                                rms=float(np.sqrt((residual**2).mean())),
                                zero_pixel_fraction=float(np.all(residual==0,axis=2).mean())))
            if len(records)%64==0:
                print('ENCODE',a.part,split,len(records),time.monotonic()-start,flush=True)
        (out / f'{split}-part{a.part}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        counts[split] = len(records)
    atomic_json(marker,{'counts':counts,'seconds':time.monotonic()-start,
                        'wrapper_sha256':sha256(Path('wmloc/watermarks.py')),
                        'model_sha256':sha256(Path('checkpoints/pixelseal/checkpoint.pth')),
                        'no_post_quantization_transform':True,'source_manifest_sha256':sha256(prepared/'completed.json')})


if __name__ == '__main__':
    main()
