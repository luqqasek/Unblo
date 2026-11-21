"""Check generated data, official metrics, nested splits and immutable sources."""
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent; OUT = HERE / 'results'; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from wmloc.watermarks import StegaStamp
from prepare import cover_and_mask, digest
spec = importlib.util.spec_from_file_location('synthetic_report_checks', HERE / 'report.py')
report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
official, measure = report.official, report.measure

def main():
    torch.set_num_threads(4)
    config = json.loads((OUT / 'config.json').read_text())
    assert all(digest(ROOT / p) == h for p, h in config['sources_sha256'].items())
    generation = json.loads((OUT / 'generation_config.json').read_text())
    assert digest(HERE / 'prepare.py') == generation['code_sha256']
    assert digest(ROOT / 'checkpoints/stegastamp/stega_stamp_logits.onnx') == generation['model_sha256']
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert len(manifest) == 2200 and len({r['image_id'] for r in manifest}) == 2200
    assert len({r['coco_id'] for r in manifest}) == len({r['cover_sha256'] for r in manifest}) == 2200
    assert all(r['split'] == ('train' if i < 2000 else 'validation') for i, r in enumerate(manifest))
    assert all(len(r['message']) == 100 and set(r['message']) <= {'0', '1'} for r in manifest)
    assert all(r['min_real_dhash'] > 12 for r in manifest)
    verified = 0
    for shard in sorted((OUT / 'shards').glob('*.pt')):
        c = torch.load(shard, weights_only=True)
        assert len(c['names']) == 100
        for f in c['features']:
            assert f.dtype == torch.float32 and torch.isfinite(f).all()
        for name, target in zip(c['names'], c['target']):
            image = np.asarray(Image.open(OUT / 'images' / f'{name}.png').convert('RGB'))
            mask = np.asarray(Image.open(OUT / 'masks' / f'{name}.png'))
            assert image.shape == (512, 512, 3) and mask.shape == (512, 512)
            assert set(np.unique(mask)) <= {0, 255}
            np.testing.assert_array_equal(official.downsample_mask(mask > 127), target.numpy())
            r = manifest[int(name.rsplit('_', 1)[1])]
            assert hashlib.sha256(image.tobytes()).hexdigest() == r['image_sha256']
            assert float(target.float().mean()) == r['area']
            verified += 1
    assert verified == 2200
    splits = json.loads((OUT / 'splits.json').read_text())
    assert splits == json.loads((HERE.parent / 'stegastamp_stn_alignment/results/splits.json').read_text())
    rows = pd.read_csv(OUT / 'per_image_metrics.csv')
    assert len(rows) == 1020 and rows.image_id.nunique() == 170
    max_metric_error = 0.
    for variant in ('control', 'synthetic'):
        for seed in (42, 7, 123):
            seen = []
            for split in splits:
                stem = f'{variant}_fold{split["fold"]}_seed{seed}'
                ck = torch.load(OUT / f'{stem}.pt', weights_only=True)
                z = np.load(OUT / f'{stem}.npz')
                r = json.loads((OUT / f'{stem}.json').read_text())
                assert ck['validation_names'] == z['names'].tolist() == split['validation'] == r['names']
                assert ck['train_names'] == split['train']
                assert not set(ck['train_names']) & set(ck['validation_names'])
                assert ck['config_sha256'] == digest(OUT / 'config.json')
                assert ck['epoch'] == r['epoch'] and ck['threshold'] == r['threshold'] == float(z['threshold'])
                assert r['epoch'] == np.argmax([h['miou'] for h in r['inner_history']]) + 1
                assert r['threshold'] == max(r['inner_threshold_scores'], key=lambda t: t['miou'])['threshold']
                if variant == 'synthetic':
                    assert ck['pretrain_sha256'] == digest(OUT / f'pretrain_seed{seed}.pt')
                seen += ck['validation_names']
                official_scores = [official.miou_image(p >= r['threshold'], g) for p, g in zip(z['probability'], z['target'])]
                np.testing.assert_allclose(official_scores, r['per_image'], atol=1e-12)
                actual = rows[(rows.variant == variant) & (rows.seed == seed) & (rows.fold == split['fold'])].set_index('image_id')
                np.testing.assert_allclose(actual.loc[r['names'], 'miou'], official_scores, atol=1e-12)
                max_metric_error = max(max_metric_error, abs(np.mean(official_scores) - r['miou']))
            assert len(seen) == len(set(seen)) == 170
    # Edge cases: official empty-union handling and undefined rankings.
    zeros = np.zeros((128, 128), bool); ones = ~zeros
    assert measure(zeros.astype(float), zeros, .5)['miou'] == 1
    assert measure(ones.astype(float), ones, .5)['miou'] == 1
    assert measure(zeros.astype(float), ones, .5)['miou'] == 0
    assert np.isnan(measure(zeros.astype(float), zeros, .5)['auc'])
    half = zeros.copy(); half[:64] = True
    tie = measure(np.full(half.shape, .5), half, .5)
    assert tie['auc'] == .5 and tie['ap'] == .5 and tie['topk'] == .5
    # Regenerate a diverse deterministic subset including full/empty/local/restored/object examples.
    with zipfile.ZipFile(OUT / 'sources/annotations_trainval2017.zip') as archive:
        coco = json.loads(archive.read('annotations/instances_val2017.json'))
    by_id = {}
    for a in coco['annotations']: by_id.setdefault(a['image_id'], []).append(a)
    wm = StegaStamp(ROOT / 'checkpoints', 'cpu')
    reproduced = []
    with zipfile.ZipFile(OUT / 'sources/val2017.zip') as archive:
        for index in (0, 1, 3, 4, 5, 6, 17, 18, 1999, 2000, 2001, 2199):
            r = manifest[index]; raw = archive.read('val2017/' + r['source_filename'])
            rng = np.random.default_rng(generation['seed'] + index)
            cover, region, shape, crop = cover_and_mask(raw, by_id.get(r['coco_id'], []), rng, ['rectangle', 'ellipse', 'object'][index % 3])
            bits = rng.integers(0, 2, 100, dtype=np.uint8)
            assert r['message'] == ''.join(map(str, bits.tolist()))
            if r['mode'] == 'clean':
                image = cover.copy(); mask = np.zeros((512, 512), bool)
            elif r['mode'] in ('full', 'restore'):
                marked = wm.embed(cover, bits)
                mask = np.ones((512, 512), bool) if r['mode'] == 'full' else ~region
                image = np.where(mask[:, :, None], marked, cover)
            else:
                yy, xx = np.where(region); y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
                marked = cover.copy(); marked[y0:y1, x0:x1] = wm.embed(cover[y0:y1, x0:x1], bits)
                mask = region; image = np.where(mask[:, :, None], marked, cover)
            assert hashlib.sha256(image.tobytes()).hexdigest() == r['image_sha256']
            assert np.array_equal(image[~mask], cover[~mask])
            np.testing.assert_array_equal(mask, np.asarray(Image.open(OUT / 'masks' / (r['image_id'] + '.png'))) > 127)
            reproduced.append(r['image_id'])
    summary = json.loads((OUT / 'summary.json').read_text())
    for variant in ('control', 'synthetic'):
        assert abs(rows[rows.variant == variant].miou.mean() - summary['metrics']['miou'][variant]) < 1e-12
    result = dict(status='passed', generated_masks_verified=verified, unique_covers=2200, oof_rows=1020,
                  unique_real_images=170, nested_runs=18, source_hashes_unchanged=True, mask_downsample_exact=True,
                  official_score_max_error=max_metric_error, regenerated_exactly=reproduced,
                  notes='Each checkpoint prediction was reloaded/reproduced on GPU during training. No test data used.')
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__': main()
