"""Full Head on inverse-STN features; matched stable nested CV, all three seeds."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
BASE = HERE.parent / 'stegastamp_stable_selection/results'
AUDIT = HERE.parent / 'diagnostic_audit'
spec = importlib.util.spec_from_file_location('stable_selection_runner', BASE.parent / 'run.py')
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)
cv = stable.cv
sys.path.insert(0, str(AUDIT))
from execution_audit import inverse_feature_grid


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def geometry_checks():
    """Independent affine coordinate-ramp test catches direction, scale and axis errors."""
    neutral = np.array([[399 / 398, 0, 1 / 398], [0, 399 / 398, 1 / 398]])
    matrices = [neutral, np.array([[1.08, .03, .04], [-.02, 1.03, -.05]])]
    max_error = 0.
    for level, size in enumerate((32, 32, 25, 13)):
        native, stride, start = ((200, 2, 1), (100, 4, 3), (25, 16, 15), (13, 32, 15))[level]
        centers = start + stride * np.arange(native, dtype=float)
        if size < native:
            centers = np.array([centers[int(np.floor(j * native / size)):int(np.ceil((j + 1) * native / size))].mean() for j in range(size)])
        yy, xx = np.meshgrid(centers, centers, indexing='ij')
        for theta in matrices:
            # Each source feature stores the ORIGINAL pixel coordinates at its STN location.
            coords = (np.stack([xx, yy], -1) * 2 / 399 - 1) @ theta[:, :2].T + theta[:, 2]
            ramp = torch.from_numpy(((coords + 1) * 199).transpose(2, 0, 1).copy()).float()[None]
            grid = inverse_feature_grid(theta, level, size)
            recovered = F.grid_sample(ramp, grid, align_corners=False, padding_mode='border')[0].numpy().transpose(1, 2, 0)
            # Border replication cannot reconstruct coordinates outside available features.
            grid_values = grid[0].numpy()
            valid = (np.abs(grid_values) <= 1 - 1 / size - 1e-6).all(-1)
            assert valid.any()
            error = float(np.max(np.abs(recovered[valid] - np.stack([xx, yy], -1)[valid])))
            assert error < 1e-3, (level, theta, error)
            max_error = max(max_error, error)
        expected_y, expected_x = torch.meshgrid((torch.arange(size) + .5) * 2 / size - 1, (torch.arange(size) + .5) * 2 / size - 1, indexing='ij')
        np.testing.assert_allclose(inverse_feature_grid(neutral, level, size), torch.stack([expected_x, expected_y], -1)[None], atol=1e-7)
    return max_error


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sources = [cv.model.OUT / 'stegastamp_features.pt', cv.OUT / 'stegastamp_oof.npz',
               cv.OUT / 'folds.json', BASE / 'splits.json', BASE / 'config.json', BASE / 'metrics.json',
               AUDIT / 'results/stn_transforms.csv', AUDIT / 'results/cache/stegastamp_aligned_features.pt',
               AUDIT / 'execution_audit.py', cv.HERE / 'run.py', cv.HERE / 'cross_validate.py',
               BASE.parent / 'run.py', cv.model.ROOT / 'data/train/masks_128.npz',
               cv.model.ROOT / 'data/train/metadata.json', cv.model.ROOT / 'checkpoints/stegastamp/stega_stamp_logits.onnx']
    sources += sorted(BASE.glob('stegastamp_fold*.*'))
    sources += [p for exp in ('frozen_extractor', 'stegastamp_stable_selection', 'decoder_evidence')
                for p in (HERE.parent / exp / 'results/full_train').glob('*.pt')]
    hashes = {str(p.relative_to(cv.model.ROOT)): digest(p) for p in sources}
    config = dict(seeds=list(stable.SEEDS), baseline='stable nested selection; existing raw checkpoints reloaded and verified',
                  change='inverse actual STN on all four cached feature levels before unchanged Head; fresh random training',
                  selection='unchanged: 3 stratified inner folds, pooled mIoU at 0.5 for epoch, inner threshold search, fresh outer refit',
                  training=cv.model.CONFIG, sources_sha256=hashes, runner_sha256=digest(__file__),
                  torch=str(torch.__version__), numpy=np.__version__, threads=4,
                  primary='paired mean per-image OOF mIoU over all seeds; report all 9 fold-seed pairs',
                  secondary=['macro ROC-AUC', 'macro AP', 'watermark IoU', 'internal boundary F1@2', 'small-region IoU'])
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (OUT / 'config.json').exists():
        previous = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
        if previous != config:
            assert not any(OUT.glob('stegastamp_fold*')) and not (OUT / 'aligned_features.pt').exists(), 'Sources/config changed; do not resume stale results.'
            save(OUT / 'config.json', config)
    else:
        save(OUT / 'config.json', config)
    raw = torch.load(cv.model.OUT / 'stegastamp_features.pt', weights_only=True)
    source = np.load(cv.OUT / 'stegastamp_oof.npz')
    names = raw['names']
    assert names == source['names'].tolist() and len(set(names)) == 170
    gt = source['target']
    mask_source = np.load(cv.model.ROOT / 'data/train/masks_128.npz')
    masks_by_name = dict(zip(mask_source['names'].tolist(), mask_source['masks']))
    for i, name in enumerate(names):
        np.testing.assert_array_equal(gt[i], masks_by_name[name])
    old_config = json.loads((BASE / 'config.json').read_text())
    assert all(old_config[k] == v for k, v in cv.model.CONFIG.items())
    assert old_config['seeds'] == list(stable.SEEDS)
    splits = json.loads((BASE / 'splits.json').read_text())
    original = json.loads((cv.OUT / 'folds.json').read_text())['stegastamp']
    metadata = json.loads((cv.model.ROOT / 'data/train/metadata.json').read_text())
    flags_by_name = {r['image_name']: r['full_watermark'] for r in metadata}
    flags = np.array([flags_by_name[n] for n in names])
    seen = []
    for split, orig in zip(splits, original):
        assert split['train'] == orig['train'] and split['validation'] == orig['validation']
        assert not set(split['train']) & set(split['validation'])
        fit = np.array([names.index(n) for n in split['train']])
        expected = [[names[i] for i in part] for part in stable.inner_folds(fit, gt, flags)]
        assert split['inner_validation'] == expected
        assert sorted(sum(expected, [])) == sorted(split['train'])
        seen += split['validation']
    assert sorted(seen) == sorted(names) and len(splits) == 3
    save(OUT / 'splits.json', splits)
    ramp_error = geometry_checks()
    transforms = pd.read_csv(AUDIT / 'results/stn_transforms.csv').set_index('image_id')
    assert transforms.index.is_unique and set(transforms.index) == set(names)
    aligned = [f.clone() for f in raw['features']]
    reference = torch.load(AUDIT / 'results/cache/stegastamp_aligned_features.pt', weights_only=True)
    assert reference['names'] == names
    for i, name in enumerate(names):
        r = transforms.loc[name]
        theta = np.array([[r.a00, r.a01, r.tx], [r.a10, r.a11, r.ty]])
        assert np.linalg.det(theta[:, :2]) > 0
        for level, features in enumerate(aligned):
            features[i:i + 1] = F.grid_sample(features[i:i + 1], inverse_feature_grid(theta, level, features.shape[-1]),
                                             mode='bilinear', align_corners=False, padding_mode='border')
    for generated, expected in zip(aligned, reference['features']):
        torch.testing.assert_close(generated, expected, atol=1e-7, rtol=0)
        assert torch.isfinite(generated).all()
    feature_path = OUT / 'aligned_features.pt'
    if not feature_path.exists():
        torch.save(dict(names=names, features=aligned, config_sha256=config_hash), feature_path)
    else:
        saved = torch.load(feature_path, weights_only=True)
        assert saved['config_sha256'] == config_hash and saved['names'] == names
        for a, b in zip(aligned, saved['features']):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    baseline_error = 0.
    for split in splits:
        ids = [names.index(n) for n in split['validation']]
        for seed in stable.SEEDS:
            stem = f"stegastamp_fold{split['fold']}_seed{seed}"
            old = json.loads((BASE / f'{stem}.json').read_text())
            ck = torch.load(BASE / f'{stem}.pt', weights_only=True)
            head = cv.model.Head(ck['channels'])
            head.load_state_dict(ck['state_dict'])
            p = cv.predict(head, raw['features'], ids)
            z = np.load(BASE / f'{stem}.npz')
            assert z['names'].tolist() == old['names'] == split['validation']
            assert ck['epoch'] == old['epoch'] and ck['threshold'] == old['threshold'] == float(z['threshold'])
            np.testing.assert_allclose(p, z['probability'], atol=1e-7, rtol=0)
            np.testing.assert_allclose(cv.model.metrics(p, gt[ids], old['threshold'])['per_image'], old['per_image'], atol=1e-12)
            baseline_error = max(baseline_error, float(np.max(np.abs(p - z['probability']))))
    save(OUT / 'preflight.json', dict(geometry_ramp_max_error=ramp_error, baseline_probability_max_error=baseline_error,
                                    raw_checkpoints_verified=9, aligned_cache_matches_probe=True, splits_verified=True,
                                    config_sha256=config_hash, feature_sha256=digest(feature_path)))
    print('PREFLIGHT OK: geometry, 170 aligned features, 9 baseline checkpoints, identical outer/inner splits.', flush=True)
    return names, aligned, gt, splits, config_hash


def fit(features, gt, names, split, seed, config_hash):
    fold = split['fold']
    stem = f'stegastamp_fold{fold}_seed{seed}'
    result_path = OUT / f'{stem}.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['config_sha256'] == config_hash
        return result
    start = time.perf_counter()
    fit_ids = np.array([names.index(n) for n in split['train']])
    val = np.array([names.index(n) for n in split['validation']])
    parts = [np.array([names.index(n) for n in part]) for part in split['inner_validation']]
    targets = torch.from_numpy(gt).float()
    models = [stable.initialize(features, seed) for _ in parts]
    inner_fit = [np.setdiff1d(fit_ids, p) for p in parts]
    best, stale, history = -1, 0, []
    inner_gt = gt[np.concatenate(parts)]
    for epoch in range(1, cv.model.CONFIG['max_epochs'] + 1):
        probs = []
        for (head, opt, rng), train_ids, valid_ids in zip(models, inner_fit, parts):
            assert not set(train_ids) & set(valid_ids) and not set(train_ids) & set(val)
            cv.train_epoch(head, opt, rng, features, targets, train_ids)
            probs.append(cv.predict(head, features, valid_ids))
        pooled = np.concatenate(probs)
        metric = cv.model.metrics(pooled, inner_gt, .5)['miou']
        history.append(metric)
        if metric > best:
            best, stale, selected_epoch = metric, 0, epoch
            best_prob = pooled.copy()
        else:
            stale += 1
        if epoch % 5 == 0:
            print(f'fold={fold} seed={seed} inner epoch={epoch} mIoU={metric:.6f}', flush=True)
        if stale >= cv.model.CONFIG['patience']:
            break
    threshold_scores = [dict(threshold=t, miou=cv.model.metrics(best_prob, inner_gt, t)['miou']) for t in cv.model.CONFIG['threshold_candidates']]
    threshold = max(threshold_scores, key=lambda r: r['miou'])['threshold']
    # Fresh random initialization; no existing state_dict is loaded for training.
    head, opt, rng = stable.initialize(features, seed)
    for _ in range(selected_epoch):
        cv.train_epoch(head, opt, rng, features, targets, fit_ids)
    prob = cv.predict(head, features, val)
    measured = cv.model.metrics(prob, gt[val], threshold)
    old = json.loads((BASE / f'{stem}.json').read_text())
    result = dict(fold=fold, seed=seed, epoch=selected_epoch, threshold=threshold,
                  miou=measured['miou'], baseline=old['miou'], delta=measured['miou'] - old['miou'],
                  per_image=measured['per_image'], names=split['validation'], inner_history=history,
                  inner_threshold_scores=threshold_scores, config_sha256=config_hash,
                  seconds=time.perf_counter() - start)
    checkpoint = dict(state_dict=head.state_dict(), channels=[f.shape[1] for f in features], epoch=selected_epoch,
                      threshold=threshold, seed=seed, config_sha256=config_hash, feature_geometry='inverse_actual_STN',
                      train_names=split['train'], validation_names=split['validation'])
    torch.save(checkpoint, OUT / f'{stem}.pt')
    np.savez_compressed(OUT / f'{stem}.npz', names=split['validation'], probability=prob, threshold=threshold)
    restored = cv.model.Head(checkpoint['channels'])
    restored.load_state_dict(torch.load(OUT / f'{stem}.pt', weights_only=True)['state_dict'])
    np.testing.assert_allclose(cv.predict(restored, features, val), prob, atol=1e-7, rtol=0)
    save(result_path, result)
    print(f"COMPLETE fold={fold} seed={seed}: {result['baseline']:.6f} -> {result['miou']:.6f}, delta={result['delta']:+.6f}, epoch={selected_epoch}, threshold={threshold}, seconds={result['seconds']:.1f}", flush=True)
    return result


def main():
    names, features, gt, splits, config_hash = prepare()
    runs = []
    for split in splits:
        for seed in stable.SEEDS:
            runs.append(fit(features, gt, names, split, seed, config_hash))
            save(OUT / 'metrics.json', dict(status='complete' if len(runs) == 9 else 'running', runs=runs, config_sha256=config_hash))
    config = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
    assert all(digest(cv.model.ROOT / p) == h for p, h in config['sources_sha256'].items())
    print('ALL 9 ALIGNED RUNS COMPLETE; source artifacts unchanged.', flush=True)


if __name__ == '__main__':
    main()
