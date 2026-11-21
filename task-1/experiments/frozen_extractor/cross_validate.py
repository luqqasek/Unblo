"""Three-fold OOF evaluation; epoch and threshold selection stay inside training folds."""
import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_extractor_model', HERE / 'run.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
OUT = HERE / 'results' / 'cv3'
METHODS = ('pixelseal', 'trustmark', 'stegastamp')


def initialize(features):
    torch.manual_seed(42)
    head = model.Head([f.shape[1] for f in features])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
    return head, optimizer, np.random.default_rng(42)


def train_epoch(head, optimizer, rng, features, targets, ids):
    head.train()
    shuffled = rng.permutation(ids)
    for start in range(0, len(shuffled), 8):
        batch = shuffled[start:start + 8]
        optimizer.zero_grad()
        logits = head([f[batch] for f in features])
        target = targets[batch]
        p = logits.sigmoid()
        dice = 1 - ((2 * (p * target).sum((1, 2)) + 1) /
                    (p.sum((1, 2)) + target.sum((1, 2)) + 1)).mean()
        loss = .5 * F.binary_cross_entropy_with_logits(logits, target) + .5 * dice
        loss.backward()
        optimizer.step()


def predict(head, features, ids):
    head.eval()
    with torch.inference_mode():
        return torch.cat([head([f[ids[j:j + 8]] for f in features]).sigmoid()
                          for j in range(0, len(ids), 8)]).numpy()


def summarize(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                minimum=float(np.min(values)), maximum=float(np.max(values)),
                range=float(np.ptp(values)))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = model.LocalisationDataset(model.ROOT / 'data', 'train')
    # Fail rather than silently put exact duplicate images in different folds.
    hashes = [model.hashlib.sha256(item['image'].tobytes()).hexdigest() for item in ds]
    assert len(set(hashes)) == len(hashes), 'Exact duplicates require grouped folds'
    results = dict(config=model.CONFIG, folds=3, seed=42, methods={},
                   protocol='Inner 80/20 epoch/threshold selection; fresh refit on entire outer train; outer OOF evaluation',
                   exact_duplicate_images=0)
    splits = {}
    global_scores = [[] for _ in range(3)]
    global_fixed = [[] for _ in range(3)]
    for method in METHODS:
        cache = model.extract(method, ds)
        names = cache['names']
        assert set(names) == {m['image_name'] for m in ds.meta if m['method'] == method}
        assert len(names) == len(set(names))
        features = cache['features']
        targets = torch.from_numpy(np.stack([ds.m128[n] for n in names])).float()
        parts = np.array_split(np.random.default_rng(42).permutation(len(names)), 3)
        seen = np.zeros(len(names), dtype=int)
        oof = np.zeros(targets.shape, dtype=np.float32)
        thresholds = np.zeros(len(names))
        rows = []
        splits[method] = []
        for fold, val in enumerate(parts):
            start = time.perf_counter()
            fit = np.setdiff1d(np.arange(len(names)), val)
            inner = np.random.default_rng(42 + fold).permutation(fit)
            nval = int(np.ceil(.2 * len(fit)))
            inner_val, inner_fit = inner[:nval], inner[nval:]
            assert not set(fit) & set(val)
            assert set(inner_fit) | set(inner_val) == set(fit)
            assert not set(inner_fit) & set(inner_val)
            splits[method].append({key: [names[i] for i in ids] for key, ids in
                                   [('train', fit), ('validation', val), ('inner_train', inner_fit), ('inner_validation', inner_val)]})
            head, optimizer, rng = initialize(features)
            best, stale, history = -1, 0, []
            for epoch in range(model.CONFIG['max_epochs']):
                train_epoch(head, optimizer, rng, features, targets, inner_fit)
                prob = predict(head, features, inner_val)
                measured = model.metrics(prob, targets[inner_val].numpy(), .5)['miou']
                history.append(measured)
                if measured > best:
                    best, stale, selected_epoch = measured, 0, epoch + 1
                    state = copy.deepcopy(head.state_dict())
                else:
                    stale += 1
                if stale >= model.CONFIG['patience']:
                    break
            head.load_state_dict(state)
            prob = predict(head, features, inner_val)
            threshold = max(model.CONFIG['threshold_candidates'],
                            key=lambda t: model.metrics(prob, targets[inner_val].numpy(), t)['miou'])
            head, optimizer, rng = initialize(features)
            for _ in range(selected_epoch):
                train_epoch(head, optimizer, rng, features, targets, fit)
            prob = predict(head, features, val)
            measured = model.metrics(prob, targets[val].numpy(), threshold)
            fixed = model.metrics(prob, targets[val].numpy(), .5)
            oof[val] = prob
            thresholds[val] = threshold
            seen[val] += 1
            global_scores[fold].extend(measured['per_image'])
            global_fixed[fold].extend(fixed['per_image'])
            row = dict(fold=fold + 1, train_n=len(fit), validation_n=len(val),
                       epoch=selected_epoch, threshold=threshold, selected=measured, fixed=fixed,
                       inner_history=history, seconds=time.perf_counter() - start)
            rows.append(row)
            torch.save(dict(state_dict=head.state_dict(), epoch=selected_epoch, threshold=threshold,
                            channels=[f.shape[1] for f in features], config=model.CONFIG),
                       OUT / f'{method}_fold{fold + 1}_head.pt')
            print(f'{method} fold {fold + 1}: mIoU={measured["miou"]:.6f}, fixed={fixed["miou"]:.6f}, epoch={selected_epoch}, threshold={threshold}', flush=True)
        assert np.all(seen == 1)
        assert np.isfinite(oof).all()
        np.savez_compressed(OUT / f'{method}_oof.npz', probability=oof,
                            target=targets.numpy(), names=names, threshold=thresholds)
        oof_score = model.score(oof >= thresholds[:, None, None], targets.numpy())
        results['methods'][method] = dict(folds=rows, stability=summarize([r['selected']['miou'] for r in rows]),
                                         oof_miou=oof_score)
    results['overall'] = dict(fold_miou=[float(np.mean(v)) for v in global_scores],
                              fixed_fold_miou=[float(np.mean(v)) for v in global_fixed],
                              oof_miou=float(np.mean(sum(global_scores, []))))
    results['overall']['stability'] = summarize(results['overall']['fold_miou'])
    (OUT / 'folds.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    lines = ['# Frozen extractors: 3-fold CV', '',
             '510 training images; each image evaluated exactly once outside training. Seed 42.',
             'Independent heads refitted from scratch on 113/114 images per method; validation 57/56.',
             'Backbones frozen; existing label-independent feature caches reused.',
             'Epoch (max 50, patience 8) and threshold selected on an inner 80/20 split, then head refitted on all outer training images.', '',
             '| Method | Fold 1 | Fold 2 | Fold 3 | Mean | SD (sample) | OOF |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method, data in results['methods'].items():
        values = [r['selected']['miou'] for r in data['folds']]
        lines.append('| ' + method + ' | ' + ' | '.join(f'{v:.6f}' for v in values +
                     [data['stability']['mean'], data['stability']['sd'], data['oof_miou']]) + ' |')
    data = results['overall']
    lines.append('| Overall | ' + ' | '.join(f'{v:.6f}' for v in data['fold_miou'] +
                 [data['stability']['mean'], data['stability']['sd'], data['oof_miou']]) + ' |')
    lines += ['', 'Metric: official per-image binary mIoU, averaged across images. OOF weights all images equally; fold mean weights folds equally.',
              'SD describes variation across three data splits, not random-seed stability or a confidence interval.',
              'No exact duplicate images found. Source grouping metadata unavailable; shared-source leakage cannot be excluded.',
              'Architecture was previously explored on this dataset; this is not a new untouched test set.',
              'No test data or leaderboard used. Fixed-threshold scores, inner histories, checkpoints, split names and OOF probabilities are in results/cv3/.', '',
              'Reproduce from repo root: `task-1/.venv/Scripts/python.exe task-1/experiments/frozen_extractor/cross_validate.py`.']
    (HERE / 'CV3_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(results['overall'], indent=2), flush=True)


if __name__ == '__main__':
    main()
