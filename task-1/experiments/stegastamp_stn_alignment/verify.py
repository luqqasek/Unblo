"""Reproduce all new checkpoints, verify selection/coverage, and preserve source artifacts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('alignment_runner', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
OUT, cv = runner.OUT, runner.cv


def main():
    torch.set_num_threads(4)
    config = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
    assert config['runner_sha256'] == runner.digest(HERE / 'run.py')
    for path, expected in config['sources_sha256'].items():
        assert runner.digest(cv.model.ROOT / path) == expected, path
    data = json.loads((OUT / 'metrics.json').read_text())
    assert data['status'] == 'complete'
    assert {(r['fold'], r['seed']) for r in data['runs']} == {(f, s) for f in (1, 2, 3) for s in (42, 7, 123)}
    splits = json.loads((OUT / 'splits.json').read_text())
    assert splits == json.loads((runner.BASE / 'splits.json').read_text())
    aligned = torch.load(OUT / 'aligned_features.pt', weights_only=True)
    assert aligned['config_sha256'] == data['config_sha256']
    source = np.load(cv.OUT / 'stegastamp_oof.npz')
    names = source['names'].tolist()
    assert names == aligned['names']
    rows = pd.read_csv(OUT / 'per_image_metrics.csv')
    assert len(rows) == 1020 and rows.image_id.nunique() == 170
    summary = json.loads((OUT / 'summary.json').read_text())
    max_error = 0.
    for run in data['runs']:
        stem = f"stegastamp_fold{run['fold']}_seed{run['seed']}"
        checkpoint = torch.load(OUT / f'{stem}.pt', weights_only=True)
        split = splits[run['fold'] - 1]
        assert checkpoint['train_names'] == split['train']
        assert checkpoint['validation_names'] == split['validation'] == run['names']
        assert not set(checkpoint['train_names']) & set(checkpoint['validation_names'])
        inner = sum(split['inner_validation'], [])
        assert sorted(inner) == sorted(split['train']) and len(inner) == len(set(inner))
        assert run['epoch'] == np.argmax(run['inner_history']) + 1 == checkpoint['epoch']
        chosen = max(run['inner_threshold_scores'], key=lambda r: r['miou'])['threshold']
        assert run['threshold'] == chosen == checkpoint['threshold']
        assert checkpoint['config_sha256'] == run['config_sha256'] == data['config_sha256']
        raw_ck = torch.load(runner.BASE / f'{stem}.pt', weights_only=True)
        assert raw_ck['state_dict'].keys() == checkpoint['state_dict'].keys()
        assert all(raw_ck['state_dict'][k].shape == v.shape for k, v in checkpoint['state_dict'].items())
        head = cv.model.Head(checkpoint['channels'])
        head.load_state_dict(checkpoint['state_dict'])
        ids = [names.index(n) for n in run['names']]
        prob = cv.predict(head, aligned['features'], ids)
        saved = np.load(OUT / f'{stem}.npz')
        assert saved['names'].tolist() == run['names']
        np.testing.assert_allclose(prob, saved['probability'], atol=1e-7, rtol=0)
        max_error = max(max_error, float(np.max(np.abs(prob - saved['probability']))))
        measured = cv.model.metrics(prob, source['target'][ids], chosen)
        np.testing.assert_allclose(measured['per_image'], run['per_image'], atol=1e-12)
        assert abs(measured['miou'] - run['miou']) < 1e-12
        metrics = rows[(rows.seed == run['seed']) & (rows.fold == run['fold']) & (rows.variant == 'aligned')].set_index('image_id').loc[run['names']]
        np.testing.assert_allclose(metrics.miou, measured['per_image'], atol=1e-12)
        print(stem, 'verified', flush=True)
    for seed in (42, 7, 123):
        runs = [r for r in data['runs'] if r['seed'] == seed]
        assert sorted(sum([r['names'] for r in runs], [])) == sorted(names)
        record = next(r for r in summary['seeds'] if r['seed'] == seed)
        assert abs(record['metrics']['miou']['aligned'] - np.mean([x for r in runs for x in r['per_image']])) < 1e-12
    for metric in summary['metrics']:
        for variant in ('raw', 'aligned'):
            assert abs(rows[rows.variant == variant][metric].mean() - summary['metrics'][metric][variant]) < 1e-12
    assert all(np.isfinite(rows[m]).all() for m in ('miou', 'auc', 'ap', 'fg_iou', 'boundary_f1'))
    result = dict(status='passed', new_checkpoints=9, baseline_checkpoints_verified_in_preflight=9,
                  unique_images=170, compared_variants=2, seeds=[42, 7, 123], max_probability_replay_error=max_error,
                  same_head_shapes=True, same_inner_outer_splits=True, inner_only_selection_verified=True,
                  summary_reconciled=True, original_sources_unchanged=True, geometry_ramp_max_error=runner.geometry_checks())
    runner.save(OUT / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
