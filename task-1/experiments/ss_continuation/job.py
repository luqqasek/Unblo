"""Two workers, one fixed-fold selection, optional frozen test inference."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.native_localizer.common import atomic_json, sha256
from experiments.native_localizer.evaluate import image_metrics
from experiments.next_ideas.audit import fallback
from experiments.triforensic_wm.evaluate import summarize


def score(probabilities, targets, names):
    rows = []
    for p, y, name in zip(probabilities, targets, names):
        row = image_metrics(p, y)
        scored = image_metrics(fallback(p, 'top1pct').astype(float), y)
        for key in ('miou', 'fg_iou', 'clean_iou', 'empty', 'full', 'pred_area',
                    'complete_fg_miss', 'fp_area', 'internal_boundary_f1', 'minority_iou'):
            if key in scored:
                row[key] = scored[key]
        rows.append(dict(row, image_name=str(name), family='stegastamp', fold=1))
    return rows


def select(root, out):
    protocol = json.loads(Path('experiments/ss_continuation/protocol.json').read_text())
    parent = root / 'next-ideas-extended-runs/stegastamp/fold1/maps-450.npz'
    with np.load(parent) as z0, np.load(out / 'stegastamp/fold1/maps-1000.npz') as z:
        assert np.array_equal(z0['names'], z['names'])
        assert np.array_equal(z0['targets'], z['targets'])
        p0, p, y, names = z0['probabilities'], z['probabilities'], z['targets'], z['names']
    before = score(p0, y, names)
    baseline = summarize(before)
    candidates = {}
    for name, probabilities in [('raw1000', p), ('equal_probability_450_1000', (p0 + p) / 2)]:
        rows = score(probabilities, y, names)
        summary = summarize(rows)
        delta = np.array([b['miou'] - a['miou'] for a, b in zip(before, rows)])
        rng = np.random.default_rng(42)
        ci = np.quantile(delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(1), [.025, .975])
        small0 = np.mean([r['fg_iou'] for r in before if 0 < r['true_area'] <= .05])
        small = np.mean([r['fg_iou'] for r in rows if 0 < r['true_area'] <= .05])
        guard = small >= small0 - .01 and summary['wm_over50']['clean_iou'] >= baseline['wm_over50']['clean_iou'] - .01
        passed = delta.mean() >= .005 and ci[0] > 0 and guard
        candidates[name] = {'summary': summary, 'per_image': rows,
                            'delta': float(delta.mean()), 'ci95': ci.tolist(),
                            'improved': int((delta > 1e-8).sum()), 'worse': int((delta < -1e-8).sum()),
                            'minority_guard_passed': bool(guard), 'passed': bool(passed)}
        np.savez_compressed(out / f'maps-{name}.npz', names=names, probabilities=probabilities, targets=y)
    passing = [name for name, r in candidates.items() if r['passed']]
    selected = max(passing, key=lambda name: candidates[name]['summary']['all']['miou']) if passing else None
    result = {'protocol': protocol, 'baseline': baseline, 'baseline_per_image': before,
              'candidates': candidates, 'selected': selected, 'public_score_used': False}
    atomic_json(out / 'evaluation.json', result)
    print('SELECTION', json.dumps({'baseline_miou': baseline['all']['miou'], 'selected': selected,
                                  'candidates': {k: {s: v[s] for s in ('delta', 'ci95', 'passed')} for k, v in candidates.items()}}), flush=True)
    if selected:
        recipe = {'recipe_frozen': True, 'family': 'stegastamp', 'candidate': selected,
                  'checkpoint': str(out / 'stegastamp/all170/last.pt'),
                  'parent_checkpoint': str(root / 'next-ideas-extended-runs/stegastamp/all170/last.pt'),
                  'updates': 1000, 'threshold': .5, 'tta': False, 'postprocess': 'top1pct',
                  'base_submission_id': 3188, 'evidence_sha256': sha256(out / 'evaluation.json'),
                  'validation_miou': candidates[selected]['summary']['all']['miou'],
                  'protocol': 'one reused57-image development fold, not full OOF', 'test_labels_used': False}
        atomic_json(out / 'selected_recipe.json', recipe)
    return selected


def main():
    root = Path('/p/project1/training2666/pszenny1/native-localizer')
    out = root / 'ss1000-screen-runs'
    out.mkdir(exist_ok=True)
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(devices) == 2
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    workers, handles = [], []
    start = time.monotonic()
    for gpu, name, args in [(0, 'fold1', ['--fold', '1']), (1, 'all170', ['--all-real'])]:
        run = out / 'stegastamp' / name
        if (run / 'completed.json').exists():
            assert json.loads((run / 'completed.json').read_text())['step'] == 1000
            continue
        if (run / 'last.pt').exists():
            args += ['--resume']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[gpu], OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6')
        if gpu == 1:
            # A separately selected CPU screen uses this GPU briefly before refit.
            # The fold worker is already running on GPU0; never exceed two GPUs.
            with (logs / 'carrier-fallback.log').open('a') as pslog:
                ps = subprocess.run([sys.executable, '-u', '-m', 'experiments.carrier_fallback.infer',
                                     '--root', str(root)], env=env, stdout=pslog, stderr=subprocess.STDOUT)
            print('CARRIER_FALLBACK_EXIT', ps.returncode, flush=True)
        log = (logs / f'{name}.log').open('a')
        handles.append(log)
        workers.append(subprocess.Popen([sys.executable, '-u', '-m', 'experiments.ss_continuation.train',
                                         '--root', str(root), '--continue450', '--steps', '1000',
                                         '--wall-seconds', '1550', *args], env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in workers]
    for h in handles:
        h.close()
    assert not any(codes), codes
    for name in ('fold1', 'all170'):
        assert json.loads((out / 'stegastamp' / name / 'completed.json').read_text())['step'] == 1000
    selected = select(root, out)
    if selected:
        workers, handles = [], []
        for part in range(2):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[part])
            log = (logs / f'inference{part}.log').open('a')
            handles.append(log)
            workers.append(subprocess.Popen([sys.executable, '-u', '-m', 'experiments.ss_continuation.infer',
                                             '--root', str(root), '--part', str(part)], env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in workers]
        for h in handles:
            h.close()
        assert not any(codes), codes
    atomic_json(out / 'completed.json', {'selected': selected, 'seconds': time.monotonic() - start,
                                        'job_id': os.environ.get('SLURM_JOB_ID'), 'gpu_workers': 2})


if __name__ == '__main__':
    main()
