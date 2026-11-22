"""Fixed empty-mask corrections using strong cached local evidence; CPU only."""
import json
from pathlib import Path

import numpy as np

from experiments.native_localizer.evaluate import image_metrics
from experiments.pixelseal_regions.evaluate import summarize
from experiments.pixelseal_regions.geometry import candidates, draw


def sums(field, boxes):
    ii = np.pad(field.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    x0, y0, x1, y1 = boxes.T
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def rectangle(local, boxes):
    outer = boxes + np.array([-3, -3, 3, 3])
    outer = outer.clip(0, 128)
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    oa = np.prod(outer[:, 2:] - outer[:, :2], axis=1)
    inside = sums(local, boxes)
    ring = (sums(local, outer) - inside) / np.maximum(oa - area, 1)
    score = (inside / area - ring) * np.sqrt(area)
    i = int(score.argmax())
    return boxes[i], float(score[i])


def main():
    here = Path(__file__).parent
    out = here / 'results'
    out.mkdir(exist_ok=True)
    source = here.parent / 'pcmv/results/base-candidate/all-real-maps.npz'
    split = json.loads((here.parent / 'pixelseal_regions/results/split.json').read_text())
    z = np.load(source)
    valid = set(split['validation'])
    boxes = candidates(128, 128, step=2)
    before, per_image = [], {'local_top164': [], 'local_ring_rectangle': []}
    predictions, proposals, targets, names = {k: [] for k in per_image}, [], [], []
    for name, p, local, y in zip(z['names'], z['probabilities'], z['local_logits'], z['targets']):
        if str(name) not in valid:
            continue
        base = dict(image_metrics(p, y), image_name=str(name))
        before.append(base)
        names.append(str(name))
        targets.append(y)
        eligible = not (p >= .5).any()
        box, score = rectangle(local, boxes) if eligible else (None, None)
        proposals.append({'name': str(name), 'eligible': bool(eligible), 'box': box.tolist() if box is not None else None, 'score': score})
        for kind in per_image:
            result = p.copy()
            if eligible:
                mask = np.zeros((128, 128), bool)
                if kind == 'local_top164':
                    mask.flat[np.argsort(local.ravel(), kind='stable')[-164:]] = True
                else:
                    mask = draw(box)
                result[mask] = 1
            per_image[kind].append(dict(image_metrics(result, y), image_name=str(name)))
            predictions[kind].append(result)
    baseline = summarize(before)
    result = {'baseline': baseline, 'baseline_per_image': before, 'proposals': proposals,
              'protocol': 'two predeclared fixed corrections; one reused57-image PixelSeal fold; no fitting; empty base masks only; no GT inference', 'candidates': {}}
    for key, rows in per_image.items():
        delta = np.array([b['miou']-a['miou'] for a, b in zip(before, rows)])
        rng = np.random.default_rng(42)
        ci = np.quantile(delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(1), [.025, .975])
        summary = summarize(rows)
        passed = delta.mean() >= .003 and ci[0] > 0 and summary['wm_over50']['clean_iou'] >= baseline['wm_over50']['clean_iou']-.01
        result['candidates'][key] = {'summary': summary, 'delta': float(delta.mean()), 'ci95': ci.tolist(),
                                     'passed': bool(passed), 'per_image': rows,
                                     'improved': int((delta > 1e-8).sum()), 'worse': int((delta < -1e-8).sum())}
        np.savez_compressed(out / f'{key}.npz', names=names, probabilities=predictions[key], targets=targets)
    passing = [k for k, r in result['candidates'].items() if r['passed']]
    result['selected'] = max(passing, key=lambda k: result['candidates'][k]['summary']['all']['miou']) if passing else None
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'selected': result['selected'], 'eligible': sum(r['eligible'] for r in proposals),
                      'candidates': {k: {s: r[s] for s in ('delta', 'ci95', 'passed', 'improved', 'worse')} for k, r in result['candidates'].items()}}, indent=2))


if __name__ == '__main__':
    main()
