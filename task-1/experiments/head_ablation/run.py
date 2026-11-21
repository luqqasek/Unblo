"""Screen symmetric Dice and native-detail 128 heads; CV only after a preset gate."""
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'frozen_extractor'))
import cross_validate as cv

OUT = HERE / 'results'


class Head128(cv.model.Head):
    def forward(self, features):
        x = torch.cat([F.interpolate(p(f), size=(128, 128), mode='bilinear', align_corners=False)
                       for p, f in zip(self.projections, features)], 1)
        return self.fuse(x)[:, 0]


def dice_loss(p, target):
    return 1 - ((2 * (p * target).sum((1, 2)) + 1) /
                (p.sum((1, 2)) + target.sum((1, 2)) + 1)).mean()


def train_epoch(head, optimizer, rng, features, y, ids, symmetric, loss_fn=None):
    head.train()
    shuffled = rng.permutation(ids)
    for j in range(0, len(ids), 8):
        batch = shuffled[j:j+8]
        optimizer.zero_grad()
        logits = head([f[batch] for f in features])
        target = y[batch]
        p = logits.sigmoid()
        dice = dice_loss(p, target)
        if symmetric:
            dice = .5 * (dice + dice_loss(1-p, 1-target))
        loss = (.5 * F.binary_cross_entropy_with_logits(logits, target) + .5 * dice
                if loss_fn is None else loss_fn(logits, target))
        loss.backward()
        optimizer.step()


def fit(variant, method, fold, ds, splits, loss_fn=None):
    path = OUT / variant / f'{method}_fold{fold+1}.json'
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    high = variant == 'decoder128'
    cache = cv.model.extract(method, ds, cache_dir=OUT/'features128', spatial_limit=128) if high else cv.model.extract(method, ds)
    names = cache['names']
    features = cache['features']
    y = torch.from_numpy(np.stack([ds.m128[n] for n in names])).float()
    split = splits[method][fold]
    ids = {key: np.array([names.index(n) for n in values]) for key, values in split.items()}
    assert not set(ids['train']) & set(ids['validation'])
    assert set(ids['inner_train']) | set(ids['inner_validation']) == set(ids['train'])
    assert not set(ids['inner_train']) & set(ids['inner_validation'])
    cls = Head128 if high else cv.model.Head
    def initialize():
        torch.manual_seed(42)
        head = cls([f.shape[1] for f in features])
        return head, torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001), np.random.default_rng(42)
    head, opt, rng = initialize()
    best, stale, history = -1, 0, []
    start = time.perf_counter()
    for epoch in range(50):
        train_epoch(head, opt, rng, features, y, ids['inner_train'], not high, loss_fn)
        prob = cv.predict(head, features, ids['inner_validation'])
        value = cv.model.metrics(prob, y[ids['inner_validation']].numpy(), .5)['miou']
        history.append(value)
        if value > best:
            best, stale, chosen = value, 0, epoch+1
            state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
        print(variant, method, fold+1, 'inner epoch', epoch+1, round(value, 5), flush=True)
        if stale >= 8:
            break
    head.load_state_dict(state)
    prob = cv.predict(head, features, ids['inner_validation'])
    threshold = max(cv.model.CONFIG['threshold_candidates'], key=lambda t: cv.model.metrics(prob, y[ids['inner_validation']].numpy(), t)['miou'])
    head, opt, rng = initialize()
    for epoch in range(chosen):
        train_epoch(head, opt, rng, features, y, ids['train'], not high, loss_fn)
        if high:
            print(variant, method, fold+1, 'refit', epoch+1, '/', chosen, flush=True)
    prob = cv.predict(head, features, ids['validation'])
    report = dict(variant=variant, method=method, fold=fold+1, epoch=chosen, threshold=threshold,
                  selected=cv.model.metrics(prob, y[ids['validation']].numpy(), threshold),
                  fixed=cv.model.metrics(prob, y[ids['validation']].numpy(), .5),
                  history=history, seconds=time.perf_counter()-start,
                  feature_shapes=[list(f.shape) for f in features], train_names=split['train'], val_names=split['validation'])
    checkpoint = path.with_suffix('.pt')
    torch.save(dict(state_dict=head.state_dict(), channels=[f.shape[1] for f in features], epoch=chosen, threshold=threshold, variant=variant), checkpoint)
    restored = cls([f.shape[1] for f in features])
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict'])
    np.testing.assert_allclose(cv.predict(restored, features, ids['validation']), prob, atol=1e-7)
    np.savez_compressed(path.with_suffix('.npz'), probability=prob, target=y[ids['validation']].numpy(), names=split['validation'])
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULT', variant, method, fold+1, report['selected']['miou'], flush=True)
    return report


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = cv.model.LocalisationDataset(cv.model.ROOT/'data', 'train')
    splits = json.loads((cv.OUT/'folds.json').read_text())
    baseline = json.loads((cv.OUT/'metrics.json').read_text())
    summary = dict(gate='First-fold overall delta >= 0.005 and improvement in at least two methods', variants={})
    for variant in ('symmetric_dice', 'decoder128'):
        rows = [fit(variant, m, 0, ds, splits) for m in cv.METHODS]
        differences = [r['selected']['miou']-baseline['methods'][r['method']]['folds'][0]['selected']['miou'] for r in rows]
        passed = bool(np.mean(differences) >= .005 and sum(d>0 for d in differences) >= 2)
        summary['variants'][variant] = dict(screen=rows.copy(), screen_delta=float(np.mean(differences)), passed=passed, cv=[])
        (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('GATE', variant, passed, differences, flush=True)
        if passed:
            for fold in (1, 2):
                rows.extend(fit(variant, m, fold, ds, splits) for m in cv.METHODS)
            summary['variants'][variant]['cv'] = rows
        (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
