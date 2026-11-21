"""Matched control vs 2k synthetic pretrain then nested real finetuning. No test access."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
BASE = HERE.parent / 'stegastamp_stn_alignment/results'
spec = importlib.util.spec_from_file_location('synthetic_stable', HERE.parent / 'stegastamp_stable_selection/run.py')
stable = importlib.util.module_from_spec(spec); spec.loader.exec_module(stable)
cv = stable.cv
DEVICE = None

def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def initialize(features, seed, state=None):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    head = cv.model.Head([f.shape[1] for f in features]).to(DEVICE)
    if state is not None:
        head.load_state_dict(state)
    return head, torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001), np.random.default_rng(seed)

def predict(head, features, ids):
    head.eval()
    with torch.inference_mode():
        return torch.cat([head([f[ids[j:j + 8]] for f in features]).sigmoid().cpu() for j in range(0, len(ids), 8)]).numpy()

def train_epoch(head, opt, rng, features, target, ids):
    head.train(); total = 0.
    order = rng.permutation(ids)
    for start in range(0, len(order), 8):
        batch = order[start:start + 8]
        opt.zero_grad(set_to_none=True)
        logits = head([f[batch] for f in features]); y = target[batch]
        p = logits.sigmoid()
        dice = 1 - ((2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)).mean()
        loss = .5 * F.binary_cross_entropy_with_logits(logits, y) + .5 * dice
        loss.backward(); opt.step(); total += float(loss.detach()) * len(batch)
    return total / len(order)

def real_data():
    cached = torch.load(BASE / 'aligned_features.pt', weights_only=True)
    names = cached['names']; features = [f.to(DEVICE) for f in cached['features']]
    z = np.load(cv.OUT / 'stegastamp_oof.npz')
    assert names == z['names'].tolist()
    gt = z['target'].astype(np.uint8)
    splits = json.loads((BASE / 'splits.json').read_text())
    for split in splits:
        assert not set(split['train']) & set(split['validation'])
        assert sorted(sum(split['inner_validation'], [])) == sorted(split['train'])
    return names, features, gt, splits

def pretrain(seed):
    path = OUT / f'pretrain_seed{seed}.pt'
    if path.exists():
        return torch.load(path, weights_only=True)['state_dict']
    chunks = [torch.load(p, weights_only=True) for p in sorted((OUT / 'shards').glob('*.pt'))]
    assert len(chunks) == 22
    names = sum([c['names'] for c in chunks], [])
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert names == [r['image_id'] for r in manifest]
    assert len({r['coco_id'] for r in manifest}) == 2200
    features = [torch.cat([c['features'][i] for c in chunks]).to(DEVICE) for i in range(4)]
    target = torch.cat([c['target'] for c in chunks]).float().to(DEVICE)
    del chunks
    head, opt, rng = initialize(features, seed)
    start = time.perf_counter(); history = []
    train_ids = np.arange(2000); val_ids = np.arange(2000, 2200)
    val_gt = target[val_ids].cpu().numpy()
    for epoch in range(1, 11):
        loss = train_epoch(head, opt, rng, features, target, train_ids)
        p = predict(head, features, val_ids)
        measured = cv.model.metrics(p, val_gt, .5)
        history.append(dict(epoch=epoch, loss=loss, validation_miou=measured['miou'], validation_fg_iou=measured['iou_watermark']))
        print(f'PRETRAIN seed={seed} epoch={epoch}/10 loss={loss:.5f} synth_val={measured["miou"]:.5f}', flush=True)
        save(OUT / f'pretrain_seed{seed}_history.json', history)
    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    torch.save(dict(state_dict=state, seed=seed, epochs=10, train_n=2000, channels=[f.shape[1] for f in features],
                    config_sha256=digest(OUT / 'config.json'), generation_sha256=digest(OUT / 'generation_config.json')), path)
    np.savez_compressed(OUT / f'pretrain_seed{seed}_synthetic_validation.npz', probability=p, target=val_gt, names=names[2000:])
    save(OUT / f'pretrain_seed{seed}.json', dict(seed=seed, epochs=10, seconds=time.perf_counter() - start, history=history,
         validation=measured, protocol='fixed 10 epochs; held-out synthetic validation NOT used for selection'))
    return state

def fit(variant, seed, split, names, features, gt, state):
    stem = f'{variant}_fold{split["fold"]}_seed{seed}'
    path = OUT / f'{stem}.json'
    if path.exists():
        return json.loads(path.read_text())
    start = time.perf_counter()
    fit_ids = np.array([names.index(n) for n in split['train']])
    val = np.array([names.index(n) for n in split['validation']])
    parts = [np.array([names.index(n) for n in p]) for p in split['inner_validation']]
    inner_fit = [np.setdiff1d(fit_ids, p) for p in parts]
    models = [initialize(features, seed, state) for _ in parts]
    target = torch.from_numpy(gt).float().to(DEVICE)
    best, stale, history = -1, 0, []
    inner_gt = gt[np.concatenate(parts)]
    for epoch in range(1, 51):
        probabilities = []; losses = []
        for (head, opt, rng), train_ids, valid_ids in zip(models, inner_fit, parts):
            assert not set(train_ids) & set(valid_ids) and not set(train_ids) & set(val)
            losses.append(train_epoch(head, opt, rng, features, target, train_ids))
            probabilities.append(predict(head, features, valid_ids))
        p = np.concatenate(probabilities)
        score = cv.model.metrics(p, inner_gt, .5)['miou']
        history.append(dict(epoch=epoch, miou=score, loss=float(np.mean(losses))))
        if score > best:
            best, stale, selected_epoch = score, 0, epoch; best_prob = p.copy()
        else:
            stale += 1
        if epoch % 5 == 0:
            print(f'{stem} inner epoch={epoch} miou={score:.5f}', flush=True)
        if stale >= 8:
            break
    del models
    threshold_scores = [dict(threshold=t, miou=cv.model.metrics(best_prob, inner_gt, t)['miou']) for t in cv.model.CONFIG['threshold_candidates']]
    threshold = max(threshold_scores, key=lambda r: r['miou'])['threshold']
    head, opt, rng = initialize(features, seed, state)
    for _ in range(selected_epoch):
        train_epoch(head, opt, rng, features, target, fit_ids)
    probability = predict(head, features, val)
    measured = cv.model.metrics(probability, gt[val], threshold)
    state_out = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    ck = dict(state_dict=state_out, seed=seed, epoch=selected_epoch, threshold=threshold, channels=[f.shape[1] for f in features],
              train_names=split['train'], validation_names=split['validation'], config_sha256=digest(OUT / 'config.json'),
              pretrain_sha256=digest(OUT / f'pretrain_seed{seed}.pt') if state is not None else None)
    torch.save(ck, OUT / f'{stem}.pt')
    restored, _, _ = initialize(features, seed, torch.load(OUT / f'{stem}.pt', weights_only=True)['state_dict'])
    np.testing.assert_allclose(predict(restored, features, val), probability, atol=1e-7, rtol=0)
    np.savez_compressed(OUT / f'{stem}.npz', names=split['validation'], probability=probability, target=gt[val], threshold=threshold)
    historical = json.loads((BASE / f'stegastamp_fold{split["fold"]}_seed{seed}.json').read_text())
    result = dict(variant=variant, seed=seed, fold=split['fold'], epoch=selected_epoch, threshold=threshold,
                  miou=measured['miou'], iou_watermark=measured['iou_watermark'], iou_clean=measured['iou_clean'],
                  per_image=measured['per_image'], names=split['validation'], inner_history=history,
                  inner_threshold_scores=threshold_scores, fixed_half_miou=cv.model.metrics(probability, gt[val], .5)['miou'],
                  historical_cpu_miou=historical['miou'], seconds=time.perf_counter() - start,
                  config_sha256=ck['config_sha256'])
    save(path, result)
    print(f'COMPLETE {stem}: mIoU={measured["miou"]:.6f} FG={measured["iou_watermark"]:.6f} epoch={selected_epoch} t={threshold} seconds={result["seconds"]:.1f}', flush=True)
    return result

def main():
    global DEVICE
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', choices=['control', 'synthetic', 'all'], default='all')
    ap.add_argument('--device', default='cuda:1'); args = ap.parse_args(); DEVICE = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [BASE / 'aligned_features.pt', BASE / 'splits.json', HERE / 'run.py', HERE / 'README.md',
               cv.HERE / 'run.py', cv.HERE / 'cross_validate.py', cv.model.ROOT / 'wmloc/watermarks.py', cv.model.ROOT / 'wmloc/utils.py']
    sources += sorted(BASE.glob('stegastamp_fold*.pt'))
    config = dict(seeds=[42, 7, 123], device=str(DEVICE), device_name=torch.cuda.get_device_name(DEVICE),
                  torch=str(torch.__version__), cuda=torch.version.cuda, deterministic=True, tf32=False,
                  synthetic_train_n=2000, synthetic_validation_n=200, pretrain_epochs=10,
                  finetune=cv.model.CONFIG, selection='unchanged three inner folds; fresh refit from same pretrain; reset optimizer',
                  variants=['control', 'synthetic'], sources_sha256={str(p.relative_to(cv.model.ROOT)): digest(p) for p in sources})
    if (OUT / 'config.json').exists():
        assert json.loads((OUT / 'config.json').read_text()) == config, 'Configuration changed; no stale resume.'
    else:
        save(OUT / 'config.json', config)
    names, features, gt, splits = real_data()
    save(OUT / 'splits.json', splits)
    for variant in (['control', 'synthetic'] if args.variant == 'all' else [args.variant]):
        for seed in (42, 7, 123):
            state = pretrain(seed) if variant == 'synthetic' else None
            if state is not None and not (OUT / f'synthetic_only_seed{seed}.npz').exists():
                head, _, _ = initialize(features, seed, state)
                p = predict(head, features, np.arange(len(names)))
                np.savez_compressed(OUT / f'synthetic_only_seed{seed}.npz', names=names, probability=p, target=gt, threshold=.5)
            for split in splits:
                fit(variant, seed, split, names, features, gt, state)
    assert all(digest(cv.model.ROOT / p) == sha for p, sha in config['sources_sha256'].items())
    print('REQUESTED RUNS COMPLETE; source hashes unchanged.', flush=True)

if __name__ == '__main__':
    main()
