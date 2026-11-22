"""Selected fixed PixelSeal fallback; reproduce cached carrier before test inference."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.native_localizer.common import atomic_json, rgb_tensor, sha256
from experiments.native_localizer.evaluate import image_metrics
from experiments.pixelseal_regions.evaluate import summarize
from experiments.triforensic_wm.model import Carrier, observability
from wmloc.data import LocalisationDataset
from watermark_localization.submission import save_submission, validate_submission


def mask_from_local(local):
    mask = np.zeros((128, 128), np.uint8)
    mask.flat[np.argsort(local.ravel(), kind='stable')[-164:]] = 1
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    out = a.root / 'carrier-fallback-runs'
    out.mkdir(exist_ok=True)
    if (out / 'completed.json').exists():
        return
    evidence_path = Path('experiments/carrier_fallback/results/evaluation.json')
    evidence = json.loads(evidence_path.read_text())
    assert evidence['selected'] == 'local_top164'
    assert evidence['candidates']['local_top164']['passed']
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = a.root / 'triforensic-runs/pixelseal/unified-v1/resume.pt'
    source_hash = sha256(source)
    parent_evidence = json.loads((a.root / 'pcmv-runs/base-candidate/evaluation.json').read_text())
    assert source_hash == parent_evidence['checkpoint_sha256']
    state = torch.load(source, map_location='cpu', weights_only=False)
    assert state['step'] == 1561
    weights = state['ema']
    carrier = Carrier(checkpointing=False).cuda().eval().requires_grad_(False)
    carrier.load_state_dict({k[len('carrier.'):]: v for k, v in weights.items() if k.startswith('carrier.')}, strict=True)
    sys.path.insert(0, str(Path('checkpoints/videoseal').resolve()))
    from videoseal.modules.jnd import JND
    jnd = JND(in_channels=1, out_channels=1).cuda().eval().requires_grad_(False)
    jnd.load_state_dict({k[len('base.encoder.jnd.'):]: v for k, v in weights.items() if k.startswith('base.encoder.jnd.')}, strict=True)
    del state, weights
    started = time.monotonic()

    @torch.inference_mode()
    def forward(items):
        x = torch.stack([rgb_tensor(it['image']) for it in items]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            obs = observability(x, jnd)
            _, local = carrier(x, obs)
        return local[:, 0].float().cpu().numpy()

    cached = np.load(a.root / 'pcmv-runs/base-candidate/all-real-maps.npz')
    old = {str(n): (p, local) for n, p, local in zip(cached['names'], cached['probabilities'], cached['local_logits'])}
    valid = {r['image_name'] for r in evidence['baseline_per_image']}
    train = LocalisationDataset('data', 'train', methods=['pixelseal'])
    # Match the per-device batch lengths used by the archived two-GPU producer.
    errors, masks_equal, actual = [], [], {}
    for begin in range(0, len(train), 16):
        items = [train[i] for i in range(begin, min(begin+16, len(train)))]
        cut = (len(items)+1)//2
        for batch in (items[:cut], items[cut:]):
            if not any(it['image_name'] in valid and not (old[it['image_name']][0] >= .5).any() for it in batch):
                continue
            logits = forward(batch)
            for it, local in zip(batch, logits):
                name = it['image_name']
                if name in valid and not (old[name][0] >= .5).any():
                    expected = old[name][1]
                    errors.append(float(np.max(np.abs(local-expected))))
                    masks_equal.append(bool(np.array_equal(mask_from_local(local), mask_from_local(expected))))
                    actual[name] = local
    reproduction = {'eligible_images': len(errors), 'max_abs_logit_difference': max(errors), 'all_fallback_masks_identical': all(masks_equal)}
    atomic_json(out / 'reproduction.json', reproduction)
    assert len(errors) == 9, reproduction
    # BF16 kernels can perturb tied local rankings. Validate the actual deployment
    # subgraph on the same fold; do not call non-identical masks a reproduction pass.
    real = {it['image_name']: it for it in train}
    before = evidence['baseline_per_image']
    rows, maps, names, targets = [], [], [], []
    for oldrow in before:
        name = oldrow['image_name']
        p = old[name][0].copy()
        if name in actual:
            p[mask_from_local(actual[name]).astype(bool)] = 1
        y = real[name]['mask128']
        rows.append(dict(image_metrics(p, y), image_name=name))
        names.append(name);maps.append(p);targets.append(y)
    delta = np.array([b['miou']-a['miou'] for a, b in zip(before, rows)])
    rng = np.random.default_rng(42)
    ci = np.quantile(delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(1), [.025, .975])
    summary = summarize(rows)
    passed = delta.mean() >= .003 and ci[0] > 0 and summary['wm_over50']['clean_iou'] >= evidence['baseline']['wm_over50']['clean_iou']-.01
    deployment = {'baseline': evidence['baseline'], 'summary': summary, 'per_image': rows,
                  'delta': float(delta.mean()), 'ci95': ci.tolist(), 'passed': bool(passed),
                  'reproduction': reproduction, 'checkpoint_sha256': source_hash,
                  'protocol': 'Actual carrier-only deployment, same fixed57-image fold and fixed rule; unchanged acceptance gate',
                  'torch': torch.__version__, 'cuda': torch.version.cuda}
    atomic_json(out / 'deployment_validation.json', deployment)
    np.savez_compressed(out / 'validation-maps.npz', names=names, probabilities=maps, targets=targets)
    assert passed, deployment['delta']
    print('CARRIER_REPRODUCTION', json.dumps(reproduction), flush=True)
    z = np.load(a.root / 'pcmv-runs/base-candidate/test-probabilities.npz')
    base = {str(n): p >= .5 for n, p in zip(z['names'], z['probabilities'])}
    test = LocalisationDataset('data', 'test', methods=['pixelseal'])
    assert len(test) == len(base) == 340 and not test.train
    eligible = [test[i] for i in range(len(test)) if not base[test[i]['image_name']].any()]
    corrections, local_maps = {}, []
    for begin in range(0, len(eligible), 8):
        items = eligible[begin:begin+8]
        for it, local in zip(items, forward(items)):
            corrections[it['image_name']] = mask_from_local(local)
            local_maps.append(local)
    names = [test[i]['image_name'] for i in range(len(test))]
    masks = torch.tensor(np.stack([corrections.get(n, base[n]).astype(np.uint8) for n in names]))
    dest = out / 'pixelseal-subset.pt'
    save_submission(dest, names, masks, names)
    validate_submission(torch.load(dest, weights_only=True), names)
    np.savez_compressed(out / 'test-carrier.npz', names=list(corrections), local_logits=local_maps)
    receipt = {'family': 'pixelseal', 'n': 340, 'changed_n': len(corrections), 'changed_names': list(corrections),
               'artifact_sha256': sha256(dest), 'checkpoint_sha256': source_hash,
               'evidence_sha256': sha256(evidence_path), 'rule': 'top164 carrier cells only when #3188 base is empty',
               'deployment_evidence_sha256': sha256(out / 'deployment_validation.json'),
               'test_GT_used': False, 'threshold_tuned': False, 'reproduction': reproduction,
               'seconds': time.monotonic()-started}
    atomic_json(out / 'completed.json', receipt)
    print('CARRIER_TEST_DONE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
