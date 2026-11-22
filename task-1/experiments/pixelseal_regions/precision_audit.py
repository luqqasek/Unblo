"""Matched FP32/BF16 recognition audit; never selects test predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .model import RegionPresence
from .geometry import tight


def summarize(rows, key):
    delta = np.array([r[key][1] - r[key][0] for r in rows])
    return {
        "pairs": len(rows),
        "auc": float(roc_auc_score([0, 1] * len(rows), [s for r in rows for s in r[key]])),
        "paired_order_accuracy": float((delta > 0).mean()),
        "tie_fraction": float((delta == 0).mean()),
        "paired_auc_with_half_ties": float((delta > 0).mean() + .5 * (delta == 0).mean()),
        "mean_signed_difference": float(delta.mean()),
        "median_absolute_difference": float(np.median(np.abs(delta))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(a.root / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model = RegionPresence().to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    rows = []
    with torch.inference_mode():
        for i, f in enumerate(sorted((a.root / 'recognition-pairs').glob('*.npz'))):
            with np.load(f) as z:
                x = torch.from_numpy(np.stack([z['clean'], z['marked']])).permute(0, 3, 1, 2).to(device).float() / 255
                hw = z['mask'].shape
                box = torch.tensor(tight(z['mask']), device=device)[None]
            row = {'id': f.stem}
            for amp in (False, True):
                if amp and device.type != 'cuda':
                    continue
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    feat, obs = model.encode(x)
                    tables = [model.tables(fi, oi) for fi, oi in zip(feat, obs)]
                    scores = [model.score(t, box, hw)[0, 0].float().item() for t in tables]
                row['bf16' if amp else 'fp32'] = scores
                if amp:
                    with torch.autocast(device.type, enabled=False):
                        row['bf16_encoder_fp32_head'] = [model.score(t, box, hw)[0, 0].float().item() for t in tables]
            rows.append(row)
            if (i + 1) % 50 == 0:
                print('AUDITED', i + 1, flush=True)
    keys = ['fp32'] + (['bf16', 'bf16_encoder_fp32_head'] if device.type == 'cuda' else [])
    result = {
        'protocol': 'Same EMA400 weights, exact held-out clean/marked pairs and known geometry. Diagnostic only; no training or threshold selection.',
        'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
        'torch': torch.__version__,
        'summary': {k: summarize(rows, k) for k in keys},
        'per_pair': rows,
    }
    (a.root / 'precision_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
