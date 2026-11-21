"""Per-image diagnostics; official score never replaced by pooled ROC/AP."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
import torch

from wmloc.utils import miou_image
from .common import atomic_json, rgb_tensor
from .generator import EncodedDataset, pair_views


def average_precision(prob, target):
    """sklearn non-interpolated average precision, with ties handled as score groups."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(target.ravel(), prob.ravel())) if target.any() else None


def boundary_f1(pred, target):
    # Exclude image-frame edges from both maps; tolerance is 1 grid cell.
    edges = []
    for mask in (pred, target):
        edge = mask ^ binary_erosion(mask)
        edge[[0, -1], :] = False; edge[:, [0, -1]] = False
        edges.append(edge)
    p, g = edges
    if not p.any() and not g.any():
        return 1.
    precision = (p & binary_dilation(g)).sum()/max(1, p.sum())
    recall = (g & binary_dilation(p)).sum()/max(1, g.sum())
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.


def image_metrics(prob, target, threshold=.5):
    from sklearn.metrics import roc_auc_score
    p, y = prob >= threshold, target.astype(bool)
    ious = []
    for c in (False, True):
        pc, gc = p == c, y == c
        union = (pc | gc).sum()
        ious.append(float((pc & gc).sum()/union) if union else 1.)
    fraction = float(y.mean())
    return {
        "miou": miou_image(p, y), "fg_iou": ious[1], "clean_iou": ious[0],
        "fg_ap": average_precision(prob, y), "clean_ap": average_precision(1-prob, ~y),
        "roc_auc": float(roc_auc_score(y.ravel(), prob.ravel())) if y.any() and not y.all() else None,
        "true_area": fraction, "pred_area": float(p.mean()), "minority_fraction": min(fraction, 1-fraction),
        "empty": float(not p.any()), "full": float(p.all()),
        "complete_fg_miss": float(not (p & y).any()) if y.any() else None,
        "fp_area": float((p & ~y).mean()),
        "internal_boundary_f1": boundary_f1(p, y),
        "constant_clean_miou": miou_image(np.zeros_like(y), y),
        "constant_marked_miou": miou_image(np.ones_like(y), y),
        "minority_ap": average_precision(prob if fraction <= .5 else 1-prob, y if fraction <= .5 else ~y),
        "minority_iou": ious[1] if fraction <= .5 else ious[0],
    }


def summarize(rows):
    def group(rs):
        keys = [k for k, v in rows[0].items() if isinstance(v, (float, int)) or v is None] if rows else []
        out = {"n": len(rs)}
        for key in keys:
            values = [r[key] for r in rs if r.get(key) is not None]
            out[key] = {"mean": float(np.mean(values)) if values else None,
                        "defined": len(values), "undefined": len(rs)-len(values)}
        return out
    groups = {"all": rows, "partial": [r for r in rows if 0 < r["true_area"] < 1],
              "fully_clean": [r for r in rows if r["true_area"] == 0],
              "fully_marked": [r for r in rows if r["true_area"] == 1],
              "fg_le_1pct": [r for r in rows if 0 < r["true_area"] <= .01],
              "fg_1_to_5pct": [r for r in rows if .01 < r["true_area"] <= .05],
              "fg_gt_50pct": [r for r in rows if .5 < r["true_area"] < 1]}
    for lo, hi in ((0, .0025), (.0025, .01), (.01, .05), (.05, .2), (.2, .5)):
        groups[f"minority_{lo}_{hi}"] = [r for r in rows if lo < r["minority_fraction"] <= hi]
    for regime in sorted({r.get("regime", "real") for r in rows}):
        groups[regime] = [r for r in rows if r.get("regime", "real") == regime]
    return {"groups": {k: group(v) for k, v in groups.items()},
            "ap_implementation": "sklearn.metrics.average_precision_score, non-interpolated, per image, absent positive class excluded",
            "roc_undefined": "single-class GT excluded", "official_empty_union": 1,
            "boundary": "1-cell tolerance, image frame excluded"}


def gallery(path, image, target, prob, baseline=None):
    pred = prob >= .5
    panels = [image, np.repeat(target[..., None]*255, 3, 2).astype(np.uint8),
              np.stack((prob, np.zeros_like(prob), 1-prob), 2)*255,
              np.repeat(pred[..., None]*255, 3, 2),
              np.stack((pred & ~target.astype(bool), np.zeros_like(pred), ~pred & target.astype(bool)), 2)*255]
    labels = ["RGB", "GT", "Probability", "Prediction @0.5", "FP red / FN blue"]
    if baseline is not None:
        panels.append(np.repeat((baseline >= .5)[..., None]*255, 3, 2)); labels.append("Baseline")
    canvas = Image.new("RGB", (256*len(panels), 282), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray(panel.astype(np.uint8)).resize((256, 256)), (i*256, 26))
        draw.text((i*256+4, 5), label, fill="black")
    canvas.save(path)


@torch.inference_mode()
def evaluate_synthetic(model, dataset, device, limit=256, seed=1776, threshold=.5, out=None, update=0):
    was_training = model.training; model.eval()
    rows, maps, targets, names, gallery_data = [], [], [], [], []
    # Reinitialize per evaluation: exact same sources, recipes, order and messages.
    for i in range(min(limit, len(dataset.rows))):
        clean, marked, row = dataset.load(i)
        rng = np.random.default_rng(seed+i)
        pair_kind = "move" if row["rectangle"] == [0, 0, 512, 512] else "clean_twin"
        xs, ys, _ = pair_views(clean, marked, row, rng, pair_kind, simple=update < 5000)
        p = model(torch.stack([rgb_tensor(x) for x in xs]).to(device))["logits"].sigmoid().cpu().numpy()
        for j, (image, target, prob) in enumerate(zip(xs, ys, p)):
            metrics = image_metrics(prob, target, threshold)
            # Spatial shuffle is seeded per image; never selected as a model.
            shuffle = np.random.default_rng(seed+100000+i*2+j).permutation(prob.ravel()).reshape(prob.shape)
            metrics["spatial_shuffle_miou"] = miou_image(shuffle >= threshold, target)
            metrics.update(image_name=f"{i}:{j}", source_id=row["source_id"], regime=row["regime"], primary=j == 0)
            rows.append(metrics); maps.append(prob.astype(np.float32)); targets.append(target); names.append(f"{i}:{j}")
            if j == 0:
                gallery_data.append((metrics["miou"], i, image, target, prob))
        changed = ys[0] != ys[1]
        if changed.any():
            direction = ys[0].astype(np.int8)-ys[1].astype(np.int8)
            rows[-2]["pair_changed_order_accuracy"] = float((direction[changed]*(p[0]-p[1])[changed] > 0).mean())
            rows[-2]["pair_probability_movement"] = float((direction[changed]*(p[0]-p[1])[changed]).mean())
    # Paired companions must not silently alter the primary sampling/evaluation mix.
    primary_rows = [r for r in rows if r["primary"]]
    summary = summarize(primary_rows)
    summary["companions"] = summarize([r for r in rows if not r["primary"]])
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        atomic_json(out/"metrics.json", summary)
        keys = sorted({k for r in rows for k in r})
        with open(out/"per_image.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(out/"maps.npz", names=names, probabilities=np.stack(maps), targets=np.stack(targets))
        (out/"gallery").mkdir(exist_ok=True)
        ordered = sorted(gallery_data, key=lambda x:x[0])
        selected = ordered[:8]+ordered[-4:]
        for score, i, image, target, prob in selected:
            gallery(out/"gallery"/f"{i:05d}-{score:.3f}.png", image, target, prob)
    model.train(was_training)
    return summary


def main():
    from .model import NativeLocalizer
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--unlock-locked", action="store_true")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = EncodedDataset(a.manifest)
    if any(r["partition"] == "locked" for r in dataset.rows) and not a.unlock_locked:
        raise SystemExit("Locked suite remains closed")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = NativeLocalizer(checkpointing=False).to(a.device)
    model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    print(json.dumps(evaluate_synthetic(model, dataset, a.device, a.limit, out=a.out), indent=2))


if __name__ == "__main__":
    main()
