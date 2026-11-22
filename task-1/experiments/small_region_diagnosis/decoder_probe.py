"""Frozen official decoders on every real <=5% foreground, with GT-oracle crops."""
import argparse
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from .common import *


def tensor(image, device):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].float().to(device)/255


def bbox(mask):
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]


def centered(box, side):
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    x = int(np.clip(round(cx-side/2), 0, 512-side)); y = int(np.clip(round(cy-side/2), 0, 512-side))
    return [x, y, x+side, y+side]


def background(mask, side, rng):
    # Enumerate all all-clean windows, then pick one uniformly; no contaminated control.
    a = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    counts = a[side:, side:] - a[:-side, side:] - a[side:, :-side] + a[:-side, :-side]
    y, x = np.where(counts == 0)
    if not len(x):
        return None
    j = int(rng.integers(len(x)))
    return [int(x[j]), int(y[j]), int(x[j]+side), int(y[j]+side)]


def views(item):
    im, mask = item["image"], item["mask"].astype(bool)
    box = bbox(mask)
    rng = np.random.default_rng(int(item["image_name"][:8], 16))
    boxes = {"full": [0, 0, 512, 512], "bbox256": box, "margin128_to256": centered(box, 128),
             "margin192_to256": centered(box, 192), "background128_to256": background(mask, 128, rng)}
    for margin in (16, 32):
        boxes[f"bbox_margin{margin}_to256"] = [max(0, box[0]-margin), max(0, box[1]-margin), min(512, box[2]+margin), min(512, box[3]+margin)]
    result = []
    for kind, b in boxes.items():
        if b is None:
            continue
        x0, y0, x1, y1 = b
        result.append((kind, im[y0:y1, x0:x1], {"box": b, "gt_retained": float(mask[y0:y1, x0:x1].sum()/mask.sum()),
                       "crop_gt_fraction": float(mask[y0:y1, x0:x1].mean())}))
    return result


def decode(wm, crop, kind, device):
    x = tensor(crop, device)
    if kind != "full" and kind != "bbox_native":
        x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False, antialias=True)
    return wm.logits(x)[0].cpu().numpy()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--family", choices=FAMILIES, required=True); p.add_argument("--device", default="cuda:0"); p.add_argument("--limit", type=int)
    a = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out = OUT / ("smoke" if a.limit else "decoder") / a.family; out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); wm = load_watermark(a.family, ROOT/"checkpoints", a.device)
    for v in vars(wm).values():
        if isinstance(v, torch.nn.Module):
            v.eval().requires_grad_(False)
    ds = LocalisationDataset(ROOT/"data", "train", methods=[a.family])
    indices = [i for i, m in enumerate(ds.meta) if 0 < ds.m128[m["image_name"]].mean() <= .05]
    if a.limit:
        indices = indices[:a.limit]
    rows = []; logit_rows = []; keys = []
    for j, i in enumerate(indices):
        item = ds[i]; name = item["image_name"]; bits = item["message"]
        cropviews = views(item)
        if a.family == "trustmark":
            b = bbox(item["mask"]); x0, y0, x1, y1 = b
            cropviews.append(("bbox_native", item["image"][y0:y1, x0:x1], {"box": b, "gt_retained": 1., "crop_gt_fraction": float(item["mask"][y0:y1, x0:x1].mean())}))
        for kind, crop, detail in cropviews:
            with torch.inference_mode():
                logits = decode(wm, crop, kind, a.device)
            correct = (logits > 0) == bits
            wrong = np.random.default_rng(int(name[:8], 16)+123).integers(0, 2, (64, len(bits)))
            null = ((logits > 0)[None] == wrong).mean(1)
            rows.append({"image_name": name, "family": a.family, "bucket": bucket(float(item["mask128"].mean())),
                         "true_area": float(item["mask128"].mean()), "full_watermark": bool(item["full_watermark"]),
                         "view": kind, "bit_accuracy": float(correct.mean()), "wrong_message_accuracy": float(null.mean()),
                         "signed_logit_mean": float((logits*(bits.astype(float)*2-1)).mean()), **detail})
            logit_rows.append(logits); keys.append(name+":"+kind)
        if j % 5 == 0 or j+1 == len(indices):
            save(out/"per_image.json", rows)
            print(a.family, j+1, "/", len(indices), "elapsed", round(time.monotonic()-start, 1), flush=True)
    groups = {}
    for group in ("le1", "1to5"):
        groups[group] = {}
        for kind in sorted({r["view"] for r in rows}):
            rr = [r for r in rows if r["bucket"] == group and r["view"] == kind]
            groups[group][kind] = {"n": len(rr), **{k: mean(rr, k) for k in ("bit_accuracy", "wrong_message_accuracy", "gt_retained", "crop_gt_fraction")},
                                  "n_ge75": sum(r["bit_accuracy"] >= .75 for r in rr), "n_ge90": sum(r["bit_accuracy"] >= .9 for r in rr)}
    save(out/"summary.json", {"family": a.family, "n_images": len(indices), "groups": groups, "seconds": time.monotonic()-start,
         "peak_allocated_bytes": torch.cuda.max_memory_allocated() if a.device.startswith("cuda") else None,
         "torch": torch.__version__, "device": torch.cuda.get_device_name() if a.device.startswith("cuda") else a.device,
         "wrapper_sha256": sha(ROOT/"wmloc/watermarks.py"), "code_sha256": sha(__file__), "native_resolution": wm.resolution,
         "protocol": "GT oracle crops; bilinear antialias 256 followed by unchanged official preprocessing. Full image uses wrapper directly. TM bbox_native tests direct resize to245. Diagnostic only, no segmentation or model fitting."})
    csv_rows(out/"per_image.csv", rows)
    np.savez_compressed(out/"logits.npz", keys=keys, logits=np.stack(logit_rows))
    print(json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
