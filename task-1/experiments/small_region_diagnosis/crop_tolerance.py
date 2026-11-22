"""Follow-up diagnostic: sensitivity of GT-oracle crops to small geometric errors."""
import argparse
import time

import numpy as np
import torch
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

from .common import *
from .decoder_probe import bbox, decode


def shifted(box, dx, dy):
    width, height = box[2]-box[0], box[3]-box[1]
    x = int(np.clip(box[0]+dx, 0, 512-width)); y = int(np.clip(box[1]+dy, 0, 512-height))
    return [x, y, x+width, y+height]


def boxes(mask, mask128):
    b = bbox(mask)
    out = {"exact": b, "oracle_grid128": [4*v for v in bbox(mask128)]}
    for margin in (2, 4, 8):
        out[f"expand_{margin}"] = [max(0, b[0]-margin), max(0, b[1]-margin), min(512, b[2]+margin), min(512, b[3]+margin)]
        for axis, dx, dy in (("xplus", margin, 0), ("xminus", -margin, 0), ("yplus", 0, margin), ("yminus", 0, -margin)):
            out[f"shift_{margin}_{axis}"] = shifted(b, dx, dy)
    return out


def summarize(rows):
    results = []
    for group in ("le1", "1to5"):
        for shape in ("all", "near_rectangle", "irregular"):
            for variant in ("exact", "oracle_grid128", "expand_2", "expand_4", "expand_8", "shift_2", "shift_4", "shift_8"):
                rr = [r for r in rows if r["bucket"] == group and (shape == "all" or r["shape"] == shape)
                      and (r["variant"] == variant or r["variant"].startswith(variant+"_"))]
                if not rr:
                    continue
                names = sorted({r["image_name"] for r in rr})
                per_image = [np.mean([r["bit_accuracy"] for r in rr if r["image_name"] == n]) for n in names]
                results.append({"bucket": group, "shape": shape, "variant": variant, "n_images": len(names), "n_views": len(rr),
                                "bit_accuracy": float(np.mean(per_image)), "fraction_images_ge90": float((np.array(per_image) >= .9).mean()),
                                "gt_retained": mean(rr, "gt_retained"), "crop_iou_with_bbox": mean(rr, "crop_iou_with_bbox")})
    return results


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", choices=FAMILIES, required=True); a = ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic(); device = "cuda:0"; wm = load_watermark(a.family, ROOT/"checkpoints", device)
    for v in vars(wm).values():
        if isinstance(v, torch.nn.Module):
            v.eval().requires_grad_(False)
    ds = LocalisationDataset(ROOT/"data", "train", methods=[a.family]); rows = []
    baseline = {r["image_name"]: r["bit_accuracy"] for r in read(OUT/"decoder"/a.family/"per_image.json") if r["view"] == "bbox256"}
    for i in range(len(ds)):
        name = ds.meta[i]["image_name"]
        if name not in baseline:
            continue
        item = ds[i]; mask = item["mask"].astype(bool); b = bbox(mask)
        barea = (b[2]-b[0])*(b[3]-b[1]); fill = mask.sum()/barea
        for kind, box in boxes(mask, item["mask128"]).items():
            x0, y0, x1, y1 = box
            with torch.inference_mode():
                logits = decode(wm, item["image"][y0:y1, x0:x1], "bbox256", device)
            accuracy = float(((logits > 0) == item["message"]).mean())
            if kind == "exact":
                assert accuracy == baseline[name]
            intersection = max(0, min(x1, b[2])-max(x0, b[0]))*max(0, min(y1, b[3])-max(y0, b[1]))
            rows.append({"family": a.family, "image_name": name, "bucket": bucket(float(item["mask128"].mean())),
                         "shape": "near_rectangle" if fill > .99 else "irregular", "variant": kind, "box": box,
                         "bit_accuracy": accuracy, "baseline_accuracy": baseline[name], "delta": accuracy-baseline[name],
                         "gt_retained": float(mask[y0:y1, x0:x1].sum()/mask.sum()),
                         "crop_iou_with_bbox": float(intersection/(barea+(x1-x0)*(y1-y0)-intersection))})
    out = OUT/"crop_tolerance"/a.family
    save(out/"per_image.json", rows); csv_rows(out/"per_image.csv", rows)
    groups = summarize(rows)
    save(out/"summary.json", {"family": a.family, "groups": groups, "n_images": len(baseline), "views_per_image": 17,
         "seconds": time.monotonic()-start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "code_sha256": sha(__file__),
         "protocol": "Follow-up chosen after exact crop recovery and margin16/32 failure. Same17 views/image for PS/TM. Oracle GT boxes, original RGB pixels, then bilinear antialias to256 and original decoder. Four axial translations averaged per image, not best-of-four. Grid128 uses bounding box of official GT128 multiplied by4. No candidate-generation or deployment claim."})
    print(a.family, "seconds", round(time.monotonic()-start, 1), flush=True)
    for r in groups:
        if r["bucket"] == "1to5" and r["shape"] == "near_rectangle":
            print(r["variant"], round(r["bit_accuracy"], 4), flush=True)


if __name__ == "__main__":
    main()
