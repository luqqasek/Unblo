"""Stage 0 only: fixed development split, exact official masks and wrappers."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
RUNS = ROOT / "experiments/message_localizer/results/runs"
FAMILIES = ("pixelseal", "trustmark")
BUCKETS = ("le1", "1to5", "5to20", "gt20", "clean_holes")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bucket(area):
    return "empty" if area == 0 else "le1" if area <= .01 else "1to5" if area <= .05 else "5to20" if area <= .2 else "gt20"


def select(rows, name):
    if name == "all":
        return rows
    if name == "clean_holes":
        return [r for r in rows if r["full_watermark"] and 0 < r["true_area"] < 1]
    return [r for r in rows if r["bucket"] == name]


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def metadata():
    return {m["image_name"]: m for m in read(ROOT / "data/train/metadata.json")}


def masks128():
    with np.load(ROOT / "data/train/masks_128.npz") as z:
        return dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


def ious(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    vals = []
    for p, y in ((pred, gt), (~pred, ~gt)):
        union = (p | y).sum()
        vals.append(float((p & y).sum() / union) if union else 1.)
    return {"fg_iou": vals[0], "clean_iou": vals[1], "miou": sum(vals) / 2}
