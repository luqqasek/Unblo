"""CPU reference baselines; evaluation requires only NumPy, submission also PyTorch."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from watermark_localization.metrics import mean_iou

METHODS = ("pixelseal", "trustmark", "stegastamp")


def partition(meta, seed=42):
    """Hold out 34 records per method, using sorted names before seeded shuffling."""
    rng = np.random.default_rng(seed)
    fit, validation = [], []
    for method in METHODS:
        indices = sorted((i for i, m in enumerate(meta) if m["method"] == method),
                         key=lambda i: meta[i]["image_name"])
        if len(indices) != 170:
            raise ValueError(f"Expected 170 training records for {method}")
        indices = rng.permutation(indices).tolist()
        validation.extend(indices[:34])
        fit.extend(indices[34:])
    return fit, validation


def fit_priors(meta, masks, indices):
    return {method: masks[[i for i in indices if meta[i]["method"] == method]].mean(0)
            for method in METHODS}


def predict(meta, priors, variant):
    if variant in ("zeros", "ones"):
        return np.full((len(meta), 128, 128), int(variant == "ones"), dtype=np.uint8)
    if variant != "prior":
        raise ValueError(variant)
    return np.stack([(priors[m["method"]] >= 0.5).astype(np.uint8) for m in meta])


def read_metadata(data, split):
    meta = json.loads((data / split / "metadata.json").read_text(encoding="utf-8"))
    expected = 170 if split == "train" else 340
    if Counter(m["method"] for m in meta) != Counter({m: expected for m in METHODS}):
        raise ValueError(f"Unexpected {split} method counts")
    names = [m["image_name"] for m in meta]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate {split} names")
    for m in meta:
        bits = m["message"]
        if len(bits) != (256 if m["method"] == "pixelseal" else 100) or set(bits) - {"0", "1"}:
            raise ValueError(f"Invalid message: {m['image_name']}")
        if not (data / split / "images" / f"{m['image_name']}.png").is_file():
            raise FileNotFoundError(f"Missing image: {m['image_name']}")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=TASK / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()
    if not (args.data / "train" / "metadata.json").is_file():
        parser.error(f"Training data missing at {args.data}; see task-1/scripts/setup.ps1")
    meta = read_metadata(args.data, "train")
    with np.load(args.data / "train" / "masks_128.npz", allow_pickle=False) as archive:
        names, raw = archive["names"].tolist(), archive["masks"]
        if len(set(names)) != len(names) or set(names) != {m["image_name"] for m in meta}:
            raise ValueError("Mask names do not match training metadata")
        if raw.shape != (510, 128, 128) or not np.isin(raw, [0, 1]).all():
            raise ValueError("Invalid training masks")
        lookup = dict(zip(names, raw))
        masks = np.stack([lookup[m["image_name"]] for m in meta])
    fit, validation = partition(meta, args.seed)
    priors = fit_priors(meta, masks, fit)
    val_meta = [meta[i] for i in validation]
    report = {"seed": args.seed, "threshold": 0.5,
              "fit_names": [meta[i]["image_name"] for i in fit],
              "validation_names": [m["image_name"] for m in val_meta], "scores": {}}
    for variant in ("zeros", "ones", "prior"):
        predicted = predict(val_meta, priors, variant)
        scores = {"overall": mean_iou(predicted, masks[validation])}
        for method in METHODS:
            subset = [i for i, m in enumerate(val_meta) if m["method"] == method]
            scores[method] = mean_iou(predicted[subset], masks[np.array(validation)[subset]])
        report["scores"][variant] = scores
    best = max(report["scores"], key=lambda k: report["scores"][k]["overall"])
    report["selected"] = best
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "priors.npz", **priors)
    print(json.dumps({"scores": report["scores"], "selected": best}, indent=2))
    if args.submission:
        import torch
        from watermark_localization.submission import save_submission
        test = read_metadata(args.data, "test")
        # Refit on all labelled data only after validation and variant selection.
        final_priors = fit_priors(meta, masks, list(range(len(meta))))
        names = [m["image_name"] for m in test]
        path = save_submission(args.out / "submission.pt", names,
                               torch.from_numpy(predict(test, final_priors, best)), names)
        print(f"Saved {path} ({path.stat().st_size} bytes); no upload performed")


if __name__ == "__main__":
    main()
