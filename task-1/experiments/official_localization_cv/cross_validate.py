"""Five stratified folds; frozen detectors cached once, prior refit in every fold."""
import json
from pathlib import Path
import sys
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(TASK / "experiments/baseline"))
from run import METHODS, fit_priors, predict
from watermark_localization.metrics import mean_iou


def folds(meta, seed=42):
    rng = np.random.default_rng(seed)
    shuffled = {}
    for method in METHODS:
        ids = sorted((i for i, m in enumerate(meta) if m["method"] == method), key=lambda i: meta[i]["image_name"])
        if len(ids) != 170:
            raise ValueError(f"Expected 170 images for {method}")
        shuffled[method] = rng.permutation(ids)
    for k in range(5):
        val = np.concatenate([shuffled[m][k*34:(k+1)*34] for m in METHODS])
        train = np.setdiff1d(np.arange(len(meta)), val)
        yield train, val


def per_image(p, g):
    return np.asarray([mean_iou(a, b) for a, b in zip(p, g)])


def effect(base, detector, fold_ids):
    difference = detector - base
    by_fold = [{"fold": k+1, "baseline": float(base[fold_ids == k].mean()),
                "detector": float(detector[fold_ids == k].mean()),
                "delta": float(difference[fold_ids == k].mean())} for k in range(5)]
    # Conditional on the fitted OOF predictions; not a retraining-aware confidence interval.
    rng = np.random.default_rng(42)
    boot = difference[rng.integers(0, len(difference), size=(10000, len(difference)))].mean(1)
    return {"baseline": float(base.mean()), "detector": float(detector.mean()),
            "delta": float(difference.mean()), "fold_delta_std": float(np.std([f["delta"] for f in by_fold], ddof=1)),
            "positive_folds": sum(f["delta"] > 0 for f in by_fold),
            "conditional_paired_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "images_improved": int((difference > 1e-12).sum()),
            "images_worse": int((difference < -1e-12).sum()), "folds": by_fold}


def main():
    out = Path(__file__).parent / "results"
    cache = out / "cache"
    meta = json.loads((TASK / "data/train/metadata.json").read_text())
    names = [m["image_name"] for m in meta]
    z = np.load(TASK / "data/train/masks_128.npz")
    lookup = dict(zip(z["names"].tolist(), z["masks"]))
    targets = np.stack([lookup[n] for n in names])
    baseline = np.empty_like(targets)
    fold_ids = np.full(len(meta), -1)
    fold_manifest = []
    for k, (train, val) in enumerate(folds(meta)):
        assert not set(train) & set(val)
        baseline[val] = predict([meta[i] for i in val], fit_priors(meta, targets, train.tolist()), "prior")
        fold_ids[val] = k
        fold_manifest.append({"fold": k+1, "fit_names": [names[i] for i in train], "validation_names": [names[i] for i in val]})
    assert np.all(fold_ids >= 0)
    baseline_scores = per_image(baseline, targets)
    report = {"seed": 42, "folds": 5, "n_images": 510,
              "design": "5-fold CV, 136 fit / 34 validation per method; no detector training or tuning",
              "uncertainty": "Bootstrap interval is conditional on fixed OOF predictions; overlapping training sets and any shared source images are not modeled.",
              "stegastamp_provenance": "community mirror, equivalence to unavailable official weights unverified",
              "methods": {}, "overall": {"baseline": float(baseline_scores.mean())}}
    hybrids = {"trustmark_only": baseline.copy(), "trustmark_and_stegastamp": baseline.copy()}
    for method in ("trustmark", "stegastamp"):
        ids = [i for i, m in enumerate(meta) if m["method"] == method]
        metadata = json.loads((cache / f"{method}_metrics.json").read_text())
        cached_names = metadata["image_names"]
        assert len(set(cached_names)) == 170 and set(cached_names) == {names[i] for i in ids}
        order = {name: i for i, name in enumerate(cached_names)}
        masks = np.load(cache / f"{method}_masks.npz")
        variants = {}
        for variant in masks.files:
            predicted = masks[variant][[order[names[i]] for i in ids]]
            assert predicted.shape == (170, 128, 128) and np.isin(predicted, [0, 1]).all()
            variants[variant] = effect(baseline_scores[ids], per_image(predicted, targets[ids]), fold_ids[ids])
            if variant in ("official_top1", "official_segmentation"):
                hybrids["trustmark_and_stegastamp"][ids] = predicted
                if method == "trustmark":
                    hybrids["trustmark_only"][ids] = predicted
        report["methods"][method] = variants
    for label, predicted in hybrids.items():
        value = mean_iou(predicted, targets)
        report["overall"][label] = {"miou": value, "delta": value - report["overall"]["baseline"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds.json").write_text(json.dumps(fold_manifest, indent=2))
    (out / "cv_metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "oof.npz", image_names=names, baseline=baseline, fold_ids=fold_ids, **hybrids)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
