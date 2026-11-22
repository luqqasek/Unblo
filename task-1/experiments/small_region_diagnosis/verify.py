"""Verify final measured artifacts and record code/data/checkpoint provenance."""
import csv
import importlib.metadata
import importlib.util
import platform
import sys

import numpy as np
import torch

from .common import *
from .decoder_probe import views
from .crop_tolerance import summarize as summarize_tolerance
from .oof import hysteresis
from .test_diagnostics import (test_oracle_matches_exhaustive_with_ties_and_single_class_images,
                               test_official_metric_and_hysteresis_connectivity,
                               test_crop_controls_are_clean_and_centered_without_padding)
from wmloc.data import LocalisationDataset


def main():
    test_oracle_matches_exhaustive_with_ties_and_single_class_images()
    test_official_metric_and_hysteresis_connectivity()
    test_crop_controls_are_clean_and_centered_without_padding()
    oof = read(OUT/"oof.json"); meta = metadata(); gt = masks128(); checks = {}
    for path, digest in oof["source_hashes"].items():
        assert sha(path) == digest
    for family in FAMILIES:
        summary = oof["families"][family]; assert summary["n_maps"] == 170
        maps = {}
        for fold in (1, 2, 3):
            with np.load(RUNS/family/f"matched-v1-adapt/fold{fold}/maps.npz") as z:
                assert z["probabilities"].dtype == np.float32
                for name, p, y in zip(z["names"], z["probabilities"], z["targets"]):
                    name = str(name); assert name not in maps
                    assert np.array_equal(y.squeeze(), gt[name])
                    maps[name] = p.squeeze()
        for label, policy in summary["policies"].items():
            rows = []
            for name, p in maps.items():
                params = policy["parameters"]
                pred = hysteresis(p, params["low"], params["high"]) if "low" in params else p >= params["threshold"]
                rows.append({**ious(pred, gt[name]), "true_area": float(gt[name].mean()), "bucket": bucket(float(gt[name].mean())), "full_watermark": meta[name]["full_watermark"]})
            for b, g in policy["groups"].items():
                rr = select(rows, b); assert len(rr) == g["n"]
                for k in ("miou", "fg_iou", "clean_iou"):
                    assert abs(mean(rr, k)-g[k]) < 1e-10
        for b in BUCKETS:
            g = summary["groups"][b]
            assert g["oracle_shared_threshold"]["fg_iou"]+1e-9 >= g["fg_iou"]
            assert g["per_image_oracle_fg"]+1e-9 >= g["oracle_shared_threshold"]["fg_iou"]
        ds = LocalisationDataset(ROOT/"data", "train", methods=[family])
        expected = {m["image_name"] for m in ds.meta if 0 < ds.m128[m["image_name"]].mean() <= .05}
        rows = read(OUT/"decoder"/family/"per_image.json")
        assert {r["image_name"] for r in rows} == expected
        assert len(rows) == len({(r["image_name"], r["view"]) for r in rows})
        with np.load(OUT/"decoder"/family/"logits.npz") as z:
            logits = dict(zip(z["keys"].tolist(), z["logits"]))
        assert len(logits) == len(rows)
        for r in rows:
            bits = np.array(list(map(int, meta[r["image_name"]]["message"])))
            computed = float(((logits[r["image_name"]+":"+r["view"]] > 0) == bits).mean())
            assert computed == r["bit_accuracy"]
            if r["view"].startswith("bbox"):
                assert r["gt_retained"] == 1
            if r["view"].startswith("background"):
                assert r["crop_gt_fraction"] == r["gt_retained"] == 0
        for i, m in enumerate(ds.meta):
            if m["image_name"] not in expected:
                continue
            calculated = {kind: detail for kind, _, detail in views(ds[i])}
            for r in rows:
                if r["image_name"] == m["image_name"] and r["view"] in calculated:
                    assert r["box"] == calculated[r["view"]]["box"]
        controls = read(OUT/"signal"/family/"controls.json")
        assert len(controls) == 16*4*3
        assert len({r["source"] for r in controls}) == 16
        assert all(r["bit_accuracy"] == 1 for r in controls if r["side"] == 512 and r["regime"] != "clean")
        signals = read(OUT/"signal"/family/"per_cover.json")
        assert all(0 <= r["zero_pixel_fraction"] <= r["zero_channel_fraction"] <= 1 for r in signals)
        tolerance = read(OUT/"crop_tolerance"/family/"per_image.json")
        assert len(tolerance) == len(expected)*17
        assert len({(r["image_name"], r["variant"]) for r in tolerance}) == len(tolerance)
        assert {r["image_name"] for r in tolerance} == expected
        assert summarize_tolerance(tolerance) == read(OUT/"crop_tolerance"/family/"summary.json")["groups"]
        for r in tolerance:
            assert abs(r["delta"]-(r["bit_accuracy"]-r["baseline_accuracy"])) < 1e-12
            assert 0 <= r["gt_retained"] <= 1 and 0 < r["crop_iou_with_bbox"] <= 1
            if r["variant"].startswith("expand") or r["variant"] == "exact":
                assert r["gt_retained"] == 1
        with (OUT/f"geometry_{family}.csv").open(encoding="utf-8", newline="") as stream:
            geometry = list(csv.DictReader(stream))
        assert len(geometry) == 2*(170+1024)
        checks[family] = {"oof_maps": 170, "real_small_images": len(expected), "decoder_views": len(rows),
                          "crop_tolerance_views": len(tolerance),
                          "synthetic_covers": 16, "geometry_real": 170, "geometry_replay_views": 1024,
                          "probability_metrics_and_policy_buckets_recomputed": True, "decoder_bits_recomputed": True}
    files = list(HERE.glob("*.py"))+list(HERE.glob("*.md"))
    files += [ROOT/"wmloc/watermarks.py", ROOT/"experiments/native_localizer/generator.py", ROOT/"data/train/metadata.json", ROOT/"data/train/masks_128.npz", ROOT/"checkpoints/pixelseal/checkpoint.pth", ROOT/"checkpoints/pixelseal_card.yaml"]
    spec = importlib.util.find_spec("trustmark")
    modeldir = Path(spec.origin).parent/"models"
    files += [p for p in modeldir.glob("*") if p.is_file() and p.suffix in (".pth", ".pt", ".ckpt")]
    provenance = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "packages": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "pillow", "trustmark")},
                  "file_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in files},
                  "data_scope": "Competition train only. No test masks, no leaderboard tuning, no training updates."}
    save(OUT/"provenance.json", provenance)
    save(OUT/"verification.json", {"passed": True, "metric_contract_tests": 3, "families": checks,
                                  "all_oof_source_hashes_unchanged": True, "limitations": ["geometry is code replay, not archived training realizations", "oracle postprocessing selects using evaluation GT", "boundary agreement uses Sobel proxy, not semantic object annotations"]})
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
