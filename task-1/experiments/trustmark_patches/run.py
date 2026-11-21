"""Compare TrustMark bbox limits, empty fallback, and a 2x2 grid of 256px patches."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(TASK / "experiments/official_localization_cv"))
sys.path.insert(0, str(TASK / "checkpoints/trustmark-localization/python"))
from cross_validate import effect, per_image
from watermark_localization.metrics import downsample_mask


def rasterize(detections, regions, limit, fallback):
    """Raw local xyxy pixels -> full-image mask; never resize patch coordinates."""
    full = np.zeros((512, 512), dtype=np.uint8)
    for detection, (left, top, size) in zip(detections, regions):
        boxes = np.asarray(detection["boxes"], dtype=float).reshape(-1, 4)
        scores = np.asarray(detection["scores"])
        if len(boxes):
            boxes = boxes[np.argsort(-scores, kind="stable")[:limit]]
        elif fallback:
            boxes = np.array([[0, 0, size, size]], dtype=float)
        for box in boxes:
            x1, y1, x2, y2 = np.clip(box, 0, size).astype(int)
            # Match upstream's replacement of degenerate boxes with the full region.
            if x2-x1 < 1 or y2-y1 < 1:
                if not fallback:
                    continue
                x1, y1, x2, y2 = 0, 0, size, size
            full[top+y1:top+y2, left+x1:left+x2] = 1
    return downsample_mask(full)


def main():
    torch.set_num_threads(4)
    from trustmark import TrustMark
    model = TrustMark(use_ECC=False, verbose=False, model_type="Q", device="cpu",
                      loadRemover=False, loadBBoxDetector=True)
    detector = model.detector.yolo.yolo_model
    detector.roi_heads.detections_per_img = 20
    assert detector.roi_heads.score_thresh == 0.05
    assert detector.roi_heads.nms_thresh == 0.5
    meta = json.loads((TASK / "data/train/metadata.json").read_text())
    ids = [i for i, m in enumerate(meta) if m["method"] == "trustmark"]
    names = [meta[i]["image_name"] for i in ids]
    full_region = [(0, 0, 512)]
    patches = [(x, y, 256) for y in (0, 256) for x in (0, 256)]
    configurations = {"full_top1_default": ("full", 1, True),
                      "full_top1_empty": ("full", 1, False),
                      "full_all4_default": ("full", 4, True),
                      "full_all20_default": ("full", 20, True),
                      "full_all4_empty": ("full", 4, False),
                      "full_all20_empty": ("full", 20, False),
                      "patch2x2_top1_default": ("patch", 1, True),
                      "patch2x2_top1_empty": ("patch", 1, False),
                      "patch2x2_all4_default": ("patch", 4, True),
                      "patch2x2_all20_default": ("patch", 20, True),
                      "patch2x2_all4_empty": ("patch", 4, False),
                      "patch2x2_all20_empty": ("patch", 20, False)}
    out = Path(__file__).parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "predictions.npz"
    start = time.perf_counter()
    if cache.exists():
        saved = np.load(cache)
        assert saved["image_names"].tolist() == names
        predicted = {k: saved[k] for k in configurations if k in saved.files}
        missing = set(configurations) - set(predicted)
        if missing:
            records = json.loads((out / "detections.json").read_text())
            assert [r["image_name"] for r in records] == names
            for label in missing:
                mode, limit, fallback = configurations[label]
                predicted[label] = np.stack([rasterize(r["full"] if mode == "full" else r["patches"],
                                                     full_region if mode == "full" else patches, limit, fallback)
                                             for r in records])
            saved.close()
            np.savez_compressed(cache, image_names=names, **predicted)
    else:
        predicted = {k: [] for k in configurations}
        records = []
        for j, name in enumerate(names):
            image = np.asarray(Image.open(TASK / "data/train/images" / f"{name}.png").convert("RGB"))
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
            with torch.inference_mode():
                # Direct Faster R-CNN call preserves confidence/empty output, unlike localize().
                full, _ = detector([tensor])
                cropped, _ = detector([tensor[:, y:y+s, x:x+s] for x, y, s in patches])
            convert = lambda ds: [{"boxes": d["boxes"].cpu().numpy().tolist(),
                                   "scores": d["scores"].cpu().numpy().tolist()} for d in ds]
            full, cropped = convert(full), convert(cropped)
            records.append({"image_name": name, "full": full, "patches": cropped})
            for label, (mode, limit, fallback) in configurations.items():
                predicted[label].append(rasterize(full if mode == "full" else cropped,
                                                   full_region if mode == "full" else patches,
                                                   limit, fallback))
            if (j+1) % 10 == 0 or j+1 == len(names):
                print(f"TrustMark patches {j+1}/{len(names)}", flush=True)
        predicted = {k: np.asarray(v, dtype=np.uint8) for k, v in predicted.items()}
        np.savez_compressed(cache, image_names=names, **predicted)
        (out / "detections.json").write_text(json.dumps(records))
    duration = time.perf_counter() - start
    z = np.load(TASK / "data/train/masks_128.npz")
    gt_lookup = dict(zip(z["names"].tolist(), z["masks"]))
    targets = np.stack([gt_lookup[n] for n in names])
    oof = np.load(TASK / "experiments/official_localization_cv/results/oof.npz")
    assert oof["image_names"].tolist() == [m["image_name"] for m in meta]
    baseline = per_image(oof["baseline"][ids], targets)
    fold_ids = oof["fold_ids"][ids]
    reference_file = TASK / "experiments/official_localization_cv/results/cache/trustmark_masks.npz"
    reference_meta = json.loads(reference_file.with_name("trustmark_metrics.json").read_text())
    order = {n: i for i, n in enumerate(reference_meta["image_names"])}
    reference = np.load(reference_file)
    equivalent = {}
    for label, original in [("full_top1_default", "official_top1"), ("full_all4_default", "official_all_boxes")]:
        expected = reference[original][[order[n] for n in names]]
        equivalent[label] = {"identical_masks": int(np.all(predicted[label] == expected, axis=(1, 2)).sum()),
                             "reference_miou": float(per_image(expected, targets).mean())}
    top1 = per_image(predicted["full_top1_default"], targets)
    report = {"n": 170, "seed": 42, "folds": 5, "patch_size": 256, "patch_grid": [2, 2],
              "evaluation_grid": [128, 128], "max_detections": 20, "score_threshold": 0.05, "nms_threshold": 0.5,
              "seconds_prediction_or_cache_load": duration, "comparison_to_upstream": equivalent,
              "configurations": configurations, "variants": {}}
    for label, masks in predicted.items():
        assert masks.shape == (170, 128, 128) and np.isin(masks, [0, 1]).all()
        scores = per_image(masks, targets)
        report["variants"][label] = {"vs_prior": effect(baseline, scores, fold_ids),
                                     "vs_full_top1": effect(top1, scores, fold_ids),
                                     "full_masks": int((masks.sum((1, 2)) == 128*128).sum()),
                                     "empty_masks": int((masks.sum((1, 2)) == 0).sum())}
    (out / "cv_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {"miou": v["vs_prior"]["detector"], "delta_vs_prior": v["vs_prior"]["delta"],
                         "delta_vs_top1": v["vs_full_top1"]["delta"], "positive_folds_vs_top1": v["vs_full_top1"]["positive_folds"]}
                      for k, v in report["variants"].items()}, indent=2))
    print("Upstream equivalence:", equivalent)


if __name__ == "__main__":
    main()
