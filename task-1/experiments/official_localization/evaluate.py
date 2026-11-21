"""Evaluate official pretrained detectors on the baseline's fixed holdout."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(TASK / "experiments/baseline"))
from run import partition, fit_priors, predict
from watermark_localization.metrics import downsample_mask, mean_iou


def box_mask(boxes):
    full = np.zeros((512, 512), dtype=np.uint8)
    for box in boxes or []:
        x1, y1, x2, y2 = np.clip(np.asarray(box) * 512, 0, 512).astype(int)
        full[y1:y2, x1:x2] = 1
    return downsample_mask(full)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["trustmark", "stegastamp"], required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--all-images", action="store_true", help="Cache frozen detector predictions for all 170 images for CV")
    args = ap.parse_args()
    meta = json.loads((TASK / "data/train/metadata.json").read_text())
    archive = np.load(TASK / "data/train/masks_128.npz")
    lookup = dict(zip(archive["names"].tolist(), archive["masks"]))
    masks = np.stack([lookup[m["image_name"]] for m in meta])
    fit, val = partition(meta, 42)
    selected = [i for i in (range(len(meta)) if args.all_images else val) if meta[i]["method"] == args.method]
    items = [meta[i] for i in selected]
    targets = masks[selected]
    start = time.perf_counter()
    predictions, records = {}, []
    if args.method == "trustmark":
        import torch
        torch.set_num_threads(args.threads)
        sys.path.insert(0, str(TASK / "checkpoints/trustmark-localization/python"))
        from trustmark import TrustMark
        model = TrustMark(use_ECC=False, verbose=False, model_type="Q", device="cpu",
                          loadRemover=False, loadBBoxDetector=True)
        predictions = {"official_top1": [], "official_all_boxes": []}
        for item in items:
            image = Image.open(TASK / "data/train/images" / (item["image_name"] + ".png")).convert("RGB")
            # Calls the upstream API twice to retain its exact top-score selection.
            box = model.localize(image)
            boxes = model.localize(image, return_all=True)
            predictions["official_top1"].append(box_mask([box] if box is not None else []))
            predictions["official_all_boxes"].append(box_mask(boxes))
            records.append({"image_name": item["image_name"], "box": box, "boxes": boxes})
            print(f"trustmark {len(records)}/{len(items)}", flush=True)
    else:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        import cv2
        tf.compat.v1.disable_eager_execution()
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=args.threads,
                                         inter_op_parallelism_threads=1)
        predictions = {"official_segmentation": []}
        with tf.compat.v1.Session(graph=graph, config=config) as session:
            with graph.as_default():
                model = tf.compat.v1.saved_model.loader.load(
                    session, [tf.saved_model.SERVING], str(TASK / "checkpoints/stegastamp-detector/detector_models/stegastamp_detector"))
            sig = model.signature_def["serving_default"]
            x = graph.get_tensor_by_name(sig.inputs["image"].name)
            y = graph.get_tensor_by_name(sig.outputs["detections"].name)
            for item in items:
                rgb = np.asarray(Image.open(TASK / "data/train/images" / (item["image_name"] + ".png")).convert("RGB"))
                inp = cv2.resize(rgb, (1024, 1024)).astype(np.float32)[None] / 255
                logits = session.run(y, {x: inp})
                assert np.isfinite(logits).all()
                # Official detector.py paints class 0 white (stamp), class 1 black.
                foreground = (np.argmax(logits[0], axis=-1) == 0).astype(np.float32)
                full = (cv2.resize(foreground, (512, 512), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
                predictions["official_segmentation"].append(downsample_mask(full))
                print(f"stegastamp {len(predictions['official_segmentation'])}/{len(items)}", flush=True)
    prior = predict(items, fit_priors(meta, masks, fit), "prior")
    scores = {"baseline_prior": mean_iou(prior, targets),
              "zeros": mean_iou(np.zeros_like(targets), targets),
              "ones": mean_iou(np.ones_like(targets), targets)}
    if args.all_images:
        scores = {} # CV recomputes the baseline separately on each training fold.
    report = {"method": args.method, "seed": 42, "validation_names": [m["image_name"] for m in items],
              "scores": scores, "per_image": {}, "seconds_including_load": time.perf_counter() - start}
    report["image_names"] = [m["image_name"] for m in items]
    report["scope"] = "all labelled images; use cross_validate.py for OOF comparison" if args.all_images else "original holdout"
    report["training"] = "none; frozen pretrained detector, no threshold tuning"
    report["provenance"] = (
        "Adobe TrustMark Q, commit 4c60b9fdb4125fca3001f0803e6890a6b55371ac; published MD5 checked"
        if args.method == "trustmark" else
        "Community mirror Gu-f/StegaStamp d95ea307c73410c96557200656d2e0a59574eedb; "
        "mirror SHA256 checked; original official weights identity unverified")
    for name, predicted in predictions.items():
        predictions[name] = np.asarray(predicted, dtype=np.uint8)
        scores[name] = mean_iou(predictions[name], targets)
        report["per_image"][name] = [mean_iou(p, g) for p, g in zip(predictions[name], targets)]
    report["diagnostics"] = {name: {"empty_masks": int((p.sum(axis=(1, 2)) == 0).sum()),
                                    "full_masks": int((p.sum(axis=(1, 2)) == 128*128).sum()),
                                    "mean_watermarked_fraction": float(p.mean())}
                             for name, p in predictions.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.method}_masks.npz", **predictions)
    (args.out / f"{args.method}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if records:
        (args.out / "trustmark_boxes.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
