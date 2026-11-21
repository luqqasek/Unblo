"""Test submission: TrustMark full-image top1 empty fallback; priors for other methods."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("patch_experiment", HERE / "run.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
TASK = experiment.TASK
from run import read_metadata, fit_priors, predict
from watermark_localization.submission import save_submission, validate_submission


def main():
    torch.set_num_threads(4)
    train = read_metadata(TASK / "data", "train")
    test = read_metadata(TASK / "data", "test")
    with np.load(TASK / "data/train/masks_128.npz") as z:
        lookup = dict(zip(z["names"].tolist(), z["masks"]))
        targets = np.stack([lookup[m["image_name"]] for m in train])
    priors = fit_priors(train, targets, list(range(len(train))))
    masks = predict(test, priors, "prior")
    original = masks.copy()
    from trustmark import TrustMark
    model = TrustMark(use_ECC=False, verbose=False, model_type="Q", device="cpu",
                      loadRemover=False, loadBBoxDetector=True)
    detector = model.detector.yolo.yolo_model
    detector.roi_heads.detections_per_img = 20
    assert detector.roi_heads.score_thresh == .05 and detector.roi_heads.nms_thresh == .5
    records = []
    start = time.perf_counter()
    for i, item in enumerate(test):
        if item["method"] != "trustmark":
            continue
        image = np.asarray(Image.open(TASK / "data/test/images" / f"{item['image_name']}.png").convert("RGB"))
        assert image.shape == (512, 512, 3)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            detections, _ = detector([tensor])
        record = {"boxes": detections[0]["boxes"].cpu().numpy().tolist(),
                  "scores": detections[0]["scores"].cpu().numpy().tolist()}
        masks[i] = experiment.rasterize([record], [(0, 0, 512)], limit=1, fallback=False)
        records.append({"image_name": item["image_name"], **record})
        if len(records) % 25 == 0:
            print(f"TrustMark test {len(records)}/340", flush=True)
    assert len(records) == 340
    other = [i for i, m in enumerate(test) if m["method"] != "trustmark"]
    assert np.array_equal(masks[other], original[other])
    names = [m["image_name"] for m in test]
    out = HERE / "results/test_top1_empty"
    path = save_submission(out / "submission.pt", names, torch.from_numpy(masks), names)
    validate_submission(torch.load(path, weights_only=True), names)
    summary = {"trustmark": "full image, top1, empty on no detection", "pixelseal": "all-train spatial prior",
               "stegastamp": "all-train spatial prior", "test_count": len(names),
               "trustmark_count": len(records), "trustmark_no_detections": sum(not r["boxes"] for r in records),
               "bytes": path.stat().st_size, "seconds_inference": time.perf_counter() - start}
    (out / "prediction_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "trustmark_detections.json").write_text(json.dumps(records))
    print(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
