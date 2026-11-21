"""Compare cached predictions and export examples, without rerunning detectors."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(TASK / "experiments/baseline"))
from run import partition, fit_priors, predict
from watermark_localization.metrics import mean_iou


def main():
    out = Path(__file__).parent / "results"
    meta = json.loads((TASK / "data/train/metadata.json").read_text())
    archive = np.load(TASK / "data/train/masks_128.npz")
    lookup = dict(zip(archive["names"].tolist(), archive["masks"]))
    masks = np.stack([lookup[m["image_name"]] for m in meta])
    fit, val = partition(meta)
    priors = fit_priors(meta, masks, fit)
    summary = {}
    for method, variant in [("trustmark", "official_top1"), ("stegastamp", "official_segmentation")]:
        report = json.loads((out / f"{method}_metrics.json").read_text())
        items = [meta[i] for i in val if meta[i]["method"] == method]
        names = [m["image_name"] for m in items]
        assert names == report["validation_names"]
        gts = np.stack([lookup[n] for n in names])
        pri = predict(items, priors, "prior")
        cached = np.load(out / f"{method}_masks.npz")
        predicted = cached[variant]
        assert predicted.shape == (34, 128, 128) and np.isin(predicted, [0, 1]).all()
        assert np.isclose(mean_iou(predicted, gts), report["scores"][variant])
        diagnostics = {k: {"empty_masks": int((cached[k].sum((1, 2)) == 0).sum()),
                           "full_masks": int((cached[k].sum((1, 2)) == 128*128).sum())}
                       for k in cached.files}
        summary[method] = {"n": 34, "scores": report["scores"], "diagnostics": diagnostics,
                           "seconds": report["seconds_including_load"]}
        ranks = np.argsort(report["per_image"][variant])
        chosen = ranks[[0, 11, 22, 33]]
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for row, index in enumerate(chosen):
            image = Image.open(TASK / "data/train/images" / (names[index] + ".png")).convert("RGB")
            images = [image, gts[index], pri[index], predicted[index]]
            for col, im in enumerate(images):
                axes[row, col].imshow(im, cmap="gray", vmin=0, vmax=1) if col else axes[row, col].imshow(im)
                axes[row, col].axis("off")
            axes[row, 0].set_title(names[index], fontsize=9)
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].set_title(f"Prior: {mean_iou(pri[index], gts[index]):.3f}")
            axes[row, 3].set_title(f"Detector: {mean_iou(predicted[index], gts[index]):.3f}")
        fig.suptitle(f"{method}: worst / 1st tercile / 2nd tercile / best validation example")
        fig.tight_layout()
        fig.savefig(out / f"{method}_examples.png", dpi=140)
        plt.close(fig)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
