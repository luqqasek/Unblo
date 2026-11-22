"""Gated D4 inference and 1020-mask artifact assembly; never uploads a submission."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .core import aligned, digest, write_json
from .models import build_model
from .run import fuse, predict


def current_test(bundle, family, names, dino_paths):
    from experiments.h100_hour.h2_decode import expected_miou_decode
    with np.load(bundle / f"maps/{family}-test.npz") as d:
        prob, base = [aligned(d, names, k) for k in ("probabilities", "masks")]
    members = []
    for path in dino_paths:
        with np.load(path) as d:
            members.append(aligned(d, names, "maps").astype(np.float32))
    m = np.mean(members, 0)
    route = base.mean((1, 2)) >= .5
    q = np.where(route[:, None, None], 1 - m[:, 0], m[:, 1])
    weight = {"pixelseal": .5, "trustmark": .2, "stegastamp": .7}[family]
    mixed = (1 - weight) * prob + weight * q
    current = (mixed > .5).astype(np.uint8) if family == "stegastamp" else expected_miou_decode(mixed)
    return {"base_prob": prob, "base_mask": base, "current_mask": current,
            "corrections": base.astype(bool) & ~(prob >= .5)}


def validate_submission(names, masks, expected):
    if len(names) != 1020 or len(set(names)) != 1020 or set(names) != set(expected):
        raise ValueError("Submission must cover exactly the official 1020 test names")
    if masks.shape != (1020, 128, 128) or masks.dtype != torch.uint8 or not ((masks == 0) | (masks == 1)).all():
        raise ValueError("Submission masks must be binary uint8 [1020,128,128]")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    out = Path(config["output"])
    inference = config["inference"]
    ss = json.loads((Path(config["packed"]) / "frozen_stegastamp.json").read_text())
    metrics = {"stegastamp": ss["miou"]}
    for family in ("trustmark", "pixelseal"):
        done = json.loads((out / family / "completed.json").read_text())
        if done["best_development_miou"] is None:
            raise ValueError(f"No completed validation for {family}")
        metrics[family] = done["best_development_miou"]
    mean = sum(metrics.values()) / 3
    if mean < .80:
        write_json(out / "inference.json", {"status": "skipped_below_0.80", "mean": mean, "families": metrics})
        return
    # Counterparts are supplied in the same frozen member order as preparation.
    if len(inference["dino_test"]) != len(ss["dino_heldout_sha256"]):
        raise ValueError("DINO test ensemble must match frozen held-out ensemble membership")
    if inference["dino_heldout_sha256"] != ss["dino_heldout_sha256"]:
        raise ValueError("Inference recipe is not the frozen development recipe")
    for path, expected in zip(inference["dino_test"], inference["dino_test_sha256"]):
        if digest(path) != expected:
            raise ValueError(f"Changed frozen test maps: {path}")
    if len(inference["dino_test_sha256"]) != len(inference["dino_test"]):
        raise ValueError("Every test map requires a frozen SHA256")
    destination = out / "submission.pt"
    if destination.exists():
        raise FileExistsError("Preserving existing submission artifact")
    from wmloc.data import LocalisationDataset
    ds = LocalisationDataset(inference["data"], "test")
    expected = [r["image_name"] for r in ds.meta]
    names, masks = [], []
    bundle = Path(inference["bundle"])
    for family in ("trustmark", "pixelseal", "stegastamp"):
        items = [ds[i] for i, r in enumerate(ds.meta) if r["method"] == family]
        fnames = [r["image_name"] for r in items]
        if len(fnames) != 340:
            raise ValueError(f"Expected 340 test images for {family}")
        maps = current_test(bundle, family, fnames, inference["dino_test"])
        if family == "stegastamp":
            candidate = maps["current_mask"]
        else:
            saved = torch.load(out / family / "selected.pt", map_location="cpu", weights_only=False)
            # Exploratory trials may change a nonparameter input scale. Rebuild
            # from the selected checkpoint's own recipe, not the latest trial.
            model_config = {**config, "highpass_gain": saved.get("config", {}).get("highpass_gain", 1.0)}
            model = build_model(family, model_config, "cuda:0")
            model.load_state_dict(saved["ema"], strict=True)
            del saved
            real = SimpleNamespace(names=fnames,
                images=torch.from_numpy(np.stack([r["image"].transpose(2, 0, 1) for r in items])).cuda(),
                bits=torch.from_numpy(np.stack([r["message"] for r in items]).astype(np.float32)).cuda())
            probability = predict(model, real, config.get("eval_batch", 4))
            candidate = fuse(family, probability, maps, config.get("trustmark_blend", .5))
            np.savez(out / family / "test.npz", names=fnames, probabilities=probability, masks=candidate)
            del real, model
            gc.collect()
            torch.cuda.empty_cache()
        names.extend(fnames)
        masks.extend(candidate)
    masks = torch.tensor(np.stack(masks), dtype=torch.uint8)
    validate_submission(names, masks, expected)
    from watermark_localization.submission import save_submission
    save_submission(destination, names, masks, expected)
    torch.load(destination, map_location="cpu", weights_only=True)
    write_json(out / "inference.json", {"status": "artifact_created", "mean": mean, "families": metrics,
               "submission_sha256": digest(destination), "submitted": False,
               "protocol": "Reused developmental folds; selected checkpoints; not hidden-final accuracy"})


if __name__ == "__main__":
    main()
