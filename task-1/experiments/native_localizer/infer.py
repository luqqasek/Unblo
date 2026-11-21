"""Family-directed native RGB inference; no STN, resizing, TTA or upload."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wmloc.data import LocalisationDataset
from watermark_localization.submission import save_submission
from .common import atomic_json, rgb_tensor, sha256
from .model import NativeLocalizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", type=Path, required=True,
                   help="Frozen JSON: family -> {checkpoint, threshold}; requires recipe_frozen=true")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = json.loads(a.recipe.read_text())
    if recipe.get("recipe_frozen") is not True:
        raise ValueError("Freeze training/selection decisions before touching competition test predictions")
    if a.out.exists():
        raise FileExistsError("Preserve existing submissions; choose a new output filename")
    ds = LocalisationDataset(a.data, "test")
    if len(ds) != 1020:
        raise ValueError("Expected all 1020 test images")
    masks = torch.empty((len(ds), 128, 128), dtype=torch.uint8)
    checkpoint_hashes = {}
    for family in ("pixelseal", "trustmark", "stegastamp"):
        config = recipe["families"][family]
        path = Path(config["checkpoint"])
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["config"]["family"] != family:
            raise ValueError("Family checkpoint mismatch")
        model = NativeLocalizer(checkpointing=False).to(a.device).eval()
        model.load_state_dict(state["ema"] if config.get("use_ema", True) else state["model"])
        checkpoint_hashes[family] = sha256(path)
        indices = [i for i, row in enumerate(ds.meta) if row["method"] == family]
        with torch.inference_mode():
            for start in range(0, len(indices), a.batch):
                selected = indices[start:start+a.batch]
                images = torch.stack([rgb_tensor(ds[i]["image"]) for i in selected]).to(a.device)
                probability = model(images)["logits"].sigmoid().cpu()
                masks[selected] = (probability >= config["threshold"]).to(torch.uint8)
        del model
    names = [row["image_name"] for row in ds.meta]
    save_submission(a.out, names, masks, names)
    artifact = torch.load(a.out, map_location="cpu", weights_only=True)
    if not torch.equal(artifact["masks"], masks):
        raise RuntimeError("Saved submission differs after safe reload")
    atomic_json(a.out.with_suffix(".validation.json"), {
        "names": len(names), "shape": list(masks.shape), "dtype": str(masks.dtype),
        "bytes": a.out.stat().st_size, "sha256": sha256(a.out),
        "checkpoint_hashes": checkpoint_hashes, "recipe_sha256": sha256(a.recipe),
        "tta": False, "uploaded": False,
    })


if __name__ == "__main__":
    main()
