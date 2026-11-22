"""User-requested test masks from a measured edited checkpoint; no upload."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.h100_sprint.core import aligned, d4, digest, undo_d4, write_json
from experiments.h100_sprint.models import PixelSealBranch
from experiments.h100_hour.h2_decode import expected_miou_decode
from watermark_localization.submission import save_submission, validate_submission
from wmloc.data import LocalisationDataset


def predict_views(model, images, bits, tta):
    if tta == "identity":
        return model(images, bits).sigmoid()
    if tta != "d4":
        raise ValueError("Unknown test-time augmentation: " + tta)
    total = None
    for k in range(4):
        for flip in (False, True):
            p = undo_d4(model(d4(images, k, flip).contiguous(), bits).sigmoid(), k, flip)
            total = p if total is None else total + p
    return total / 8


def frozen_maps(bundle, family, names, dino_paths):
    with np.load(bundle / f"maps/{family}-test.npz") as z:
        prob = aligned(z, names, "probabilities").astype(np.float32)
        base = aligned(z, names, "masks").astype(np.uint8)
    if family == "trustmark":
        # Same accepted crop/scan correction used by h100_hour.assemble.test_maps.
        prob = np.where((base == 1) & ~(prob > .5), 1., prob).astype(np.float32)
    members = []
    for path in dino_paths:
        with np.load(path) as z:
            members.append(aligned(z, names, "maps").astype(np.float32))
    dino = np.mean(members, axis=0)
    route_b = base.mean((1, 2)) >= .5
    q = np.where(route_b[:, None, None], 1 - dino[:, 0], dino[:, 1])
    weight = {"trustmark": .2, "pixelseal": .5, "stegastamp": .7}[family]
    mixed = (1 - weight) * prob + weight * q
    masks = (mixed > .5).astype(np.uint8) if family == "stegastamp" else expected_miou_decode(mixed)
    return mixed, masks


@torch.inference_mode()
def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    selected = json.loads(a.summary.read_text())
    checkpoint = a.summary.parent.parent / selected["checkpoint"]
    assert digest(checkpoint) == selected["checkpoint_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PixelSealBranch(str(a.original), device="cpu", checkpointing=False)
    model.load_state_dict(saved["ema"], strict=True)
    if "replacement_head" in saved:
        from experiments.manual_weights.load_replacement import attach
        attach(model, saved)
    model.eval().requires_grad_(False).to("mps")
    del saved
    ds = LocalisationDataset(a.data, "test")
    if len(ds) != 1020 or ds.train:
        raise ValueError("Require the official 1020-image test set without targets")
    names = [r["image_name"] for r in ds.meta]
    assert len(set(names)) == 1020
    arrays, frozen, family_names = {}, {}, {}
    for family in ("trustmark", "pixelseal", "stegastamp"):
        fnames = [r["image_name"] for r in ds.meta if r["method"] == family]
        assert len(fnames) == 340
        arrays[family], frozen[family] = frozen_maps(a.bundle, family, fnames, a.dino)
        family_names[family] = fnames
    indices = [i for i, r in enumerate(ds.meta) if r["method"] == "pixelseal"]
    items = [ds[i] for i in indices]
    images = torch.from_numpy(np.stack([item["image"].transpose(2, 0, 1) for item in items]))
    bits = torch.from_numpy(np.stack([item["message"] for item in items])).float()
    # Check batch numerics and choose throughput from only the first four test inputs.
    x = images[:4].to("mps").float() / 255
    b = bits[:4].to("mps")
    reference = model(x[:1], b[:1]).cpu()
    timings = []
    for size in ((a.batch_size,) if a.batch_size else (1, 2, 4)):
        try:
            started = time.monotonic()
            z = model(x[:size], b[:size]); torch.mps.synchronize()
            seconds = time.monotonic() - started
            error = float((reference - z[:1].cpu()).abs().max())
            if error > .001:
                raise ValueError(f"Batch {size} changed logits by {error}")
            timings.append({"batch": size, "seconds_per_image": seconds / size, "max_logit_error": error})
            del z
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.mps.empty_cache()
            break
    batch = min(timings, key=lambda r: r["seconds_per_image"])["batch"]
    write_json(a.out / "benchmark.json", {"measurements": timings, "chosen_batch": batch})
    print(json.dumps({"benchmark": timings, "chosen_batch": batch}), flush=True)
    del x, b, reference
    probability = np.empty((340, 128, 128), np.float32)
    started = time.monotonic()
    for offset in range(0, 340, batch):
        x = images[offset:offset + batch].to("mps").float() / 255
        message = bits[offset:offset + batch].to("mps")
        probability[offset:offset + len(x)] = predict_views(model, x, message, a.tta).cpu().numpy()
        finished = offset + len(x)
        progress = {"images": finished, "total": 340, "seconds": time.monotonic() - started,
                    "seconds_per_image": (time.monotonic() - started) / finished}
        write_json(a.out / "progress.json", progress)
        if finished % 20 < batch or finished == 340:
            print(json.dumps(progress), flush=True)
    candidate = expected_miou_decode(.7 * arrays["pixelseal"] + .3 * probability)
    by_name = {n: m for f in ("trustmark", "stegastamp") for n, m in zip(family_names[f], frozen[f])}
    by_name.update(zip(family_names["pixelseal"], candidate))
    masks = torch.from_numpy(np.stack([by_name[n] for n in names])).to(torch.uint8)
    target = a.out / "submission.pt"
    save_submission(target, names, masks, names)
    reloaded = torch.load(target, map_location="cpu", weights_only=True)
    validate_submission(reloaded, names)
    assert target.stat().st_size < 20_000_000
    for family in ("trustmark", "stegastamp"):
        ix = [names.index(n) for n in family_names[family]]
        assert np.array_equal(masks[ix].numpy(), frozen[family])
    np.savez_compressed(a.out / "pixelseal-test.npz", names=family_names["pixelseal"], probabilities=probability, masks=candidate)
    receipt = {"checkpoint": str(checkpoint), "checkpoint_sha256": selected["checkpoint_sha256"],
               "manual_edit": selected["edit"], "artifact_sha256": digest(target), "bytes": target.stat().st_size,
               "images": 1020, "pixelseal_images": 340, "dtype": "uint8", "shape": list(masks.shape),
               "dino_sha256": {str(p): digest(p) for p in a.dino}, "device": "mps", "precision": "FP32", "tta": a.tta,
               "trustmark_stegastamp_frozen": True, "submitted": False,
               "authorization": "User explicitly requested verification submission after manual weight edits, overriding the earlier 0.80 inference gate",
               "source_selection": "Development evidence only; no public score used to choose this artifact"}
    write_json(a.out / "inference.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("summary", "original", "bundle", "data", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--dino", type=Path, action="append", required=True)
    p.add_argument("--tta", choices=("identity", "d4"), default="d4")
    p.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
