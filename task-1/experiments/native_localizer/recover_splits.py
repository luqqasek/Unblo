"""Reconstruct the original three outer folds from original code + metadata order.

Ignored prior manifests are not in Git. This reconstruction is explicitly labelled;
if provided later, byte-independent list equality must be checked before CV runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from wmloc.data import LocalisationDataset
from .common import atomic_json, sha256


def original_ss_inner(fit, targets, flags):
    # Exact algorithm from stegastamp_stable_selection/run.py at 7e811ae.
    strata = flags.astype(int)*4+np.digitize(targets.mean((1,2)), [.25,.5,.75])
    rng = np.random.default_rng(42)
    parts = [[], [], []]; cursor = 0
    for label in sorted(set(strata[fit])):
        group = rng.permutation([i for i in fit if strata[i] == label])
        for i in group:
            parts[cursor % 3].append(int(i)); cursor += 1
    return [np.array(sorted(part), dtype=int) for part in parts]


def build_splits(dataset, method):
    meta = [r for r in dataset.meta if r["method"] == method]
    names = [r["image_name"] for r in meta]
    if len(names) != 170 or len(set(names)) != 170:
        raise ValueError("Require exactly 170 unique images per family")
    masks = np.stack([dataset.m128[name] for name in names])
    flags = np.array([r["full_watermark"] for r in meta])
    parts = np.array_split(np.random.default_rng(42).permutation(len(names)),3)
    splits = []
    for fold, validation in enumerate(parts):
        fit = np.setdiff1d(np.arange(len(names)), validation)
        if method == "stegastamp":
            inner = original_ss_inner(fit, masks, flags)
        else:
            shuffled = np.random.default_rng(42+fold).permutation(fit)
            inner = [shuffled[:int(np.ceil(.2*len(fit)))]]
        splits.append({"fold": fold+1, "train": [names[i] for i in fit],
                       "validation": [names[i] for i in validation],
                       "inner_validation": [[names[i] for i in ids] for ids in inner]})
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prior", type=Path, help="Optional consolidated prior manifests for exact-list comparison")
    a = p.parse_args()
    ds = LocalisationDataset(a.data,"train")
    splits = {m:build_splits(ds,m) for m in ("pixelseal","trustmark","stegastamp")}
    if a.prior:
        prior = json.loads(a.prior.read_text())
        for method in splits:
            for old, new in zip(prior[method], splits[method]):
                for key in ("train","validation","inner_validation"):
                    if old[key] != new[key]:
                        raise ValueError(f"Prior split mismatch: {method} fold {new['fold']} {key}")
    atomic_json(a.out, {"families":splits, "provenance":{
        "status":"reconstructed from recorded original algorithms; ignored original manifests not yet available" if not a.prior else "verified against supplied prior",
        "reference_commit":"7e811ae3be46a8bb7b4ef549bb645c8557e0df72",
        "metadata_sha256":sha256(a.data/"train/metadata.json"),
        "mask_grid_sha256":sha256(a.data/"train/masks_128.npz"),
        "outer_seed":42, "source_grouping":"no exact or pHash<=4 candidates among 510 train images; edited siblings cannot be excluded",
    }})


if __name__ == "__main__":
    main()
