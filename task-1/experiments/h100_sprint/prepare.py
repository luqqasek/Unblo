"""Pack existing assets before renting a GPU. Never embeds or downloads images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import aligned, digest, verify_split, write_json


def read_rows(manifests, family, partition):
    rows = []
    for manifest in manifests:
        manifest = Path(manifest)
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row["method"] != family or row["partition"] != partition:
                raise ValueError(f"Wrong family/partition in {manifest}")
            if not row.get("source_sha256") or not row.get("source_id"):
                raise ValueError("Source identities and hashes are required")
            rows.append((row, manifest.parent))
    return rows


def disjoint_sources(train, dev):
    for key in ("source_id", "source_sha256"):
        if {r[key] for r, _ in train} & {r[key] for r, _ in dev}:
            raise ValueError(f"Synthetic source leakage by {key}")


def pack_pairs(rows, destination, workers=1):
    destination.mkdir(parents=True)
    arrays = {k: np.lib.format.open_memmap(destination / f"{k}.npy", mode="w+",
               dtype=np.uint8, shape=(len(rows), 3, 512, 512)) for k in ("clean", "marked")}
    def load(pair):
        row, folder = pair
        x0, y0, x1, y1 = row["rectangle"]
        if not (0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512):
            raise ValueError("Invalid encoded rectangle")
        decoded = {}
        for key in arrays:
            path = folder / row[f"{key}_path"]
            if digest(path) != row[f"{key}_sha256"]:
                raise ValueError(f"Cached PNG hash mismatch: {path}")
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            if rgb.shape != (512, 512, 3):
                raise ValueError(f"Wrong cache dimensions: {path}")
            decoded[key] = rgb.transpose(2, 0, 1)
        support = np.zeros((512, 512), bool)
        support[y0:y1, x0:x1] = True
        if not np.array_equal(decoded["clean"][:, ~support], decoded["marked"][:, ~support]):
            raise ValueError("Marked pixels extend beyond documented encoded support")
        return decoded
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, decoded in enumerate(executor.map(load, rows)):
            for key in arrays:
                arrays[key][i] = decoded[key]
    for a in arrays.values():
        a.flush()
    write_json(destination / "rows.json", [r for r, _ in rows])


def pack_real(data, family, split, destination):
    metadata = json.loads((data / "train/metadata.json").read_text())
    metadata = {r["image_name"]: r for r in metadata if r["method"] == family}
    train, valid = verify_split(split.get("train", split.get("training", [])), split["validation"])
    if set(train + valid) != set(metadata):
        raise ValueError("Real split does not exactly cover this family's 170 images")
    with np.load(data / "train/masks_128.npz") as archive:
        targets = {str(n): m for n, m in zip(archive["names"], archive["masks"])}
    for label, names in (("real_train", train), ("real_valid", valid)):
        images = []
        for name in names:
            with Image.open(data / f"train/images/{name}.png") as image:
                rgb = np.asarray(image.convert("RGB"))
            if rgb.shape != (512, 512, 3):
                raise ValueError("Real images must remain at native resolution")
            images.append(rgb.transpose(2, 0, 1))
        np.savez(destination / f"{label}.npz", names=names, images=np.stack(images),
                 targets=np.stack([targets[n] for n in names]),
                 bits=np.array([[int(v) for v in metadata[n]["message"]] for n in names], np.uint8))
    write_json(destination / "split.json", {"train": train, "validation": valid})
    return valid


def pack_maps(bundle, family, names, dino_paths, destination):
    from experiments.h100_hour.h2_decode import expected_miou_decode, miou
    filename = "pixelseal-validation57.npz" if family == "pixelseal" else "trustmark-oof.npz"
    with np.load(bundle / "maps" / filename) as archive:
        p = aligned(archive, names, "probabilities").astype(np.float32)
        base = aligned(archive, names, "masks").astype(np.uint8)
        gt = aligned(archive, names, "targets")
    with np.load(destination / "real_valid.npz") as real:
        if not np.array_equal(gt, real["targets"]):
            raise ValueError("Baseline targets do not match official real validation targets")
    maps = []
    for path in dino_paths:
        with np.load(path) as archive:
            maps.append(aligned(archive, names, "maps").astype(np.float32))
    dino = np.mean(maps, axis=0)
    if dino.shape != (57, 2, 128, 128) or not np.isfinite(dino).all():
        raise ValueError("Expected held-out DINO hole/partial probabilities [57,2,128,128]")
    route_b = base.mean((1, 2)) >= .5
    q = np.where(route_b[:, None, None], 1 - dino[:, 0], dino[:, 1])
    w = .5 if family == "pixelseal" else .2
    current = expected_miou_decode((1 - w) * p + w * q)
    # Accepted OOF probabilities already contain NRG crop additions, while test
    # probabilities are raw. Match the candidate validation/test fusion exactly:
    # blend raw probabilities first, then restore the saved verified crop pixels.
    raw = p
    if family == "trustmark":
        with np.load(bundle / "cluster/region-runs/trustmark/objects-adapt/fold1/maps.npz") as archive:
            raw = aligned(archive, names, "probabilities").astype(np.float32)
            if not np.array_equal(aligned(archive, names, "targets"), gt):
                raise ValueError("Raw TrustMark fold targets do not match accepted baseline")
    corrections = (base.astype(bool) & ~(raw >= .5)) if family == "trustmark" else np.zeros_like(base, bool)
    np.savez(destination / "validation_maps.npz", names=names, base_prob=raw, base_mask=base,
             current_prob=(1 - w) * p + w * q,
             dino_partial=dino[:, 1], current_mask=current, corrections=corrections)
    return {"baseline_miou": float(miou(current, gt).mean()), "names": names,
            "baseline_source": filename, "dino_files": {str(p): digest(p) for p in dino_paths},
            "corrections": "Saved accepted mask additions outside base threshold; no new NRGMark scan"}


def freeze_stegastamp(bundle, dino_paths):
    from experiments.h100_hour.h2_decode import miou
    split = json.loads((bundle / "cluster/next-ideas-extended-runs/stegastamp/fold1/split.json").read_text())
    names = split["validation"]
    with np.load(bundle / "maps/stegastamp-oof.npz") as d:
        base, prob, target = [aligned(d, names, k) for k in ("masks", "probabilities", "targets")]
    members = []
    for path in dino_paths:
        with np.load(path) as d:
            members.append(aligned(d, names, "maps").astype(np.float32))
    m = np.mean(members, 0)
    route = base.mean((1, 2)) >= .5
    q = np.where(route[:, None, None], 1 - m[:, 0], m[:, 1])
    masks = (.3 * prob + .7 * q) > .5
    return {"miou": float(miou(masks, target).mean()), "names": names,
            "recipe": "DINO member average, weight 0.7, baseline-mask routing, threshold >0.5",
            "dino_heldout_sha256": [digest(p) for p in dino_paths], "n": len(names)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="Existing cluster cache root")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--bundle", type=Path, required=True, help="Verified submission-3188 bundle")
    p.add_argument("--dino-heldout", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--family", choices=("trustmark", "pixelseal"), nargs="+", default=["trustmark", "pixelseal"])
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Use a new output directory; packed data is never overwritten")
    all_rows = {}
    for family in a.family:
        manifests = [a.root / f"encoded/{family}/pilot-train/encoded.jsonl"]
        dev_manifests = [a.root / f"encoded/{family}/pilot-dev/encoded.jsonl"]
        tr, va = read_rows(manifests, family, "train"), read_rows(dev_manifests, family, "dev")
        disjoint_sources(tr, va)
        all_rows[family] = (tr, va, manifests + dev_manifests)
    a.out.mkdir(parents=True)
    write_json(a.out / "frozen_stegastamp.json", freeze_stegastamp(a.bundle, a.dino_heldout))
    for family, (tr, va, manifests) in all_rows.items():
        folder = a.out / family
        folder.mkdir()
        pack_pairs(tr, folder / "synthetic_train")
        pack_pairs(va, folder / "synthetic_dev")
        if family == "pixelseal":
            split_path = a.bundle / "cluster/triforensic-runs/pixelseal/unified-v1/split.json"
        else:
            split_path = a.bundle / "cluster/region-runs/trustmark/objects-adapt/fold1/split.json"
        split = json.loads(split_path.read_text())
        valid = pack_real(a.data, family, split, folder)
        maps = pack_maps(a.bundle, family, valid, a.dino_heldout, folder)
        manifest = {"family": family, "train_pairs": len(tr), "dev_pairs": len(va),
                    "source_overlap": 0, "split_sha256": digest(split_path), "maps": maps,
                    "source_manifests": {str(p): digest(p) for p in manifests},
                    "files": {str(p.relative_to(folder)): digest(p) for p in folder.rglob("*") if p.is_file()}}
        if family == "pixelseal":
            manifest["initial_checkpoint_sha256"] = digest(a.bundle / "cluster/triforensic-runs/pixelseal/unified-v1/resume.pt")
        write_json(folder / "manifest.json", manifest)
        print(json.dumps({"family": family, "train": len(tr), "dev": len(va), "baseline": maps["baseline_miou"]}), flush=True)


if __name__ == "__main__":
    main()
