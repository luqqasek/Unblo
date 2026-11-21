"""Exact uint8 embedding cache and provenance-based, same-content debug recipes.

The debug stage deliberately uses same-content hard composites only. Expansion
to cross-image/generated edits must be explicitly validated before stage 2.
"""
import argparse
import json
import importlib.metadata
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark
from .common import ROOT, PROJECT, atomic_json, append_jsonl, jsonl, rgb_tensor, sha256


def rectangle(rng, fraction, aspect_limit, side=512):
    ratio = np.exp(rng.uniform(-np.log(aspect_limit), np.log(aspect_limit)))
    w = int(np.clip(round(np.sqrt(fraction*side*side*ratio)), 1, side))
    h = int(np.clip(round(fraction*side*side/w), 1, side))
    # Integer rounding at very small areas must not break the declared bound.
    if w/h > aspect_limit:
        h = int(np.ceil(w/aspect_limit))
    if h/w > aspect_limit:
        w = int(np.ceil(h/aspect_limit))
    x = int(rng.integers(0, side-w+1)); y = int(rng.integers(0, side-h+1))
    return [x, y, x+w, y+h]


def rect_mask(rect):
    x0, y0, x1, y1 = rect
    m = np.zeros((512, 512), bool)
    m[y0:y1, x0:x1] = True
    return m


def minority_mask(rng, fraction, shape="rectangle"):
    """Procedural geometry independent of real competition masks."""
    if shape == "rectangle":
        return rect_mask(rectangle(rng, fraction, 16))
    if shape == "strip":
        return rect_mask(rectangle(rng, fraction, 512))
    # Rank a smooth random field to obtain exact-area irregular/disconnected masks.
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(rng.normal(size=(512, 512)).astype(np.float32),
                            sigma=float(rng.uniform(3, 24)))
    n = max(1, min(field.size-1, int(round(fraction*field.size))))
    cut = np.argpartition(field.ravel(), -n)[-n:]
    m = np.zeros(field.size, bool); m[cut] = True
    return m.reshape(512, 512)


def provenance_composite(clean, marked, support):
    if clean.dtype != np.uint8 or marked.dtype != np.uint8:
        raise TypeError("Composites must preserve exact embedded uint8 pixels")
    return np.where(support[..., None], marked, clean).astype(np.uint8)


def pair_views(clean, marked, row, rng, pair_kind=None, simple=False):
    """Return primary and optional companion; metadata never enters the model.

    For local-crop encoding support is constrained to the actually encoded R.
    For large-source remnants supports may be disconnected or thin.
    """
    support_r = rect_mask(row["rectangle"])
    kind = row["primary_kind"]
    if kind == "clean":
        support = np.zeros((512, 512), bool)
    elif kind == "marked":
        if not support_r.all():
            raise ValueError("Fully marked controls require a full-image encoded realization")
        support = support_r.copy()
    elif row["regime"] == "local_crop":
        support = support_r.copy()
    else:
        shape = "rectangle" if simple else str(rng.choice(["rectangle", "strip", "components"]))
        minor = minority_mask(rng, row["minority_fraction"], shape)
        support = (~minor if row["clean_minority"] else minor) & support_r
    supports = [support]
    if pair_kind == "clean_twin":
        supports.append(np.zeros_like(support))
    elif pair_kind == "move":
        # Valid only within R, and only if moving preserves area there.
        if not support_r.all():
            raise ValueError("Movement pairs require an encoded base covering the full destination")
        offset = (int(rng.integers(64, 449)), int(rng.integers(64, 449)))
        supports.append(np.roll(support, offset, axis=(0, 1)))
    elif pair_kind == "complement":
        supports.append(support_r & ~support)
    images = [provenance_composite(clean, marked, m) for m in supports]
    # Safe consistency neighborhoods are measured on OBSERVED pixels, not labels.
    safe = None
    if len(images) == 2:
        same = torch.from_numpy(np.all(images[0] == images[1], axis=2)).float()[None, None]
        # 512 ->128 cell has four input centers; include local RF radius 13 on both
        # sides of the entire 4x4 cell. Zero-padding excludes image-edge uncertainty.
        same = F.pad(same, (13, 13, 13, 13), value=0)
        bad = F.max_pool2d(1-same, kernel_size=30, stride=4)
        safe = (bad[0, 0] == 0)
    masks = [downsample_mask(m) for m in supports]
    return images, masks, safe


class EncodedDataset:
    def __init__(self, manifest, method=None):
        self.manifest = Path(manifest)
        self.rows = [r for r in jsonl(manifest) if method is None or r["method"] == method]
        if not self.rows:
            raise ValueError(f"No encoded examples in {manifest}")
        self.base = self.manifest.parent
        self.cache = {}

    def preload(self, limit=8192, workers=8):
        from concurrent.futures import ThreadPoolExecutor
        count = min(limit, len(self.rows))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, sample in enumerate(pool.map(self.load, range(count))):
                self.cache[i] = sample
        return count

    def load(self, index):
        if index in self.cache:
            return self.cache[index]
        row = self.rows[index]
        with Image.open(self.base/row["clean_path"]) as image:
            clean = np.asarray(image.convert("RGB"))
        with Image.open(self.base/row["marked_path"]) as image:
            marked = np.asarray(image.convert("RGB"))
        return clean, marked, row

    def batch(self, rng, views=16, curriculum_update=0, paired=True, fixed_indices=None):
        if views % 4:
            raise ValueError("Effective batch must be a multiple of four for exact half paired views")
        images, masks, pairs, safe, meta = [], [], [], [], []
        groups = views*3//4 if paired else views
        pair_groups = views//4 if paired else 0
        for group in range(groups):
            index = int(rng.integers(len(self.rows))) if fixed_indices is None else fixed_indices[group % len(fixed_indices)]
            clean, marked, row = self.load(index)
            pair_kind = None
            if group < pair_groups:
                pair_kind = (str(rng.choice(["move", "clean_twin", "complement"]))
                             if row["rectangle"] == [0, 0, 512, 512] else "clean_twin")
            recipe_seed = int(rng.integers(2**63))
            xs, ys, consistency = pair_views(clean, marked, row, np.random.default_rng(recipe_seed),
                                           pair_kind, simple=curriculum_update < 5000)
            if len(xs) == 2:
                pairs.append((len(images), len(images)+1)); safe.append(consistency)
            images.extend(rgb_tensor(x) for x in xs)
            masks.extend(torch.from_numpy(y.copy()).float() for y in ys)
            for j, y in enumerate(ys):
                meta.append({"source_id": row["source_id"], "realization_id": row["realization_id"],
                             "primary": j == 0, "pair_kind": pair_kind, "regime": row["regime"],
                             "recipe_seed": recipe_seed, "view_index": j,
                             "fg_fraction": float(y.mean()), "primary_kind": row["primary_kind"]})
        return torch.stack(images), torch.stack(masks), pairs, torch.stack(safe) if safe else None, meta


def generate(a):
    rng = np.random.default_rng(a.seed)
    source_manifest_hash = sha256(a.sources)
    sources = [r for r in jsonl(a.sources) if not r.get("excluded", False)]
    source_provenance = json.loads(a.sources.with_name("source_provenance.json").read_text())
    if len(source_provenance.get("competition_train_sha256", {})) != 510:
        raise ValueError("Source manifest has not been screened against all 510 competition train images")
    if not sources or any(r.get("screened_competition_n") != 510 for r in sources):
        raise ValueError("Every accepted source row must have completed the full 510-image duplicate screen")
    if any(r["partition"] != a.partition for r in sources):
        raise ValueError("Mixed source partitions")
    if a.partition == "locked" and not a.unlock_locked:
        raise ValueError("Locked source suite is closed until recipe freeze")
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out/"encoded.jsonl"
    previous = jsonl(manifest) if manifest.exists() else []
    if previous:
        for i, r in enumerate(previous):
            if r["method"] != a.method or r["seed"] != a.seed or r["partition"] != a.partition:
                raise ValueError("Resume configuration differs")
    # Use the exact official wrapper in FP32, without changing strength/config.
    torch.set_num_threads(a.threads)
    wm = load_watermark(a.method, models_dir=a.models, device=a.device)
    if a.method == "stegastamp":
        model_paths = sorted((a.models/"stegastamp").glob("*.onnx"))
    elif a.method == "pixelseal":
        model_paths = [a.models/"pixelseal/checkpoint.pth", a.models/"pixelseal_card.yaml"]
    else:
        import trustmark
        if importlib.metadata.version("trustmark") != "0.8.0":
            raise ValueError("Competition TrustMark requires exactly trustmark==0.8.0")
        model_paths = sorted((Path(trustmark.__file__).parent/"models").glob("*"))
        model_paths = [f for f in model_paths if f.is_file()]
    atomic_json(a.out/"embedder_provenance.json", {
        "method": a.method, "nbits": wm.nbits, "unchanged_official_wrapper": True,
        "wrapper_sha256": sha256(ROOT/"wmloc/watermarks.py"),
        "model_files": {str(f): sha256(f) for f in model_paths},
        "trustmark_version": importlib.metadata.version("trustmark") if a.method == "trustmark" else None,
        "trustmark_settings": {"variant": "Q", "secret_len": 100, "use_ECC": False, "MODE": "binary"} if a.method == "trustmark" else None,
    })
    for module in vars(wm).values():
        if isinstance(module, torch.nn.Module):
            module.requires_grad_(False).eval().float()
    if a.method == "stegastamp" and a.device.startswith("cuda"):
        if "CUDAExecutionProvider" not in wm.sess.get_providers():
            raise RuntimeError("StegaStamp GPU generation requested but ONNX CUDA provider is unavailable")
    bins = json.loads((PROJECT/"config.json").read_text())["minority_bins"]
    start = time.monotonic()
    # Re-create RNG stream identically on resume; only completed rows skip embedding.
    for i in range(a.count):
        source = sources[i % len(sources)]
        bits = rng.integers(0, 2, wm.nbits).tolist()
        u = rng.random(); kind = "clean" if u < .05 else "marked" if u < .1 else "partial"
        lo, hi, _ = bins[int(rng.choice(4, p=[b[2] for b in bins]))]
        fraction = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        polarity = bool(rng.integers(2))
        regime = "local_crop" if kind == "partial" and not polarity and rng.random() < .5 else "large_source_remnant"
        rect = rectangle(rng, fraction, 2 if a.method == "trustmark" else 16) if regime == "local_crop" else [0, 0, 512, 512]
        crop_seed = int(rng.integers(2**31))
        if i < len(previous):
            expected = previous[i]
            if expected["source_id"] != source["source_id"] or expected["message"] != bits or expected["rectangle"] != rect:
                raise RuntimeError("Resume source order or RNG stream differs")
            if not (a.out/expected["marked_path"]).exists():
                raise RuntimeError("Completed encoded image is missing")
            continue
        local_rng = np.random.default_rng(crop_seed)
        with Image.open(source["path"]) as image:
            image = image.convert("RGB")
            w, h = image.size; side = min(w, h)
            x = int(local_rng.integers(w-side+1)); y = int(local_rng.integers(h-side+1))
            crop_box = [x, y, x+side, y+side]
            clean = np.array(image.crop(crop_box).resize((512, 512), Image.Resampling.LANCZOS))
        x0, y0, x1, y1 = rect
        with torch.inference_mode(), torch.autocast(a.device.split(":")[0], enabled=False):
            patch = wm.embed(clean[y0:y1, x0:x1].copy(), bits)
        if patch.dtype != np.uint8 or patch.shape != clean[y0:y1, x0:x1].shape:
            raise RuntimeError("Official wrapper returned unexpected pixels")
        marked = clean.copy(); marked[y0:y1, x0:x1] = patch
        folder = a.out/f"{i//1000:05d}"; folder.mkdir(exist_ok=True)
        cleanpath = folder/f"{i:07d}.clean.png"; markedpath = folder/f"{i:07d}.marked.png"
        Image.fromarray(clean).save(cleanpath, compress_level=1)
        Image.fromarray(marked).save(markedpath, compress_level=1)
        if i < 16:
            if not np.array_equal(np.asarray(Image.open(cleanpath)), clean) or not np.array_equal(np.asarray(Image.open(markedpath)), marked):
                raise RuntimeError("PNG round-trip changed exact uint8 signal")
        delta = patch.astype(np.float32)-clean[y0:y1, x0:x1].astype(np.float32)
        row = {"realization_id": f"{a.method}:{a.partition}:{a.seed}:{i}", "method": a.method,
               "seed": a.seed, "source_id": source["source_id"], "source_sha256": source["sha256"],
               "partition": a.partition, "source_crop": crop_box, "rectangle": rect, "message": bits,
               "regime": regime, "primary_kind": kind, "minority_fraction": fraction,
               "clean_minority": polarity, "crop_resolution": [y1-y0, x1-x0],
               "clean_path": str(cleanpath.relative_to(a.out)), "marked_path": str(markedpath.relative_to(a.out)),
               "clean_sha256": sha256(cleanpath), "marked_sha256": sha256(markedpath),
               "residual_uint8": {"rms": float(np.sqrt(np.mean(delta**2))), "max_abs": float(abs(delta).max()),
                                  "zero_fraction": float((delta == 0).mean())},
               "embedding_settings": "unchanged official wmloc wrapper; FP32; exact native clipping/quantization",
               "actual_embedding_calls": 1, "construction": "same_content_debug"}
        if i < a.decode_checks:
            tensor = rgb_tensor(patch)[None].to(a.device)
            decoded = wm.decode(tensor).detach().cpu().numpy().reshape(-1)
            row["patch_bit_accuracy"] = float((decoded == bits).mean())
            wrong = np.random.default_rng(a.seed+i).integers(0, 2, wm.nbits)
            row["wrong_message_bit_accuracy"] = float((decoded == wrong).mean())
        append_jsonl(manifest, row)
        if (i+1) % 25 == 0 or i == 0:
            print(json.dumps({"encoded_realizations": i+1, "elapsed_s": time.monotonic()-start,
                              "last_residual": row["residual_uint8"]}), flush=True)
    atomic_json(a.out/"generation_summary.json", {
        "encoded_realizations": a.count, "distinct_sources": min(a.count, len(sources)),
        "embedding_calls_total": a.count, "partition": a.partition,
        "construction": "same_content_debug", "full_mixture_validated": False,
        "seconds_this_invocation": time.monotonic()-start, "source_manifest_sha256": source_manifest_hash,
        "wrapper_sha256": sha256(ROOT/"wmloc/watermarks.py"),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partition", choices=["train", "dev", "locked"], required=True)
    p.add_argument("--method", choices=["pixelseal", "trustmark", "stegastamp"], required=True)
    p.add_argument("--models", type=Path, default=ROOT/"checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--count", type=int, default=2500)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--decode-checks", type=int, default=16)
    p.add_argument("--unlock-locked", action="store_true")
    args=p.parse_args()
    if args.seed is None:
        args.seed={"train":42,"dev":424242,"locked":1234567}[args.partition]
    generate(args)


if __name__ == "__main__":
    main()
