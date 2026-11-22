"""Exact-embedder uint8 signal and local embedding vs large-source fragment controls."""
import argparse
import io
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from wmloc.watermarks import load_watermark
from .common import *
from .decoder_probe import decode, tensor


def covers(n):
    path = ROOT/"experiments/synthetic_pretrain_2k/results/sources/val2017.zip"
    with zipfile.ZipFile(path) as z:
        for name in sorted(k for k in z.namelist() if k.endswith(".jpg"))[:n]:
            im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            w, h = im.size; side = min(w, h)
            im = im.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2)).resize((512, 512), Image.Resampling.LANCZOS)
            yield name, np.array(im)


def signal_stats(clean, marked, selector=None):
    delta = marked.astype(np.float64)-clean.astype(np.float64)
    y = delta @ np.array([.299, .587, .114])
    cb = (delta[..., 2]-y)/1.772; cr = (delta[..., 0]-y)/1.402
    if selector is None:
        selector = np.ones(clean.shape[:2], bool)
    return {"pixels": int(selector.sum()), "mae_rgb_uint8": float(np.abs(delta[selector]).mean()),
            "rms_rgb_uint8": float(np.sqrt(np.mean(delta[selector]**2))),
            "zero_pixel_fraction": float((delta[selector] == 0).all(1).mean()),
            "zero_channel_fraction": float((delta[selector] == 0).mean()),
            "mae_Y": float(abs(y[selector]).mean()), "mae_Cb": float(abs(cb[selector]).mean()), "mae_Cr": float(abs(cr[selector]).mean()),
            "chroma_energy_fraction": float((cb[selector]**2+cr[selector]**2).sum()/max((y[selector]**2+cb[selector]**2+cr[selector]**2).sum(), 1e-20))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", choices=FAMILIES, required=True); ap.add_argument("--covers", type=int, default=16); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out = OUT/"signal"/a.family; out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); wm = load_watermark(a.family, ROOT/"checkpoints", a.device)
    for v in vars(wm).values():
        if isinstance(v, torch.nn.Module):
            v.eval().requires_grad_(False)
    rows, controls = [], []
    for i, (source, clean) in enumerate(covers(a.covers)):
        bits = np.random.default_rng(42100+i).integers(0, 2, wm.nbits)
        with torch.inference_mode():
            full = wm.embed(clean, bits)
        for side in (48, 64, 128, 512):
            lo, hi = (512-side)//2, (512+side)//2
            crop = clean[lo:hi, lo:hi]
            with torch.inference_mode():
                marked = full if side == 512 else wm.embed(crop.copy(), bits)
            y = crop.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
            grad = np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1))
            selectors = {"all": np.ones((side, side), bool), "smooth_q25": grad <= np.quantile(grad, .25), "textured_q75": grad >= np.quantile(grad, .75)}
            if a.family == "pixelseal":
                with torch.inference_mode():
                    jnd = wm.model.attenuation.heatmaps(tensor(crop, a.device)).mean(1)[0].cpu().numpy()
                selectors["low_jnd_q25"] = jnd <= np.quantile(jnd, .25)
                selectors["high_jnd_q75"] = jnd >= np.quantile(jnd, .75)
            for label, selector in selectors.items():
                rows.append({"source": source, "clean_sha256": hashlib.sha256(crop.tobytes()).hexdigest(), "family": a.family,
                             "embed_side": side, "stratum": label, **signal_stats(crop, marked, selector)})
            # Matched geometry: local encoded square vs same square cut from full encoded image.
            for regime, patch in (("local_embedding", marked), ("large_source_fragment", full[lo:hi, lo:hi]), ("clean", crop)):
                with torch.inference_mode():
                    logits = decode(wm, patch, "bbox256", a.device)
                controls.append({"source": source, "family": a.family, "side": side, "regime": regime,
                                 "bit_accuracy": float(((logits > 0) == bits).mean())})
            if i < 4 and side in (64, 512):
                delta = marked.astype(np.int16)-crop.astype(np.int16)
                panels = [crop, marked, np.clip(delta*16+128, 0, 255).astype(np.uint8)]
                canvas = Image.new("RGB", (768, 256))
                for k, panel in enumerate(panels):
                    canvas.paste(Image.fromarray(panel).resize((256, 256)), (k*256, 0))
                canvas.save(out/f"example_{i}_side{side}.png")
        save(out/"per_cover.json", rows); save(out/"controls.json", controls)
        print(a.family, "signal", i+1, "/", a.covers, "seconds", round(time.monotonic()-start, 1), flush=True)
    summaries = []
    for side in (48, 64, 128, 512):
        for stratum in sorted({r["stratum"] for r in rows}):
            rr = [r for r in rows if r["embed_side"] == side and r["stratum"] == stratum]
            summaries.append({"side": side, "stratum": stratum, "n": len(rr), **{k: mean(rr, k) for k in rows[0] if k not in ("source", "clean_sha256", "family", "embed_side", "stratum")}})
    cs = [{"side": side, "regime": regime, "bit_accuracy": mean([r for r in controls if r["side"] == side and r["regime"] == regime], "bit_accuracy")}
          for side in (48, 64, 128, 512) for regime in ("local_embedding", "large_source_fragment", "clean")]
    save(out/"summary.json", {"n_covers": a.covers, "signal": summaries, "controls": cs, "seconds": time.monotonic()-start,
         "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "code_sha256": sha(__file__), "wrapper_sha256": sha(ROOT/"wmloc/watermarks.py"),
         "protocol": "Diagnostic controls on first sorted COCO val2017 covers already available locally; same covers/messages for both methods. Not original cluster training realizations, no training. Exact official FP32 embedding/uint8, unchanged strength. Y/Cb/Cr are linear BT.601 residuals in 0..255 units. Low-JND measured on original embed crop."})
    csv_rows(out/"per_cover.csv", rows); csv_rows(out/"controls.csv", controls)


if __name__ == "__main__":
    main()
