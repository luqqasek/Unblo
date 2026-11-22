"""Real geometry vs a deterministic replay of the current generator's geometry sampler."""
import argparse
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from experiments.native_localizer.generator import rectangle, rect_mask, minority_mask, pair_views
from wmloc.utils import downsample_mask
from .common import *
from .signal import covers


def geometry(mask, image=None):
    mask = np.asarray(mask, bool); area = int(mask.sum())
    if not area:
        return {"area512": 0., "fill_ratio": None, "perimeter_over_area": None, "components8": 0,
                "width_ridge_p10": None, "width_ridge_median": None, "thin_ridge_le4_fraction": None, "edge_alignment": None}
    ys, xs = np.where(mask)
    padded = np.pad(mask.astype(np.int8), 1)
    perimeter = np.abs(np.diff(padded, axis=0)).sum()+np.abs(np.diff(padded, axis=1)).sum()
    distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    ridge = mask & (distance >= ndi.maximum_filter(distance, size=3))
    widths = 2*distance[ridge]
    result = {"area512": area/mask.size, "fill_ratio": float(area/((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))),
              "perimeter_over_area": float(perimeter/area), "components8": int(ndi.label(mask, np.ones((3, 3)))[1]),
              "width_ridge_p10": float(np.quantile(widths, .1)), "width_ridge_median": float(np.median(widths)),
              "thin_ridge_le4_fraction": float((widths <= 4).mean()), "edge_alignment": None}
    if image is not None:
        gray = image.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
        grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        edge = grad >= np.quantile(grad, .8)
        boundary = mask ^ ndi.binary_erosion(mask)
        boundary[[0, -1], :] = False; boundary[:, [0, -1]] = False
        if boundary.any():
            result["edge_alignment"] = float(ndi.binary_dilation(edge, iterations=2)[boundary].mean())
    return result


def metadata_replay(family):
    """Same RNG calls as generate(seed42,count2500); source order does not affect geometry."""
    rng = np.random.default_rng(42)
    bins = read(ROOT/"experiments/native_localizer/config.json")["minority_bins"]
    rows = []
    for _ in range(2500):
        rng.integers(0, 2, 256 if family == "pixelseal" else 100).tolist()
        u = rng.random(); kind = "clean" if u < .05 else "marked" if u < .1 else "partial"
        lo, hi, _ = bins[int(rng.choice(4, p=[b[2] for b in bins]))]
        fraction = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        polarity = bool(rng.integers(2))
        regime = "local_crop" if kind == "partial" and not polarity and rng.random() < .5 else "large_source_remnant"
        rect = rectangle(rng, fraction, 2 if family == "trustmark" else 16) if regime == "local_crop" else [0, 0, 512, 512]
        rng.integers(2**31)
        rows.append(dict(rectangle=rect, primary_kind=kind, minority_fraction=fraction, clean_minority=polarity, regime=regime))
    return rows


def support_views(row, seed, pair_kind):
    rng = np.random.default_rng(seed); support_r = rect_mask(row["rectangle"])
    if row["primary_kind"] == "clean":
        support = np.zeros((512, 512), bool)
    elif row["primary_kind"] == "marked" or row["regime"] == "local_crop":
        support = support_r.copy()
    else:
        shape = str(rng.choice(["rectangle", "strip", "components"]))
        minor = minority_mask(rng, row["minority_fraction"], shape)
        support = (~minor if row["clean_minority"] else minor) & support_r
    result = [support]
    if pair_kind == "clean_twin":
        result.append(np.zeros_like(support))
    elif pair_kind == "move":
        offset = (int(rng.integers(64, 449)), int(rng.integers(64, 449)))
        result.append(np.roll(support, offset, axis=(0, 1)))
    elif pair_kind == "complement":
        result.append(support_r & ~support)
    return result


def replay(family, updates):
    rows = metadata_replay(family); rng = np.random.default_rng(42)
    blank = np.zeros((512, 512, 3), np.uint8)
    for update in range(updates):
        for group in range(24):
            idx = int(rng.integers(len(rows))); row = rows[idx]; pair_kind = None
            if group < 8:
                pair_kind = str(rng.choice(["move", "clean_twin", "complement"])) if row["rectangle"] == [0, 0, 512, 512] else "clean_twin"
            seed = int(rng.integers(2**63))
            supports = support_views(row, seed, pair_kind)
            if update == 0:
                _, expected, _ = pair_views(blank, blank, row, np.random.default_rng(seed), pair_kind, simple=False)
                assert all(np.array_equal(downsample_mask(m), y) for m, y in zip(supports, expected))
            for j, mask in enumerate(supports):
                yield mask, dict(update=update, source_index=idx, pair_kind=pair_kind, primary=j == 0, regime=row["regime"], rectangle=row["rectangle"], recipe_seed=seed)


def summarize(rows):
    out = {}
    for population in ("real", "generator_replay"):
        out[population] = {}
        for polarity in ("foreground", "minority"):
            out[population][polarity] = {}
            for label in ("all", "empty")+BUCKETS:
                rr = select([r for r in rows if r["population"] == population and r["polarity"] == polarity], label)
                values = {"n": len(rr)}
                for key in ("area512", "fill_ratio", "perimeter_over_area", "components8", "width_ridge_p10", "width_ridge_median", "thin_ridge_le4_fraction", "edge_alignment"):
                    v = [r[key] for r in rr if r[key] is not None]
                    values[key] = {"mean": float(np.mean(v)), "p10": float(np.quantile(v, .1)), "median": float(np.median(v)), "p90": float(np.quantile(v, .9))} if v else None
                out[population][polarity][label] = values
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--updates", type=int, default=32); a = p.parse_args()
    start = time.monotonic(); meta, gts = metadata(), masks128(); source_covers = list(covers(16))
    summaries = {}
    for family in FAMILIES:
        rows = []
        def add(mask, extra, image):
            for polarity in ("foreground", "minority"):
                selected = mask if polarity == "foreground" or mask.mean() <= .5 else ~mask
                rows.append({**extra, "polarity": polarity, **geometry(selected, image)})
        for name, m in meta.items():
            if m["method"] != family:
                continue
            mask = np.asarray(Image.open(ROOT/f"data/train/masks/{name}.png")) > 127
            im = np.asarray(Image.open(ROOT/f"data/train/images/{name}.png").convert("RGB"))
            add(mask, {"image_name": name, "population": "real", "family": family, "true_area": float(gts[name].mean()), "bucket": bucket(float(gts[name].mean())), "full_watermark": m["full_watermark"]}, im)
        for i, (mask, detail) in enumerate(replay(family, a.updates)):
            area = float(downsample_mask(mask).mean())
            # Same-content composite preserves scene boundaries. This cover sample is a proxy, not the original cluster source.
            im = source_covers[detail["source_index"] % len(source_covers)][1]
            add(mask, {"image_name": f"replay_{i}", "population": "generator_replay", "family": family, "true_area": area,
                       "bucket": bucket(area), "full_watermark": detail["rectangle"] == [0, 0, 512, 512] and area > .5, **detail}, im)
            if (i+1) % 256 == 0:
                print(family, "geometry views", i+1, "seconds", round(time.monotonic()-start, 1), flush=True)
        csv_rows(OUT/f"geometry_{family}.csv", rows)
        summaries[family] = summarize(rows)
        save(OUT/"geometry.json", {"families": summaries, "views_per_family": a.updates*32, "seconds": time.monotonic()-start,
             "protocol": "Real 512 masks; generator geometry replays code seed42/count2500, then first32 batches of32 (including paired views). Not archived actual masks. First24 groups compared bytewise with original pair_views at official128 grid. All foreground and minority statistics separate. Clean-holes replay uses full embed rect and foreground>50% proxy.",
             "width": "2*Euclidean distance at 3x3 local maxima (ridge thickness proxy, original512 pixel units); not a semantic skeleton.",
             "boundary": "Fraction of internal mask boundary within2px of top20% Sobel magnitude. Image-edge proxy, no semantic object annotations; replay uses16 separate existing COCO covers, not original sources.",
             "generator_sha256": sha(ROOT/"experiments/native_localizer/generator.py"), "code_sha256": sha(__file__)})


if __name__ == "__main__":
    main()
