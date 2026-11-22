"""H2: per-image expected-mIoU decoding on saved #3188 maps (CPU only, no training).

Variants (predeclared in PLAN.md): D0 raw probabilities; D1 per-family temperature/bias, cross-fitted.
Gate per family: paired delta >= +0.003, bootstrap CI95 lower > 0, no minority bucket worse by > 0.01.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

EPS = 1e-6


def miou(pred, gt):
    p = pred.reshape(len(pred), -1).astype(bool); g = gt.reshape(len(gt), -1).astype(bool)
    out = []
    for c in (False, True):
        pc, gc = p == c, g == c
        inter = (pc & gc).sum(1); union = (pc | gc).sum(1)
        out.append(np.where(union == 0, 1.0, inter / np.maximum(union, 1)))
    return (out[0] + out[1]) / 2


def class_iou(pred, gt, cls):
    p = pred.reshape(len(pred), -1).astype(bool) == cls; g = gt.reshape(len(gt), -1).astype(bool) == cls
    inter = (p & g).sum(1); union = (p | g).sum(1)
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def expected_miou_decode(prob):
    """prob: [B,H,W] in [0,1]. Returns binary masks maximizing ratio-of-expectations mIoU, k in [1, N-1]."""
    b, h, w = prob.shape; n = h * w
    flat = prob.reshape(b, n).astype(np.float64)
    order = np.argsort(-flat, axis=1, kind="stable")
    s = np.cumsum(np.take_along_axis(flat, order, 1), axis=1)  # S(k) for k=1..n
    a1 = s[:, -1:]
    k = np.arange(1, n + 1)[None]
    iou1 = s / np.maximum(k + a1 - s, EPS)
    iou0 = (n - k - a1 + s) / np.maximum(n - s, EPS)
    obj = 0.5 * (iou1 + iou0)
    obj[:, -1] = -np.inf  # exclude k = n
    kstar = obj.argmax(1) + 1
    out = np.zeros((b, n), np.uint8)
    rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(n)[None].repeat(b, 0), 1)
    out[rank < kstar[:, None]] = 1
    return out.reshape(b, h, w)


def calibrate(prob, t, bias):
    z = np.log(np.clip(prob, EPS, 1 - EPS)) - np.log(np.clip(1 - prob, EPS, 1 - EPS))
    return 1 / (1 + np.exp(-(z - bias) / t))


T_GRID = (0.5, 1.0, 2.0, 4.0)
B_GRID = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def fit_tb(prob, gt):
    best = None
    for t in T_GRID:
        for bias in B_GRID:
            s = miou(expected_miou_decode(calibrate(prob, t, bias)), gt).mean()
            if best is None or s > best[0] + 1e-12: best = (s, t, bias)
    return best[1], best[2]


def buckets(gt):
    n = gt[0].size; fg = gt.reshape(len(gt), -1).mean(1)
    return {"fg_le1": (fg <= .01, 1), "fg_1to5": ((fg > .01) & (fg <= .05), 1),
            "clean_le1": (1 - fg <= .01, 0), "clean_1to5": ((1 - fg > .01) & (1 - fg <= .05), 0),
            "wm_over50": (fg > .5, 0)}


def boot(delta, reps=10000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(delta), (reps, len(delta)))
    m = delta[idx].mean(1); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def report(name, base, cand, gt):
    sb, sc = miou(base, gt), miou(cand, gt); d = sc - sb
    row = {"variant": name, "n": len(gt), "base": float(sb.mean()), "cand": float(sc.mean()), "delta": float(d.mean()),
           "ci95": boot(d), "improved": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()), "buckets": {}}
    worst = 0.0
    for key, (sel, cls) in buckets(gt).items():
        if sel.sum() == 0: continue
        b0 = float(class_iou(base[sel], gt[sel], cls).mean()); c0 = float(class_iou(cand[sel], gt[sel], cls).mean())
        row["buckets"][key] = {"n": int(sel.sum()), "base": b0, "cand": c0}
        worst = min(worst, c0 - b0)
    row["gate"] = bool(row["delta"] >= 0.003 and row["ci95"][0] > 0 and worst >= -0.01)
    return row


def hsplit(names, k=3):
    return np.array([int(hashlib.sha256(n.encode()).hexdigest(), 16) % k for n in names])


def load(bundle):
    m = Path(bundle) / "maps"; c = Path(bundle) / "cluster"
    fams = {}
    d = np.load(m / "pixelseal-validation57.npz")
    fams["pixelseal"] = dict(names=d["names"], prob=d["probabilities"], base=d["masks"], gt=d["targets"],
                             folds=hsplit(list(d["names"])))
    for fam, sp in [("stegastamp", "next-ideas-extended-runs/stegastamp/fold%d/split.json"),
                    ("trustmark", "region-runs/trustmark/objects-adapt/fold%d/split.json")]:
        d = np.load(m / f"{fam}-oof.npz"); names = list(d["names"])
        fold = np.full(len(names), -1)
        for k in (1, 2, 3):
            for nm in json.load(open(c / (sp % k)))["validation"]: fold[names.index(nm)] = k - 1
        assert (fold >= 0).all()
        fams[fam] = dict(names=d["names"], prob=d["probabilities"], base=d["masks"], gt=d["targets"], folds=fold)
    return fams


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); fams = load(a.bundle); res = {}
    for fam, f in fams.items():
        prob, base, gt, folds = f["prob"], f["base"], f["gt"], f["folds"]
        rows = [report("D0", base, expected_miou_decode(prob), gt)]
        cand = np.zeros_like(base); params = []
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            t, bias = fit_tb(prob[tr], gt[tr]); params.append((float(t), float(bias)))
            cand[te] = expected_miou_decode(calibrate(prob[te], t, bias))
        r = report("D1", base, cand, gt); r["params"] = params; rows.append(r)
        res[fam] = rows
        for r in rows:
            print(fam, r["variant"], f"base={r['base']:.6f} cand={r['cand']:.6f} d={r['delta']:+.6f} ci={r['ci95']}",
                  f"+{r['improved']}/-{r['worse']} gate={r['gate']}", r.get("params", ""), flush=True)
            print("   ", {k: (v["n"], round(v["base"], 4), round(v["cand"], 4)) for k, v in r["buckets"].items()})
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
