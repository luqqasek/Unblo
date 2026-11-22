"""H3 gate: fuse DINO head maps with #3188 base maps on the 171 held-out images (fixed rule, no tuning on held-out)."""
import argparse, json
from pathlib import Path
import numpy as np
from h2_decode import expected_miou_decode, load, report

DEPLOYED_DECODE = {"pixelseal": True, "trustmark": True, "stegastamp": False}  # H2 decision


def fuse(prob, base_mask, maps, w=0.5, route_b_only=False):
    area = base_mask.reshape(len(base_mask), -1).mean(1)
    route_b = area >= 0.5
    q = np.where(route_b[:, None, None], 1 - maps[:, 0], maps[:, 1])
    p = (1 - w) * prob + w * q
    if route_b_only: p = np.where(route_b[:, None, None], p, prob)
    return p, route_b


def current(fam, prob, base):
    return expected_miou_decode(prob) if DEPLOYED_DECODE[fam] else base


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--maps", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    fams = load(a.bundle); d = np.load(a.maps); names = list(d["names"]); maps = d["maps"].astype(np.float32)
    res = {}
    for fam, f in fams.items():
        sel = [i for i, n in enumerate(f["names"]) if n in names]
        if fam != "pixelseal": sel = [i for i in sel if f["folds"][i] == 0]
        prob, base, gt = f["prob"][sel], f["base"][sel], f["gt"][sel]
        m = maps[[names.index(f["names"][i]) for i in sel]]
        cur = current(fam, prob, base); rows = []
        for tag, kw in [("blend_all", {}), ("blend_routeB", {"route_b_only": True})]:
            p, rb = fuse(prob, base, m, **kw)
            cand = expected_miou_decode(p) if DEPLOYED_DECODE[fam] else (p > 0.5).astype(np.uint8)
            r = report(tag, cur, cand, gt); r["routed_b"] = int(rb.sum())
            # repo minority guard (next_ideas 450 extension): no >0.01 drop in FG<=5% IoU or clean IoU on WM>50%
            from h2_decode import class_iou
            fg = gt.reshape(len(gt), -1).mean(1); small = fg <= 0.05; big = fg > 0.5
            g_small = float(class_iou(cand[small], gt[small], 1).mean() - class_iou(cur[small], gt[small], 1).mean()) if small.any() else 0.0
            g_big = float(class_iou(cand[big], gt[big], 0).mean() - class_iou(cur[big], gt[big], 0).mean()) if big.any() else 0.0
            r["guard"] = {"fg_le5": g_small, "clean_wm_over50": g_big}
            r["gate"] = bool(r["delta"] >= 0.005 and r["ci95"][0] > 0 and g_small >= -0.01 and g_big >= -0.01)
            rows.append(r)
            print(fam, tag, f"n={r['n']} routedB={r['routed_b']} cur={r['base']:.6f} cand={r['cand']:.6f} d={r['delta']:+.6f} ci={[round(x, 5) for x in r['ci95']]} +{r['improved']}/-{r['worse']} gate={r['gate']}")
            print("   ", {k: (v["n"], round(v["base"], 4), round(v["cand"], 4)) for k, v in r["buckets"].items()}, "guard", {k: round(v, 4) for k, v in r["guard"].items()})
        res[fam] = rows
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
