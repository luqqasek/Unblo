"""Reuse the exact adapted OOF; oracle thresholds are diagnostics, never validation claims."""
import argparse
import time

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_auc_score

from .common import *


def exact_oracle(probs, targets, metric="fg_iou"):
    """Maximize macro IoU over every distinct shared threshold, handling ties together."""
    scores, changes, initial = [], [], 0.
    for p, y in zip(probs, targets):
        p, y = p.ravel(), y.ravel().astype(bool)
        order = np.argsort(-p, kind="stable")
        s = p[order]
        ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
        count = ends + 1
        tp = np.cumsum(y[order])[ends]
        pos, total = int(y.sum()), len(y)
        fg = np.divide(tp, pos + count - tp, out=np.ones(len(tp)), where=(pos + count - tp) != 0)
        bg = (total-pos-count+tp) / np.maximum(total-tp, 1)
        bg[(total-tp) == 0] = 1
        values = fg if metric == "fg_iou" else (fg+bg)/2
        start = ious(np.zeros_like(y), y)[metric]
        initial += start
        scores.append(s[ends]); changes.append(np.diff(np.r_[start, values]))
    s, delta = np.concatenate(scores), np.concatenate(changes)
    order = np.argsort(-s, kind="stable")
    s, delta = s[order], delta[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    values = (initial + np.cumsum(delta)[ends]) / len(probs)
    best = int(np.argmax(values))
    if initial/len(probs) > values[best]:
        return {"threshold": float(np.nextafter(s[0], np.array(np.inf, dtype=s.dtype))), "objective": float(initial/len(probs))}
    return {"threshold": float(s[ends[best]]), "objective": float(values[best])}


def hysteresis(prob, low, high):
    labels, _ = ndi.label(prob >= low, structure=np.ones((3, 3)))
    keep = np.unique(labels[prob >= high]); keep = keep[keep != 0]
    return np.isin(labels, keep)


def run(runs, out, require_maps=False):
    start = time.monotonic(); meta, gt = metadata(), masks128()
    summaries, allrows, hashes = {}, [], {}
    for family in FAMILIES:
        rows, probs, targets = [], {}, {}
        for fold in (1, 2, 3):
            base = runs / family / f"matched-v1-adapt/fold{fold}"
            split = read(base / "split.json")
            saved = read(base / "per_image.json")
            held = set(split.get("validation", split.get("val", [])))
            assert held and not set(split["train"]) & held
            assert {r["image_name"] for r in saved} == held
            for r in saved:
                name = r["image_name"]
                assert meta[name]["method"] == family
                assert abs(r["true_area"] - gt[name].mean()) < 1e-8
                rows.append(dict(r, family=family, fold=fold, bucket=bucket(r["true_area"]), full_watermark=meta[name]["full_watermark"]))
            for file in (base / "split.json", base / "per_image.json"):
                hashes[str(file)] = sha(file)
            path = base / "maps.npz"
            if path.exists():
                hashes[str(path)] = sha(path)
                with np.load(path) as z:
                    assert set(z["names"].tolist()) == held
                    assert len(z["names"]) == len(held) == len(z["probabilities"]) == len(z["targets"])
                    for name, p, y in zip(z["names"], z["probabilities"], z["targets"]):
                        name = str(name); p = p.squeeze(); y = y.squeeze().astype(bool)
                        assert p.shape == y.shape == (128, 128) and np.isfinite(p).all()
                        assert p.min() >= 0 and p.max() <= 1 and np.array_equal(y, gt[name])
                        probs[name], targets[name] = p, y
        assert len(rows) == len({r["image_name"] for r in rows}) == 170
        expected = .571916 if family == "pixelseal" else .578721
        assert abs(mean(rows, "miou") - expected) < 1e-6
        # Reproduce the reported failure, independently of new diagnostic hypotheses.
        assert mean(select(rows, "le1"), "fg_iou") == 0
        complete = len(probs) == 170
        if require_maps and not complete:
            raise FileNotFoundError(f"{family}: only {len(probs)}/170 probability maps")
        for r in rows:
            r.update(max_prob_gt=None, max_prob_bg=None, pr_auc_trapezoid=None, per_image_oracle_fg=None)
            if r["image_name"] not in probs:
                continue
            p, y = probs[r["image_name"]], targets[r["image_name"]]
            for k, value in ious(p >= .5, y).items():
                assert abs(value-r[k]) < 1e-7, (family, r["image_name"], k)
            assert abs(float(not (p >= .5).any())-r["empty"]) < 1e-8
            if y.any() and not y.all():
                assert abs(roc_auc_score(y.ravel(), p.ravel())-r["roc_auc"]) < 1e-6
                assert abs(average_precision_score(y.ravel(), p.ravel())-r["fg_ap"]) < 1e-6
                precision, recall, _ = precision_recall_curve(y.ravel(), p.ravel())
                r["pr_auc_trapezoid"] = float(auc(recall, precision))
            r["max_prob_gt"] = float(p[y].max()) if y.any() else None
            r["max_prob_bg"] = float(p[~y].max()) if (~y).any() else None
            r["per_image_oracle_fg"] = exact_oracle([p], [y])["objective"]
        # Evaluate a fixed candidate grid once; reuse the exact image-level scores
        # for each cohort and for a single global policy optimized for official mIoU.
        hyst_params, hyst_scores = [], []
        if complete:
            for low in (.001, .005, .01, .02, .05, .1, .2, .3, .4, .5):
                for high in (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9):
                    if high < low:
                        continue
                    hyst_params.append({"low": low, "high": high})
                    hyst_scores.append([ious(hysteresis(probs[r["image_name"]], low, high), targets[r["image_name"]]) for r in rows])
            print(family, "validated170 maps; hysteresis candidates", len(hyst_params), flush=True)
        groups = {}
        for label in ("all",) + BUCKETS:
            rr = select(rows, label)
            g = {"n": len(rr), **{k: mean(rr, k) for k in ("miou", "fg_iou", "clean_iou", "roc_auc", "fg_ap", "empty", "true_area", "max_prob_gt", "max_prob_bg", "pr_auc_trapezoid", "per_image_oracle_fg")}}
            if complete and rr:
                ps = [probs[r["image_name"]] for r in rr]; ys = [targets[r["image_name"]] for r in rr]
                oracle = exact_oracle(ps, ys)
                oracle.update({k: float(np.mean([ious(p >= oracle["threshold"], y)[k] for p, y in zip(ps, ys)])) for k in ("miou", "fg_iou", "clean_iou")})
                chosen = {r["image_name"] for r in rr}
                indices = [i for i, r in enumerate(rows) if r["image_name"] in chosen]
                candidates = [{**params, **{k: mean([values[i] for i in indices], k) for k in ("miou", "fg_iou", "clean_iou")}}
                              for params, values in zip(hyst_params, hyst_scores)]
                best = max(candidates, key=lambda v: v["fg_iou"])
                g.update(oracle_shared_threshold=oracle, oracle_hysteresis_grid=best)
            groups[label] = g
        policies = {}
        if complete:
            ps = [probs[r["image_name"]] for r in rows]; ys = [targets[r["image_name"]] for r in rows]
            global_t = exact_oracle(ps, ys, "miou")
            best_h = int(np.argmax([mean(v, "miou") for v in hyst_scores]))
            for label, params, values in (
                ("baseline_t05", {"threshold": .5}, [ious(p >= .5, y) for p, y in zip(ps, ys)]),
                ("global_threshold_oracle_miou", global_t, [ious(p >= global_t["threshold"], y) for p, y in zip(ps, ys)]),
                ("global_hysteresis_oracle_miou", hyst_params[best_h], hyst_scores[best_h]),
            ):
                joined = [{**r, **v} for r, v in zip(rows, values)]
                policies[label] = {"parameters": params, "groups": {b: {"n": len(select(joined, b)), **{k: mean(select(joined, b), k) for k in ("miou", "fg_iou", "clean_iou")}} for b in ("all",)+BUCKETS}}
        summaries[family] = {"n_maps": len(probs), "groups": groups, "policies": policies}
        allrows.extend(rows)
        print(family, json.dumps(groups), flush=True)
    save(out / "oof.json", {"families": summaries, "source_hashes": hashes, "seconds": time.monotonic()-start,
         "protocol": "Unchanged seed42 3-fold adapted OOF, 170/family; no fitting. Oracle uses evaluation GT, not deployable or independent validation.",
         "clean_holes": "full_watermark metadata == true and 0 < official foreground area < 1; overlapping cohort",
         "pr_auc": "fg_ap is non-interpolated average precision, macro per image; trapezoidal PR area additionally computed when maps exist."})
    csv_rows(out / "oof_per_image.csv", allrows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--runs", type=Path, default=RUNS); p.add_argument("--out", type=Path, default=OUT); p.add_argument("--require-maps", action="store_true")
    a = p.parse_args(); run(a.runs, a.out, a.require_maps)
