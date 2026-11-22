"""Hillclimbing harness: score parameter and rule variants on a dataset with known answers, fast.

Evidence (the pairwise tests) is measured once per dataset; every variant that only changes thresholds or rules is
then scored from that cache in well under a second.

  uv run python validation/hillclimb.py practice testbed
"""
import json
import sys
import time
from pathlib import Path

from meq import config, decide, features, findings, scoring

# Scored on frozen copies (data/snapshots/), so two variants are always compared on the same answers even while
# collection continues. `python validation/hillclimb.py snapshot` refreshes the copies.
LIVE = {"practice": config.REFERENCE_ANSWERS,
        "testbed": config.ROOT / "testbed" / "out" / "answers-sampled.jsonl"}
SNAP = config.DATA / "snapshots"
SETS = {
    "practice": (SNAP / "practice.jsonl", config.PRACTICE_TRUTH, "ref-"),
    "testbed": (SNAP / "testbed.jsonl", config.ROOT / "testbed" / "out" / "truth.csv", "tb-"),
}
LOG = config.ROOT / "research" / "findings" / "hillclimb.jsonl"
_cache = {}


def evidence(name: str, packs_in_tests: bool, permutations: int = 300, model_prompts: str = "all", style_normalization: str = "global"):
    if name not in SETS:
        raise ValueError(f"Inactive dataset {name!r}. Active datasets: practice, testbed. OpenAI is preserved in archive/openai-test-set/.")
    key = (name, packs_in_tests, permutations, model_prompts, style_normalization)
    if key not in _cache:
        path, truth_path, prefix = SETS[name]
        truth = scoring.read_labels(truth_path, prefix)
        data = features.load(path, prefix)
        data = {e: data[e] for e in data if e in truth}
        _cache[key] = (decide.Evidence(data, permutations=permutations, packs_in_tests=packs_in_tests, model_prompts=model_prompts, style_normalization=style_normalization), truth)
    return _cache[key]


def score(name: str, overrides: dict) -> dict:
    params = {**decide.PARAMS, **overrides}
    ev, truth = evidence(name, params.get("packs_in_tests", False), model_prompts=params.get("model_prompts", "all"), style_normalization=params.get("style_normalization", "global"))
    full = decide.evaluate(truth, decide.labels_from(ev, params))
    known = decide.evaluate(truth, decide.labels_from(ev, params, known_families={e: truth[e][0] for e in ev.names}))
    # "All pairs": is the split into models right, ignoring families altogether? The whole set is treated as one family.
    one = {e: "all" for e in ev.names}
    flat = decide.evaluate({e: ("all", truth[e][1], truth[e][2]) for e in ev.names}, decide.labels_from(ev, params, known_families=one))
    return {"score": full["score"], "family": full["family"], "model": full["model"], "model_known_families": known["model"],
            "all_pairs": flat["model"]}


def record(round_name: str, change: str, scores: dict, kept: bool, settings: dict | None = None,
           snapshot_hashes: dict | None = None) -> None:
    """One line per hillclimbing round, and the card the control center shows."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"round": round_name, "time": time.strftime("%H:%M"), "change": change, "scores": scores, "kept": kept,
                                 "settings": settings, "snapshot_hashes": snapshot_hashes,
                                 "weights": WEIGHTS,
                                 "objective": sum(WEIGHTS[n] * scores[n]["score"] for n in WEIGHTS) if all(n in scores for n in WEIGHTS) else None}) + "\n")
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]

    def cell(r, name):
        s = r["scores"].get(name)
        return "-" if not s else f"{s['score']:.2f}"

    table = [[r["round"], r["time"], r["change"], cell(r, "practice"), cell(r, "testbed"),
              "62.5/37.5" if r.get("weights") == WEIGHTS else "historical (includes OpenAI)",
              f"{r['objective']:.3f}" if r.get("objective") is not None else "-", "kept" if r["kept"] else "dropped"] for r in rows]
    latest = next((r for r in reversed(rows) if r["kept"] and r.get("objective") is not None), None)
    findings.write(
        "hillclimb", "Hillclimbing log", "3",
        tiles=[{"label": "Rounds", "value": str(len(rows)), "hint": f"{sum(r['kept'] for r in rows)} kept"},
               {"label": "Latest accepted weighted score", "value": f"{latest['objective']:.3f}" if latest else "-", "hint": latest["round"] if latest else ""}],
        table={"columns": ["Round", "Time", "Change", "Practice", "Test bed", "Weights (test bed/practice)", "Weighted", "Decision"], "rows": table},
        text=["Active objective: 62.5% test bed, 37.5% practice. OpenAI is archived and excluded from future rounds. Historical weighted scores retain their original weights and are not directly comparable. Candidates use matching frozen answers. A round may improve the full-pool score or replay stability; its description records the acceptance checks. "
              "These are development scores, not independent estimates of competition performance. Older rounds used different pools or a nondeterministic permutation test."],
    )


# OpenAI retired at the user's request. Preserve the existing 50:30 ratio
# between the remaining datasets, normalized to sum to one.
WEIGHTS = {"testbed": 0.625, "practice": 0.375}


def quick(name: str, overrides: dict) -> float:
    """Just the score of one dataset (no diagnostics): what the climb needs."""
    params = {**decide.PARAMS, **overrides}
    ev, truth = evidence(name, params.get("packs_in_tests", False), model_prompts=params.get("model_prompts", "all"), style_normalization=params.get("style_normalization", "global"))
    return decide.evaluate(truth, decide.labels_from(ev, params))["score"]


def objective(overrides: dict, full: bool = True) -> tuple[float, dict]:
    if not full:
        scores = {name: {"score": quick(name, overrides)} for name in WEIGHTS}
    else:
        scores = {name: score(name, overrides) for name in WEIGHTS}
    return sum(WEIGHTS[name] * scores[name]["score"] for name in WEIGHTS), scores


GRID = {
    "audit_min_self": [0.0, 0.5, 0.65, 0.8],
    "seed_strategy": ["outlier", "central", "neighbours"],
    "model_prompts": ["all", "micro", "open"], "attach_uncertainty": [False, True],
    "attach": ["decisive", "agreement", "base"], "attach_margin": [0.01, 0.03, 0.06, 0.1],
    "same_z": [2.5, 3.0, 3.5, 4.0, 5.0, 6.0], "packs_in_tests": [True, False],
    "watermark_self": [1.2, 1.28, 1.4, 1.6, 9.0], "watermark_cap": [False, True], "hotter_self": [0.0, 0.65, 0.75, 0.85],
    "mild_shift": [6.0, 8.0, 12.0, 20.0], "keep_cosine": [0.97, 0.985, 0.995, 9.0], "keep_flip": [0.005, 0.01, 0.02],
    "aside_first": [True, False], "audit_trust": [0.0, 0.2, 0.5], "prompted_names": [0.0, 0.25, 0.34],
    "family_threshold": [0.2, 0.3, 0.4, 0.5, 0.6], "usual_share": [2, 3, 4],
}


def climb(start: dict | None = None, log=print) -> tuple[dict, float]:
    """Coordinate ascent: change one setting at a time, keep a change only if the objective rises, stop when a full
    pass over every setting changes nothing. The result is a local maximum of the objective on the snapshots."""
    current = dict(start or {})
    best, scores = objective(current, full=False)
    practice_floor = scores["practice"]["score"]
    log(f"start {best:.4f}  " + "  ".join(f"{n} {scores[n]['score']:.2f}" for n in WEIGHTS))
    improved, passes = True, 0
    while improved:
        improved, passes = False, passes + 1
        for name, values in GRID.items():
            for value in values:
                if {**decide.PARAMS, **current}.get(name) == value:
                    continue
                j, s = objective({**current, name: value}, full=False)
                if j > best + 1e-9 and s["practice"]["score"] >= practice_floor - 1e-9:
                    best, current, improved = j, {**current, name: value}, True
                    practice_floor = s["practice"]["score"]
                    log(f"pass {passes}: {name} = {value!r} -> {j:.4f}  " + "  ".join(f"{n} {s[n]['score']:.2f}" for n in WEIGHTS))
    log(f"local maximum after {passes} passes: {best:.4f} with {current}")
    return current, best


VARIANTS = {"current": {}}

if __name__ == "__main__" and sys.argv[1:2] == ["snapshot"]:
    import shutil
    SNAP.mkdir(parents=True, exist_ok=True)
    for name, path in LIVE.items():
        shutil.copyfile(path, SNAP / f"{name}.jsonl")
        print(f"{name}: {sum(1 for _ in open(path, encoding='utf-8'))} answers frozen")
elif __name__ == "__main__" and sys.argv[1:2] == ["climb"]:
    found, value = climb()
    print(json.dumps({"settings": found, "objective": value}))
elif __name__ == "__main__":
    names = sys.argv[1:] or list(WEIGHTS)
    inactive = set(names) - SETS.keys()
    if inactive:
        sys.exit("Inactive datasets: " + ", ".join(sorted(inactive)) + ". OpenAI is archived in archive/openai-test-set/.")
    print(f"{'variant':34s} " + " ".join(f"{n + ': score fam model (known fam) [all pairs]':>52s}" for n in names))
    for variant, overrides in VARIANTS.items():
        cells = []
        for n in names:
            s = score(n, overrides)
            fam = "one" if s["family"] is None else f"{s['family']:.2f}"
            cells.append(f"{s['score']:.2f} {fam:>5s} {s['model']:.2f} ({s['model_known_families']:.2f}) [{s['all_pairs']:.2f}]".rjust(52))
        print(f"{variant:34s} " + " ".join(cells), flush=True)
