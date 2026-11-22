"""The challenge metric, and format checks for answer files.

score = 0.1 * MCC_family + 0.9 * MCC_model, each MCC floored at 0.
  MCC_family: "same family?" over all endpoint pairs.
  MCC_model:  "same model?" over the pairs that truly share a family.
"""
import csv
import itertools
import math
from pathlib import Path

from . import config


def read_labels(path: Path, prefix: str) -> dict[str, tuple[str, str, str]]:
    """{endpoint: (family, model, kind)} for the lines whose endpoint starts with `prefix`. `kind` is the optional
    fourth column of practice_answers.csv."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]
    if rows and rows[0][:3] == ["endpoint", "family", "model"]:
        rows = rows[1:]
    return {r[0]: (r[1], r[2], r[3] if len(r) > 3 else "") for r in rows if len(r) >= 3 and r[0].startswith(prefix)}


def mcc(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Matthews correlation coefficient floored at 0; None when it is undefined (a zero row or column)."""
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else None


def score(truth: dict, mine: dict) -> dict:
    """Scores `mine` against `truth` (both {endpoint: (family, model, ...)}) the way the server does."""
    missing = sorted(set(truth) - set(mine))
    if missing:
        raise SystemExit(f"Your file has no line for: {', '.join(missing)}")
    fam, mod, wrong = [0, 0, 0, 0], [0, 0, 0, 0], []     # tp, fp, fn, tn

    def tally(box: list, real: bool, guess: bool) -> None:
        box[(0 if guess else 2) if real else (1 if guess else 3)] += 1

    for a, b in itertools.combinations(sorted(truth), 2):
        real_family = truth[a][0] == truth[b][0]
        tally(fam, real_family, mine[a][0] == mine[b][0])
        if real_family:
            real, guess = truth[a][:2] == truth[b][:2], mine[a][:2] == mine[b][:2]
            tally(mod, real, guess)
            if real != guess:
                wrong.append((a, truth[a][2], b, truth[b][2], guess))
    family_mcc, model_mcc = mcc(*fam), mcc(*mod)
    return {"family_mcc": family_mcc, "model_mcc": model_mcc,
            "score": 0.1 * (family_mcc or 0.0) + 0.9 * (model_mcc or 0.0),
            "family_counts": fam, "model_counts": mod, "wrong": wrong}


def validate(path: Path, endpoints: list[str] = config.SCORED) -> tuple[dict, list[str]]:
    """Checks a submission file against the server's format. Returns ({endpoint: (family, model)}, problems)."""
    problems, labels = [], {}
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]
    if rows and rows[0][:3] == ["endpoint", "family", "model"]:
        rows = rows[1:]
    for r in rows:
        if len(r) != 3:
            problems.append(f"line {r}: needs exactly 3 columns (labels must not contain commas)")
        elif not all(r):
            problems.append(f"line {r}: empty value")
        elif r[0] in labels:
            problems.append(f"{r[0]}: listed twice")
        else:
            labels[r[0]] = (r[1], r[2])
    missing, extra = sorted(set(endpoints) - set(labels)), sorted(set(labels) - set(endpoints))
    if missing:
        problems.append(f"missing endpoints: {', '.join(missing)}")
    if extra:
        problems.append(f"endpoints the server does not accept: {', '.join(extra)}")
    families_of = {}
    for family, model in labels.values():
        families_of.setdefault(model, set()).add(family)
    for model, families in sorted(families_of.items()):
        if len(families) > 1:
            problems.append(f"model label {model!r} appears in several families: {', '.join(sorted(families))}")
    return labels, problems


def describe(labels: dict) -> str:
    """One-paragraph summary of an answer: how many families and models, and how many pairs it calls the same."""
    families, models = {}, {}
    for ep, (family, model) in labels.items():
        families.setdefault(family, []).append(ep)
        models.setdefault((family, model), []).append(ep)
    pairs = lambda groups: sum(len(g) * (len(g) - 1) // 2 for g in groups.values())
    total = len(labels) * (len(labels) - 1) // 2
    return (f"{len(labels)} endpoints, {len(families)} families, {len(models)} models; "
            f"{pairs(families)}/{total} pairs called same family, {pairs(models)} called same model")
