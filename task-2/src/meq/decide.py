"""The pipeline: an answers file in, a family label and a model label per endpoint out.

Two endpoints are put together when their answers to the packed items agree about as often as each agrees with
itself (features.pair, "cosine"). Families use a loose threshold, models a tight one, both with average-linkage
clustering. The thresholds are set on datasets with known answers and held fixed everywhere else.

  uv run python -m meq.decide explore FILE --truth TRUTH --prefix ref-     pairwise table and how well it separates
  uv run python -m meq.decide score FILE --truth TRUTH --prefix ref-       labels, score, replays at a 35-query budget
"""
import argparse
from collections import Counter
import csv
import itertools
import random
import statistics
import zlib
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import config, features, findings, moments, probes, scoring

PARAMS = {
    "family_threshold": 0.40,     # R15: validated with exact null moments and central/complete clustering
    "family_metric": "cosine",   # cosine, matched_cosine or hist_cosine
    "style_normalization": "global",  # global or pair: reference used for style scaling
    "same_z": 6.0,
    "null_method": "exact",      # exact permutation moments; montecarlo retained for historical comparisons
    "seed_strategy": "central", # outlier, central, or neighbours; compared on frozen validation data
    "growth": "complete",        # pooled reference or complete pairwise compatibility
    "packs_in_tests": False,       # packed questions also feed the model-level tests (False: families only)
    "model_prompts": "micro",     # R7: short tasks; all/open retained as experiment options
    "keep_cosine": 0.985,         # an outlier whose item answers agree this well with the base, and
    "keep_flip": 0.01,            # contradicts at most this share of the base's unanimous answers, is the base in disguise
    "prompted_names": 0.25,       # names the group's maker in at most this share of "who made you" replies: hidden prompt
    "prompted_admits": 0.67,      # answers YES to the instruction audit in at least this share of replies: hidden prompt
    "audit_trust": 0.5,           # ...but only where the rest of the group says YES at most this often (small models answer at random)
    "audit_min_self": 0.5,        # R8: trust audits only with >=50% median packed-answer repeatability
    "usual_share": 4,             # a maker name counts as usual when at least this many endpoints keep using it
    "aside_first": True,          # set hidden-prompt endpoints aside before the family split
    "attach": "decisive",         # where a hidden-prompt endpoint goes: "agreement" (best-matching group), "base" (its family's
    "attach_margin": 0.15,        #   first group), "decisive" (best-matching group only if it beats the first group by this margin)
    "attach_uncertainty": False,  # scale the margin by the endpoint's observed answer variability
    "watermark_cap": False,       # apply the mild-shift cap to the self-consistency rule too
    "watermark_self": 1.2,        # an outlier this many times more self-consistent than its group is watermarked or sampled colder
    "hotter_self": 0.0,           # R9: shortcut threshold reduced to zero self-consistency
    "mild_shift": 8.0,            # "mild" for the two rules above: a strong variant moves far more than a decoding change does
    "groups": ("knowledge", "choice", "skill"),
}


class Evidence:
    """Everything measured once per answers file: profiles and pairwise statistics, computed on demand and kept."""

    def __init__(self, data: dict, repeats: dict | None = None, permutations: int = 300, seed: int = 0, packs_in_tests: bool = False,
                 model_prompts: str = "all", style_normalization: str = "global", null_method: str | None = None):
        if style_normalization not in {"global", "pair"}:
            raise ValueError(f"Unknown style normalization: {style_normalization}")
        self.style_normalization = style_normalization
        self.use_null_method(PARAMS.get("null_method", "montecarlo") if null_method is None else null_method)
        self.data, self.names = data, sorted(data)
        self.samples = {e: features.item_samples(data[e], None if repeats is None else repeats[e]) for e in self.names}
        # Families always use packed answers. Model-test scope is selected separately;
        # R7 uses short tasks after the packed-test winner regressed on updated data.
        unpacked = {e: {pid: v for pid, v in data[e].items() if packs_in_tests or pid not in probes.PACKS} for e in self.names}
        if model_prompts not in {"all", "micro", "open"}:
            raise ValueError(f"Unknown model prompt scope: {model_prompts}")
        if model_prompts != "all":
            include = lambda p: (len(p) > 2 and p[0] == "m" and p[1].isdigit()) if model_prompts == "micro" else p.startswith("open-")
            unpacked = {e: {pid: v for pid, v in answers.items() if include(pid)} for e, answers in unpacked.items()}
        self.reps = {e: features.represented(unpacked[e], None if repeats is None else repeats[e]) for e in self.names}
        self.permutations, self.seed = permutations, seed
        self.styles = {e: features.style_sets(unpacked[e], None if repeats is None else repeats[e]) for e in self.names}
        self.scale = features.style_scale(list(self.styles.values()))
        self.profiles = {e: features.profile(data[e]) for e in self.names}
        self._pair, self._z, self._flip, self._style = {}, {}, {}, {}

    def use_null_method(self, method: str) -> None:
        if method not in {"exact", "montecarlo"}:
            raise ValueError(f"Unknown null method: {method}")
        self.null_method = method

    def _test_key(self, key):
        # Preserve original Monte Carlo cache keys and RNG seeds. Exact results
        # occupy a distinct namespace, including in old pickled Evidence objects.
        return ("exact-null", key) if getattr(self, "null_method", "montecarlo") == "exact" else key

    def _mmd(self, *args):
        function = moments.mmd if getattr(self, "null_method", "montecarlo") == "exact" else features.mmd
        return function(*args)

    def _style_test(self, *args):
        function = moments.style_test if getattr(self, "null_method", "montecarlo") == "exact" else features.style_test
        return function(*args)

    def style(self, a: str, b: str) -> float:
        key = (a, b) if a < b else (b, a)
        cached_key = self._test_key(key)
        if cached_key not in self._style:
            self._style[cached_key] = self._style_test(self.styles[key[0]], self.styles[key[1]], self.style_scale(self.styles[key[0]], self.styles[key[1]]), self.permutations, self._seed(*key))["z"]
        return self._style[cached_key]

    def style_scale(self, a: dict, b: dict) -> tuple:
        if getattr(self, "style_normalization", "global") == "pair":
            shared = a.keys() & b.keys()
            return features.style_scale([{p: a[p] for p in sorted(shared)}, {p: b[p] for p in sorted(shared)}])
        return self.scale

    def _seed(self, *parts) -> int:
        """The same pair always gets the same random relabellings, whatever was computed before it."""
        return zlib.crc32(repr(parts).encode()) + self.seed

    def pair(self, a: str, b: str) -> dict:
        key = (a, b) if a < b else (b, a)
        if key not in self._pair:
            self._pair[key] = features.pair(self.samples[key[0]], self.samples[key[1]])
        return self._pair[key]

    def z(self, a: str, b: str) -> float:
        key = (a, b) if a < b else (b, a)
        cached_key = self._test_key(key)
        if cached_key not in self._z:
            self._z[cached_key] = self._mmd(self.reps[key[0]], self.reps[key[1]], self.permutations, self._seed(*key))["z"]
        return self._z[cached_key]

    def on(self, a: str, b: str, kind: str) -> dict:
        """Both tests restricted to one kind of prompt: "m" the crisp short tasks, "o" the same tasks after
        "ignore any earlier instructions"."""
        key = (kind, a, b) if a < b else (kind, b, a)
        cached_key = self._test_key(key)
        if cached_key not in self._style:
            pick = lambda sets: {p: v for p, v in sets.items() if len(p) > 2 and p[0] == kind and p[1].isdigit()}
            sa, sb = pick(self.styles[key[1]]), pick(self.styles[key[2]])
            style = self._style_test(sa, sb, self.style_scale(sa, sb), self.permutations, self._seed(*key))
            text = self._mmd(pick(self.reps[key[1]]), pick(self.reps[key[2]]), self.permutations, self._seed(*key))
            self._style[cached_key] = {"style": style["z"], "shift": style["shift"], "z": text["z"], "stat": text["stat"]}
        return self._style[cached_key]

    def against(self, e: str, pool: list) -> float:
        """How far one endpoint is from a group of endpoints taken together (their replies pooled per prompt).
        Pooling gives the reference more replies, so a real shift shows sooner than against any single member."""
        key = ("pool", e, tuple(sorted(pool)))
        cached_key = self._test_key(key)
        if cached_key not in self._style:
            styles, reps = {}, {}
            for member in sorted(pool):
                for pid, vectors in self.styles[member].items():
                    styles.setdefault(pid, []).extend(vectors)
                for pid, items in self.reps[member].items():
                    reps.setdefault(pid, []).extend(items)
            style = self._style_test(self.styles[e], styles, self.style_scale(self.styles[e], styles), self.permutations, self._seed(*key[1:]))["z"]
            text = self._mmd(self.reps[e], reps, self.permutations, self._seed(*key[1:]))["z"]
            self._style[cached_key] = max(0.0, style, text)
        return self._style[cached_key]

    def flip(self, a: str, b: str) -> dict:
        key = (a, b) if a < b else (b, a)
        if key not in self._flip:
            most = max((len(v) for v in self.samples[key[0]].values()), default=0)
            self._flip[key] = features.flips(self.samples[key[0]], self.samples[key[1]], need=3 if most >= 3 else 2)
        return self._flip[key]


def groups_from(distance: np.ndarray, threshold: float) -> list:
    """Average-linkage clusters: two groups merge while their mean distance stays below the threshold."""
    if len(distance) == 1:
        return [1]
    distance = np.array(distance, dtype=float)
    np.fill_diagonal(distance, 0.0)
    return list(fcluster(linkage(squareform(distance, checks=False), method="average"), t=threshold, criterion="distance"))


def families(ev: Evidence, params: dict, members: list | None = None) -> list:
    """Family number per endpoint (of `members`, default all). Endpoints whose packs cannot be read are placed
    afterwards, with the endpoint the distribution test finds closest."""
    if members is not None and len(members) != len(ev.names):
        sub = Evidence.__new__(Evidence)
        sub.__dict__.update(ev.__dict__)
        sub.names = list(members)
        return families(sub, params)
    readable = [k for k, e in enumerate(ev.names) if len(ev.samples[e]) >= 40]
    if len(readable) < 2:
        return [1] * len(ev.names)
    n = len(readable)
    distance = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        distance[i, j] = distance[j, i] = max(0.0, 1.0 - ev.pair(ev.names[readable[i]], ev.names[readable[j]])[params.get("family_metric", "cosine")])
    found = dict(zip(readable, groups_from(distance, params["family_threshold"])))
    for k, e in enumerate(ev.names):
        if k not in found:
            found[k] = found[min(readable, key=lambda r: ev.z(e, ev.names[r]))]
    # Two endpoints that keep naming different makers are not one family, however alike their answers look.
    # "Same maker" means their sets of usual names overlap (Hugging and Face count as one maker, not two).
    named = {}
    for k, e in enumerate(ev.names):
        replies = features.who_replies(ev.data[e])
        counts = Counter(name for text in replies for name in features.names_used(text))
        named[k] = {name for name, c in counts.items() if c >= max(2, 0.5 * len(replies))}
    final, numbers = {}, {}
    for family in sorted(set(found.values())):
        inside = [k for k in found if found[k] == family]
        makers = []                                   # groups of names that occur together
        for k in inside:
            if not named[k]:
                continue
            touching = [m for m in makers if m & named[k]]
            merged = set(named[k]).union(*touching) if touching else set(named[k])
            makers = [m for m in makers if m not in touching] + [merged]
        makers = [m for m in makers if sum(bool(named[k] & m) for k in inside) >= 2]     # a maker only one endpoint names is noise
        if len(makers) < 2:
            for k in inside:
                final[k] = (family, 0)
            continue
        for k in inside:
            hit = [i for i, m in enumerate(makers) if named[k] & m]
            if hit:
                final[k] = (family, hit[0])
        for k in inside:
            if k not in final:        # no usual maker named: go with the named endpoint whose answers are closest
                best = max((j for j in inside if j in final), key=lambda j: ev.pair(ev.names[k], ev.names[j])["cosine"])
                final[k] = final[best]
    for k in range(len(ev.names)):
        numbers.setdefault(final[k], len(numbers) + 1)
    return [numbers[final[k]] for k in range(len(ev.names))]


def attach(ev: Evidence, e: str, groups: dict, params: dict) -> tuple:
    """Which group a hidden-prompt endpoint joins. `groups` is {(family, model): [endpoints]}.

    Its packed answers are compared with every group's. On steady models the best match is its true base. On noisy
    ones the match is a coin toss between a base and its variants, so the default is the family's first group (the
    base and its disguised copies) unless another group matches decisively better."""
    agreement = {g: float(np.mean([ev.pair(e, b)["cosine"] for b in members])) for g, members in groups.items()}
    items = {g: float(np.mean([ev.pair(e, b)["items"] for b in members])) for g, members in groups.items()}
    if max(items.values()) < 20:                      # its packs could not be read: fall back on the distribution test
        agreement = {g: -ev.against(e, members) for g, members in groups.items()}
        margin = float("inf")
    else:
        margin = params["attach_margin"]
        if params.get("attach_uncertainty", False):
            margin *= max(0.01, 1.0 - ev.profiles[e]["self"]) ** 0.5
    best = max(agreement, key=agreement.get)
    first = min((g for g in groups if g[0] == best[0]), key=lambda g: g[1])
    if params["attach"] == "agreement" or best == first:
        return best
    if params["attach"] == "base":
        return first
    return best if agreement[best] - agreement[first] >= margin else first


def prompted(ev: Evidence, members: list, params: dict) -> dict:
    """{endpoint: reason} for the endpoints that show a hidden prompt or persona.

    Two signs, each judged against the rest of the group: the endpoint admits to instructions when asked directly
    (and most of the group does not), or it no longer names the maker that most of the group names."""
    out = {}
    admits = {e: features.admits_rate(ev.data[e]) for e in members}
    asked = [v for v in admits.values() if v is not None]
    reliable = float(np.median([ev.profiles[e]["self"] for e in members])) >= params.get("audit_min_self", 0.0)
    if asked and reliable and float(np.median(asked)) <= params["audit_trust"]:
        for e in members:
            if admits[e] is not None and admits[e] >= params["prompted_admits"]:
                out[e] = f"hidden prompt: admits to instructions in {admits[e]:.0%} of audit replies"
    usual = features.usual_names(ev.data, members)
    if usual:
        rates = {e: features.name_rate(ev.data[e], usual) for e in members}
        if float(np.median(list(rates.values()))) >= 0.5:
            for e in members:
                if e not in out and rates[e] <= params["prompted_names"] and features.who_replies(ev.data[e]):
                    out[e] = f"hidden prompt or persona: names {'/'.join(sorted(usual))} in {rates[e]:.0%} of replies"
    return out


def base_groups(ev: Evidence, members: list, params: dict, first: int = 1) -> tuple[dict, dict]:
    """Groups endpoints without a hidden prompt into models. ({endpoint: model number}, {endpoint: why}).

    A model with its disguised copies looks like a star with a tail: a plain base, copies that differ from it through
    decoding only, and variants lined up by strength (LoRA 0.3, 0.7, 1.5 lie on one line through the base).

    1. Seed: the endpoint farthest from the strongest outlier (the base sits at the end of the line), or with no clear
       outlier the endpoint closest to everything.
    2. The group grows from the seed while the two tests cannot tell the next-closest endpoint from the group's pooled
       replies.
    3. An endpoint outside the group still joins it if it shows a decoding change: its favourite answers are the
       group's, or it is much more, or much less, self-consistent with only a mild shift.
    4. What is left is grouped the same way, starting again from step 1."""
    if len(members) == 1:
        return {members[0]: first}, {members[0]: "on its own"}
    n = len(members)
    z = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        z[i, j] = z[j, i] = max(0.0, ev.z(members[i], members[j]), ev.style(members[i], members[j]))
    spread = {i: float(np.median([z[i, j] for j in range(n) if j != i])) for i in range(n)}
    outlier = max(spread, key=spread.get)
    strategy = params.get("seed_strategy", "outlier")
    if strategy == "neighbours":
        seed = max(range(n), key=lambda i: (sum(z[i, j] < params["same_z"] for j in range(n) if j != i), -spread[i]))
        how = "most indistinguishable neighbours"
    elif strategy == "central":
        seed = min(spread, key=spread.get)
        how = "closest to everything else"
    elif spread[outlier] >= params["same_z"] and n > 2:
        seed = max((i for i in range(n) if i != outlier), key=lambda i: (z[i, outlier], -spread[i]))
        how = f"farthest from the strongest outlier {members[outlier]}"
    else:
        seed = min(spread, key=spread.get)
        how = "closest to everything else"
    base = [seed]
    while len(base) < n:
        if params.get("growth", "pooled") == "complete":
            distance = {i: max(z[i, b] for b in base) for i in range(n) if i not in base}
        else:
            distance = {i: ev.against(members[i], [members[b] for b in base]) for i in range(n) if i not in base}
        nearest = min(distance, key=distance.get)
        if distance[nearest] >= params["same_z"]:
            break
        base.append(nearest)
    base_members = [members[i] for i in base]
    base_self = float(np.median([ev.profiles[e]["self"] for e in base_members]))
    out, why, apart = {}, {}, []
    for i, e in enumerate(members):
        if i in base:
            out[e], why[e] = first, f"seed of group {first} ({how})" if i == seed else f"not told apart from group {first}"
            continue
        far = ev.against(e, base_members)
        cosine = float(np.mean([ev.pair(e, b)["cosine"] for b in base_members]))
        flips = [ev.flip(e, b) for b in base_members]
        shared = sum(f["agree"] + f["flip"] for f in flips)
        flip_rate = sum(f["flip"] for f in flips) / shared if shared else 0.0
        steadier = ev.profiles[e]["self"] / base_self if base_self else 1.0
        if cosine >= params["keep_cosine"] and flip_rate <= params["keep_flip"] and shared >= 10:
            out[e], why[e] = first, f"decoding change: same favourite answers as group {first} (agreement {cosine:.3f}, {flip_rate:.1%} contradicted)"
        elif steadier >= params["watermark_self"] and (far < params["mild_shift"] or not params["watermark_cap"]):
            out[e], why[e] = first, f"watermark or colder sampling: {steadier:.2f} times as self-consistent as group {first} (z {far:.1f})"
        elif steadier <= params["hotter_self"] and far < params["mild_shift"]:
            out[e], why[e] = first, f"hotter sampling: {steadier:.1f} times as self-consistent as group {first}, mild shift (z {far:.1f})"
        else:
            apart.append(e)
            why[e] = f"not group {first}: z {far:.1f}, agreement {cosine:.3f}, {flip_rate:.1%} contradicted"
    if apart:
        inner, inner_why = base_groups(ev, apart, params, first + 1)
        for e in apart:
            out[e], why[e] = inner[e], why[e] + "; " + inner_why[e]
    return out, why


def models_in_family(ev: Evidence, members: list, params: dict) -> tuple[dict, dict]:
    """({endpoint: model number}, {endpoint: why}) inside one family.

    Endpoints with a hidden prompt are set aside first: a prompt changes style far more than a change of weights does,
    so they would only blur the groups. The others are grouped into models. Each prompted endpoint then joins the
    group whose answers to the packed questions it agrees with most."""
    if len(members) == 1:
        return {members[0]: 1}, {members[0]: "alone in its family"}
    hidden = prompted(ev, members, params)
    plain = [e for e in members if e not in hidden]
    if not plain:
        plain, hidden = members, {}
    out, why = base_groups(ev, plain, params)
    groups = {}
    for e in plain:
        groups.setdefault(("family", out[e]), []).append(e)
    for e, reason in hidden.items():
        best = attach(ev, e, groups, params)[1]
        out[e], why[e] = best, f"{reason}; joins group {best}"
    return out, why


def nameless(ev: Evidence, params: dict) -> dict:
    """{endpoint: reason} for endpoints that never use any of the usual maker names when asked who made them.

    Judged over all endpoints, before families exist: a hidden prompt wrecks the packed answers the family split
    relies on, so these endpoints have to be set aside first."""
    used = {e: [features.names_used(text) for text in features.who_replies(ev.data[e])] for e in ev.names}
    keeps = Counter()
    for e, replies in used.items():
        counts = Counter(name for names in replies for name in names)
        keeps.update(name for name, c in counts.items() if c >= max(2, 0.5 * len(replies)))
    usual = {name for name, n in keeps.items() if n >= params["usual_share"]}
    out = {}
    if usual:
        for e, replies in used.items():
            if len(replies) >= 2 and sum(bool(names & usual) for names in replies) / len(replies) <= params["prompted_names"]:
                out[e] = "hidden prompt or persona: never names a usual maker (" + "/".join(sorted(usual)[:4]) + ")"
    return out


def labels_from(ev: Evidence, params: dict = PARAMS, explain: dict | None = None, known_families: dict | None = None) -> dict:
    """Labels from evidence that is already measured. Changing a threshold or a rule then costs nothing to re-score.

    1. Endpoints with a hidden prompt (they never name a usual maker) are set aside.
    2. The others are split into families, and each family into models.
    3. Every endpoint set aside joins the model whose packed answers it agrees with most."""
    if isinstance(ev, Evidence):
        # Historical settings omit this option and retain their original method.
        # Cache namespaces prevent stale statistics when a search switches it.
        ev.use_null_method(params.get("null_method", "montecarlo"))
    if known_families is not None:
        missing = set(ev.names) - known_families.keys()
        if missing:
            raise ValueError("Missing supplied families for: " + ", ".join(sorted(missing)))
        # This diagnostic must never reattach an excluded endpoint across a supplied
        # family boundary. Run the model stage inside every complete supplied family.
        out = {}
        for f, family_name in enumerate(sorted({known_families[e] for e in ev.names}), 1):
            members = [e for e in ev.names if known_families[e] == family_name]
            inner, why = models_in_family(ev, members, params)
            for e in members:
                out[e] = (f"F{f}", f"F{f}-M{inner[e]}")
            if explain is not None:
                explain.update(why)
        return out
    aside = nameless(ev, params) if params.get("aside_first", True) else {}
    kept = [e for e in ev.names if e not in aside]
    if len(kept) < 2:
        aside, kept = {}, list(ev.names)
    family = dict(zip(kept, families(ev, params, kept)))
    out, reasons = {}, {}
    for f in sorted(set(family.values())):
        members = [e for e in kept if family[e] == f]
        inner, why = models_in_family(ev, members, params)
        for e in members:
            out[e] = (f"F{f}", f"F{f}-M{inner[e]}")
            reasons[e] = why[e]
    groups = {}
    for e, label in out.items():
        groups.setdefault(label, []).append(e)
    for e, reason in aside.items():
        best = attach(ev, e, groups, params)
        out[e] = best
        reasons[e] = f"{reason}; joins {best[1]}"
    if explain is not None:
        explain.update(reasons)
    return out


def labels(data: dict, params: dict = PARAMS, repeats: dict | None = None, explain: dict | None = None, permutations: int = 300,
           known_families: dict | None = None) -> dict:
    """{endpoint: (family label, model label)}. `explain`, if given, is filled with one reason per endpoint.
    `known_families` ({endpoint: family}) skips the family step: a diagnostic for the model step alone."""
    ev = Evidence(data, repeats, permutations, packs_in_tests=params.get("packs_in_tests", False),
                  model_prompts=params.get("model_prompts", "all"), style_normalization=params.get("style_normalization", "global"))
    return labels_from(ev, params, explain, known_families)


def read_truth(path: Path, prefix: str) -> dict:
    """{endpoint: (family, model, kind)}."""
    return scoring.read_labels(path, prefix)


def evaluate(truth: dict, guess: dict) -> dict:
    names = sorted(set(truth) & set(guess))
    result = scoring.score({e: truth[e] for e in names}, {e: guess[e] for e in names})
    one_family = len({truth[e][0] for e in names}) == 1
    family, model = result["family_mcc"], result["model_mcc"] or 0.0
    # The competition formula. An undefined MCC counts as 0, except that a truth with a single family has no family
    # term at all (the organizers' practice scorer reports the model MCC alone).
    return {"family": family, "model": model, "score": model if one_family else 0.1 * max(family or 0.0, 0) + 0.9 * max(model, 0),
            "one_family": one_family, "counts": result["model_counts"], "wrong": result["wrong"]}


def replays(data: dict, truth: dict, params: dict = PARAMS, per_pack: int = 2, n: int = 30, seed: int = 0) -> list:
    """Scores when each endpoint has only `per_pack` answers per pack, drawn at random from the pool."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        choice = {e: features.draw_repeats(data[e], per_pack, rng) for e in data}
        out.append(evaluate(truth, labels(data, params, choice, permutations=60))["score"])
    return out


def subsample(data: dict, plan: dict, rng: random.Random) -> dict:
    """The answers an endpoint would have under a query plan {prompt id: how many answers}, drawn from the pool."""
    out = {}
    for e, answers in data.items():
        out[e] = {}
        for pid, count in plan.items():
            have = sorted(answers.get(pid, {}), key=str)
            if have:
                out[e][pid] = {k: answers[pid][k] for k in rng.sample(have, min(count, len(have)))}
    return out


def budget_replays(data: dict, truth: dict, plan: dict, n: int = 20, params: dict = PARAMS, seed: int = 0) -> list:
    """Scores when every endpoint has only the answers in `plan`."""
    rng = random.Random(seed)
    return [evaluate(truth, labels(subsample(data, plan, rng), params, permutations=100))["score"] for _ in range(n)]


def auc(same: list, different: list) -> float:
    pairs = len(same) * len(different)
    return sum((s > d) + 0.5 * (s == d) for s in same for d in different) / pairs if pairs else float("nan")


def explore(path: Path, truth_path: Path, prefix: str, params: dict = PARAMS) -> None:
    """Prints every piece of evidence next to the known answers: per endpoint, then per pair."""
    data, truth = features.load(path, prefix), read_truth(truth_path, prefix)
    data = {e: data[e] for e in data if e in truth}
    names = sorted(data)
    families = sorted({truth[e][0] for e in names})
    consensus = {fam: features.consensus_identity(data, [e for e in names if truth[e][0] == fam]) for fam in families}
    print(f"{path.name}: {len(names)} endpoints in {len(families)} families")
    print(f"  {'endpoint':8s} {'what it is':30s} {'self':>5s} {'words':>6s} {'parsed':>6s} {'clear':>5s} {'opening':>7s} {'identity':>8s} replies")
    for e in names:
        p = features.profile(data[e])
        rate = features.identity_rate(data[e], consensus[truth[e][0]])
        print(f"  {e:8s} {truth[e][2][:30]:30s} {p['self']:5.2f} {p['words']:6.0f} {p['parsed']:6.1f} {p['clear']:5d} {p['opening_repeat']:7.1f} {rate:8.2f} "
              f"{sum(len(v) for v in data[e].values())}")
    ev = Evidence(data, permutations=200)
    rows = []
    for a, b in itertools.combinations(names, 2):
        if truth[a][0] != truth[b][0]:
            continue
        s, fl = ev.pair(a, b), ev.flip(a, b)
        m, o = ev.on(a, b, "m"), ev.on(a, b, "o")
        rows.append({"a": a, "b": b, "same": truth[a][:2] == truth[b][:2], "cosine": s["cosine"], "z": ev.z(a, b), "style": ev.style(a, b),
                     "flip": fl["rate"], "flips": f"{fl['flip']}/{fl['flip'] + fl['agree']}",
                     "micro": max(m["style"], m["z"]), "over": max(o["style"], o["z"]), "mz": m["z"], "oz": o["z"]})
    for name, sign in (("cosine", 1), ("z", -1), ("style", -1), ("flip", -1), ("micro", -1), ("over", -1), ("mz", -1), ("oz", -1)):
        same, different = [sign * r[name] for r in rows if r["same"]], [sign * r[name] for r in rows if not r["same"]]
        print(f"  {name:7s} ranks same-model pairs above different-model pairs: AUC {auc(same, different):.3f}")
    print(f"  {'pair':16s} {'truth':5s} {'micro':>6s} {'over':>6s} {'style':>6s} {'z':>6s} {'cosine':>6s} {'flips':>7s}  what they are")
    for r in sorted(rows, key=lambda r: r["micro"]):
        print(f"  {r['a'] + ' ' + r['b']:16s} {'SAME' if r['same'] else 'diff':5s} {r['micro']:+6.2f} {r['over']:+6.2f} {r['style']:+6.2f} {r['z']:+6.2f} {r['cosine']:6.3f} {r['flips']:>7s}  "
              f"{truth[r['a']][2][:26]} | {truth[r['b']][2][:26]}")


def report(path: Path, truth_path: Path, prefix: str, name: str, label: str, stage: str, params: dict = PARAMS) -> int:
    """Labels and scores one dataset, writes its result card and its labels file."""
    data, truth = features.load(path, prefix), read_truth(truth_path, prefix)
    data = {e: data[e] for e in data if e in truth}
    reasons = {}
    guess = labels(data, params, explain=reasons)
    full = evaluate(truth, guess)
    given = evaluate(truth, labels(data, params, known_families={e: truth[e][0] for e in data})) if not full["one_family"] else full
    runs = replays(data, truth, params, n=30 if len(data) <= 12 else 5)
    median, low = statistics.median(runs), float(np.percentile(runs, 10))
    out = config.DATA / f"labels-{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows([e, guess[e][0], guess[e][1]] for e in sorted(guess))
    tp, fp, fn, tn = full["counts"]
    family = "one family" if full["one_family"] else (f"{full['family']:.2f}" if full["family"] is not None else "n/a")
    wrong = [[a, kind_a, b, kind_b, "same" if said else "different"] for a, kind_a, b, kind_b, said in full["wrong"]]
    note = (f"Pipeline v7 (hidden prompts set aside, the rest grouped, prompted endpoints attached by answers). With every answer collected: {full['score']:.2f}. "
            f"With 2 answers per pack, {len(runs)} replays: median {median:.2f}, 10th percentile {low:.2f}.")
    slot = {"practice": "practice", "testbed": "testbed"}.get(name)
    findings.write(
        f"score-{name}", f"Pipeline score · {label}", stage,
        tiles=[{"label": "Score, whole pool", "value": f"{full['score']:.2f}", "hint": "0.1 × family + 0.9 × model" if not full["one_family"] else "model score only: one family"},
               {"label": "Score at the first-answer budget", "value": f"{median:.2f}", "hint": f"2 answers per pack, median of {len(runs)} replays; 10th percentile {low:.2f}"},
               {"label": "Family score", "value": family, "hint": "MCC over all pairs"},
               {"label": "Model score", "value": f"{full['model']:.2f}", "hint": f"{tp + tn} pairs right, {fp} wrongly same, {fn} wrongly different"},
               {"label": "Model score if families were known", "value": f"{given['model']:.2f}", "hint": "isolates the model step"}],
        table={"columns": ["Endpoint", "What it is", "Endpoint", "What it is", "Pipeline said"], "rows": wrong} if wrong else None,
        compare=[{"dataset": label, "what": "Pipeline v7, every answer collected", "family": family, "model": f"{full['model']:.2f}", "score": f"{full['score']:.2f}", "note": note}],
        text=[note, f"Answers used: {sum(len(v) for e in data for v in data[e].values())} from {len(data)} endpoints."], details=str(out.relative_to(config.ROOT)),
        headline=[{"slot": slot, "label": f"{label.split(' (')[0]} score", "value": f"{full['score']:.2f}",
                   "hint": f"pipeline on every answer collected ({sum(len(v) for e in data for v in data[e].values()) // max(len(data), 1)} per endpoint)"}],
    )
    print(note)
    print(f"  family score {family}; model score {full['model']:.2f}; model score with known families {given['model']:.2f}")
    for e in sorted(guess):
        print(f"  {e} -> {guess[e][1]:8s} truth {truth[e][1]:28s} {truth[e][2][:30]:30s} {reasons.get(e, '')}")
    for row in wrong:
        print(f"  {row[0]} ({row[1]}) and {row[2]} ({row[3]}): said {row[4]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["explore", "score", "budget"])
    parser.add_argument("file", type=Path)
    parser.add_argument("--truth", type=Path, default=config.PRACTICE_TRUTH)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--name", default="practice")
    parser.add_argument("--label", default="Practice endpoints (9, one family)")
    parser.add_argument("--stage", default="2")
    args = parser.parse_args()
    if args.command == "explore":
        explore(args.file, args.truth, args.prefix)
        return 0
    if args.command == "budget":
        data, truth = features.load(args.file, args.prefix), read_truth(args.truth, args.prefix)
        data = {e: data[e] for e in data if e in truth}
        for name in ("4", "5"):
            plan = probes.TRANCHES[name]
            short = {pid: n for pid, n in plan.items() if any(len(data[e].get(pid, {})) < n for e in data)}
            runs = budget_replays(data, truth, plan)
            print(f"after stage {name} ({probes.tranche_total(name)} queries per endpoint): median {statistics.median(runs):.2f}, "
                  f"10th percentile {float(np.percentile(runs, 10)):.2f}, min {min(runs):.2f}, max {max(runs):.2f}"
                  + (f"  [pool is short for {sorted(short)}]" if short else ""))
        return 0
    return report(args.file, args.truth, args.prefix, args.name, args.label, args.stage)


if __name__ == "__main__":
    sys.exit(main())
