"""From an answers file to pairwise evidence between endpoints.

The endpoints sample their answers, so a single answer says little. Each packed item is asked several times, and two
endpoints are compared by how often their answers to the same item agree, set against how often each agrees with
itself.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import probes


def load(path: Path, prefix: str) -> dict:
    """{endpoint: {prompt id: {repeat: response}}} for endpoints whose name starts with prefix."""
    data = defaultdict(lambda: defaultdict(dict))
    by_text = {pid: text for pid, text in probes.long_prompts()}
    canonical = {text: pid for pid, text in by_text.items()}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r["endpoint"].startswith(prefix):
                continue
            # The same prompt text under another id (the stage 1 recon prompt) counts as more answers to that prompt.
            pid = canonical.get(r.get("prompt", ""), r["id"])
            slot = r.get("repeat", 0) if pid == r["id"] else f"{r['id']}:{r.get('repeat', 0)}"
            data[r["endpoint"]][pid][slot] = r.get("response") or ""
    return data


def item_samples(answers: dict, repeats: dict | None = None, groups: tuple = ("knowledge", "choice", "skill")) -> dict:
    """{(pack, item number): [canonical answers]} for one endpoint. `repeats` picks which repeats of each pack to use
    ({pack: [repeat, ...]}); by default all of them."""
    out = defaultdict(list)
    for pack, (_, group, _) in probes.PACKS.items():
        if group not in groups:
            continue
        chosen = repeats.get(pack, []) if repeats is not None else sorted(answers.get(pack, {}), key=str)
        for rep in chosen:
            if rep in answers.get(pack, {}):
                for number, answer in probes.read_pack(pack, answers[pack][rep]).items():
                    out[(pack, number)].append(answer)
    return out


def pair(a: dict, b: dict) -> dict:
    """Agreement statistics between two endpoints' item samples.

    cross: chance that an answer from one matches an answer from the other on the same item
    self_a, self_b: chance that two answers from the same endpoint match
    cosine: cross / sqrt(self_a * self_b). 1.0 when both draw from the same distribution; a temperature change lowers it
            only a little, a change of the preferred answers lowers it a lot
    modes: share of items where the two most frequent answers coincide (items where both have a clear favourite)
    """
    cross = self_a = self_b = paired_cross = hist_a = hist_b = 0.0
    used = paired = mode_same = mode_all = 0
    for key in sorted(a.keys() & b.keys()):
        xs, ys = a[key], b[key]
        if not xs or not ys:
            continue
        ca, cb = Counter(xs), Counter(ys)
        item_cross = sum(ca[v] * cb[v] for v in ca) / (len(xs) * len(ys))
        cross += item_cross
        hist_a += sum(n * n for n in ca.values()) / len(xs) ** 2
        hist_b += sum(n * n for n in cb.values()) / len(ys) ** 2
        used += 1
        if len(xs) >= 2 and len(ys) >= 2:
            paired_cross += item_cross
            self_a += sum(n * (n - 1) for n in ca.values()) / (len(xs) * (len(xs) - 1))
            self_b += sum(n * (n - 1) for n in cb.values()) / (len(ys) * (len(ys) - 1))
            paired += 1
            (top_a, n_a), (top_b, n_b) = ca.most_common(1)[0], cb.most_common(1)[0]
            if n_a * 2 > len(xs) and n_b * 2 > len(ys):
                mode_all += 1
                mode_same += top_a == top_b
    if not used:
        return {"items": 0, "cross": 0.0, "self_a": 0.0, "self_b": 0.0, "cosine": 0.0, "matched_cosine": 0.0, "hist_cosine": 0.0, "modes": 0.0, "mode_items": 0}
    hist_cosine = cross / (hist_a * hist_b) ** 0.5
    cross /= used
    if not paired:                       # one answer per item: no self-agreement to set the agreement against
        return {"items": used, "cross": cross, "self_a": 0.0, "self_b": 0.0, "cosine": cross, "matched_cosine": cross, "hist_cosine": hist_cosine, "modes": 0.0, "mode_items": 0}
    self_a, self_b = self_a / paired, self_b / paired
    denominator = (self_a * self_b) ** 0.5
    return {"items": used, "cross": cross, "self_a": self_a, "self_b": self_b,
            "cosine": cross / denominator if denominator else 0.0,
            "matched_cosine": (paired_cross / paired) / denominator if denominator else 0.0,
            "hist_cosine": min(1.0, hist_cosine),
            "modes": mode_same / mode_all if mode_all else 0.0, "mode_items": mode_all}


def single_cross(a: dict, b: dict) -> float:
    """Agreement when each side has one answer per item (the smallest budget): share of shared items that match."""
    keys = [k for k in a.keys() & b.keys() if a[k] and b[k]]
    return sum(a[k][0] == b[k][0] for k in keys) / len(keys) if keys else 0.0


def draw_repeats(answers: dict, per_pack: int, rng: random.Random) -> dict:
    """A random choice of `per_pack` repeats for each pack: one replay of a smaller query budget."""
    out = {}
    for pack in probes.PACKS:
        have = sorted(answers.get(pack, {}), key=str)
        out[pack] = rng.sample(have, min(per_pack, len(have)))
    return out


_WORD = re.compile(r"[A-Za-z']+")


def opening_repeat(answers: dict) -> float:
    """How many opening words two answers to the same long prompt share, on average. Sampling gives close to 0;
    a watermark keyed on the preceding text, or greedy decoding, gives a long shared opening."""
    shared = []
    for pid, by_repeat in answers.items():
        if pid in probes.PACKS or len(by_repeat) < 2:
            continue
        texts = [by_repeat[k] for k in sorted(by_repeat, key=str)]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                n = 0
                for x, y in zip(_WORD.findall(texts[i]), _WORD.findall(texts[j])):
                    if x != y:
                        break
                    n += 1
                shared.append(n)
    return sum(shared) / len(shared) if shared else 0.0


# ---- Distribution test between two endpoints (after Model Equality Testing: MMD with a matching kernel) ----

def _opening(text: str, length: int = 24) -> list:
    return [w.lower() for w in _WORD.findall(text or "")[:length]]


def represent(prompt_id: str, response: str):
    """A reply in comparable form: its opening words, its word counts, its length, and for a pack its item answers."""
    words = [w.lower() for w in _WORD.findall(response or "")]
    items = probes.read_pack(prompt_id, response) if prompt_id in probes.PACKS else None
    return (items, _opening(response), Counter(words), len(words))


def kernel(x, y) -> float:
    """Similarity of two replies to the same prompt, between 0 and 1: what they say and how they say it."""
    opening = sum(a == b for a, b in zip(x[1], y[1])) / 24
    shared = sum((x[2] & y[2]).values())
    overlap = 2 * shared / (x[3] + y[3]) if x[3] + y[3] else 0.0
    if x[0] is None:
        return 0.5 * opening + 0.5 * overlap
    keys = x[0].keys() | y[0].keys()
    items = sum(x[0].get(k) == y[0].get(k) for k in keys) / len(keys) if keys else 0.0
    return 0.5 * items + 0.25 * opening + 0.25 * overlap


def represented(answers: dict, repeats: dict | None = None, prompts: set | None = None) -> dict:
    """{prompt id: [representations]} for one endpoint; `repeats` limits which repeats of each prompt are used."""
    out = {}
    for pid, by_repeat in answers.items():
        if prompts is not None and pid not in prompts:
            continue
        chosen = sorted(by_repeat, key=str) if repeats is None or pid not in repeats else [r for r in repeats[pid] if r in by_repeat]
        if chosen:
            out[pid] = [represent(pid, by_repeat[r]) for r in chosen]
    return out


def kernel_matrix(replies: list):
    """Exact matching-kernel matrix, computing repeated answers only once.

    Identical replies at different sample positions retain their nonzero
    similarity. Only the sample diagonal is excluded from the U-statistic.
    """
    import numpy as np
    unique, lookup, indices = [], {}, []
    for reply in replies:
        items, opening, counts, length = reply
        key = (None if items is None else tuple(sorted(items.items())),
               tuple(opening), tuple(sorted(counts.items())), length)
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(reply)
        indices.append(lookup[key])
    small = np.zeros((len(unique), len(unique)))
    for i, left in enumerate(unique):
        for j in range(i, len(unique)):
            small[i, j] = small[j, i] = kernel(left, unique[j])
    order = np.asarray(indices, dtype=int)
    matrix = small[order[:, None], order[None, :]]
    np.fill_diagonal(matrix, 0.0)
    return matrix


def mmd(a: dict, b: dict, permutations: int = 300, seed: int = 0) -> dict:
    """How different two endpoints' reply distributions are, summed over the prompts both answered at least twice.

    stat: sum over prompts of  mean k(a, a') + mean k(b, b') - 2 mean k(a, b).  Near 0 for the same distribution.
    z:    stat in units of what relabelling the replies at random produces. Around 0 for the same distribution,
          large when the endpoints differ. It needs no threshold tuned to a model family.
    The relabelling is seeded, so the same pair always gets the same z."""
    import numpy as np
    blocks = []
    for pid in sorted(a.keys() & b.keys()):
        xs, ys = a[pid], b[pid]
        if len(xs) < 2 or len(ys) < 2:
            continue
        pooled = xs + ys
        k = kernel_matrix(pooled)
        blocks.append((k, len(xs)))
    if not blocks:
        return {"prompts": 0, "stat": 0.0, "z": 0.0}
    rng = np.random.default_rng(seed)

    def statistic(k, n_a: int, orders):                     # orders: (P, n) index arrays; the diagonal of k is 0
        left, right = orders[:, :n_a], orders[:, n_a:]
        n_b = right.shape[1]
        aa = k[left[:, :, None], left[:, None, :]].sum(axis=(1, 2)) / (n_a * (n_a - 1))
        bb = k[right[:, :, None], right[:, None, :]].sum(axis=(1, 2)) / (n_b * (n_b - 1))
        ab = k[left[:, :, None], right[:, None, :]].sum(axis=(1, 2)) / (n_a * n_b)
        return aa + bb - 2 * ab

    observed, null = 0.0, np.zeros(permutations)
    for k, n_a in blocks:
        n = len(k)
        observed += float(statistic(k, n_a, np.arange(n)[None, :])[0])
        null += statistic(k, n_a, rng.permuted(np.tile(np.arange(n), (permutations, 1)), axis=1))
    sd = float(null.std())
    return {"prompts": len(blocks), "stat": observed, "z": (observed - float(null.mean())) / sd if sd else 0.0}


# ---- Temperature-robust evidence: favourite answers that changed ----

def clear_modes(samples: dict, need: int = 3) -> dict:
    """{item: answer} for the items an endpoint answered the same way every time, with at least `need` answers.
    A temperature change blurs answers but does not create a new unanimous favourite; a weight change can."""
    return {key: answers[0] for key, answers in samples.items() if len(answers) >= need and len(set(answers)) == 1}


def flips(a: dict, b: dict, need: int = 3) -> dict:
    """Items on which both endpoints are unanimous: how many agree, how many differ."""
    ma, mb = clear_modes(a, need), clear_modes(b, need)
    shared = ma.keys() & mb.keys()
    differ = sum(ma[k] != mb[k] for k in shared)
    return {"agree": len(shared) - differ, "flip": differ, "rate": differ / len(shared) if shared else 0.0,
            "clear_a": len(ma), "clear_b": len(mb)}


# ---- Per-endpoint profile ----

IDENTITY_PROMPTS = ("recon-extract-1", "recon-extract-2", "recon-extract-3", "f2-who")


def identity_replies(answers: dict) -> list:
    return [text for pid in IDENTITY_PROMPTS for _, text in sorted(answers.get(pid, {}).items(), key=lambda kv: str(kv[0]))]


def trigrams(text: str) -> set:
    words = [w.lower() for w in _WORD.findall(text or "")]
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def consensus_identity(data: dict, members: list, share: float = 0.5) -> set:
    """Word triples that at least `share` of the endpoints use when asked who they are or what came before the message.
    For a family served with a default system prompt this recovers that prompt's wording."""
    counts = Counter()
    for e in members:
        seen = set()
        for text in identity_replies(data[e]):
            seen |= trigrams(text)
        counts.update(seen)
    return {g for g, n in counts.items() if n >= max(2, share * len(members))}


def identity_rate(answers: dict, consensus: set) -> float:
    """Share of an endpoint's identity replies that use the family's usual wording. Near 0 points to a hidden prompt."""
    replies = identity_replies(answers)
    return sum(bool(trigrams(text) & consensus) for text in replies) / len(replies) if replies else 0.0


def profile(answers: dict) -> dict:
    """One endpoint in a few numbers: how consistent, how wordy, how well it keeps the pack format."""
    samples = item_samples(answers)
    pairs = same = 0
    for values in samples.values():
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pairs += 1
                same += values[i] == values[j]
    replies = [text for pack in probes.PACKS for text in answers.get(pack, {}).values()]
    parsed = [len(probes.read_pack(pack, text)) for pack in probes.PACKS for text in answers.get(pack, {}).values()]
    lengths = [len(_WORD.findall(text)) for text in replies]
    return {"self": same / pairs if pairs else 0.0, "pack_replies": len(replies),
            "words": sum(lengths) / len(lengths) if lengths else 0.0,
            "parsed": sum(parsed) / len(parsed) if parsed else 0.0,
            "clear": len(clear_modes(samples)), "opening_repeat": opening_repeat(answers)}


# ---- Style test: consistent shifts in how replies look, pooled across prompts ----

_COMMON = ("the", "a", "is", "are", "of", "in", "and", "to", "there", "it", "i", "you", "that", "this", "for", "with",
           "as", "on", "was", "be", "answer", "number", "year", "letter", "word", "name", "pick", "sure", "here")


def style_vector(prompt_id: str, text: str) -> list:
    """A reply as numbers that do not depend on what was asked: length, layout, repetition, common words."""
    import math
    text = text or ""
    words = [w.lower() for w in _WORD.findall(text)]
    lines = [line for line in text.splitlines() if line.strip()]
    chars = max(len(text), 1)
    counts = Counter(words)
    n = max(len(words), 1)
    items = probes.read_pack(prompt_id, text) if prompt_id in probes.PACKS else {}
    short = sum(len(v.split()) <= 1 for v in items.values())
    vector = [
        math.log1p(len(words)), math.log1p(len(text)), math.log1p(len(lines)),
        math.log1p(len(text) / max(len(lines), 1)),
        sum(c.isdigit() for c in text) / chars, sum(c.isupper() for c in text) / chars,
        sum(c in ".,;:!?" for c in text) / chars, sum(c in "*#`_|" for c in text) / chars,
        text.count("?") / max(len(lines), 1),
        len(counts) / n, (counts.most_common(1)[0][1] / n) if counts else 0.0,
        float(text.rstrip().endswith((".", "!", "?", ")", "`", '"'))),
        len(items) / 16, (short / len(items)) if items else 0.0,
    ]
    return vector + [counts[w] / n for w in _COMMON]


LAYOUT = 14          # length and layout numbers only: on the practice endpoints the word frequencies added noise, not signal


def style_sets(answers: dict, repeats: dict | None = None) -> dict:
    out = {}
    for pid, by_repeat in answers.items():
        chosen = sorted(by_repeat, key=str) if repeats is None or pid not in repeats else [r for r in repeats[pid] if r in by_repeat]
        if chosen:
            out[pid] = [style_vector(pid, by_repeat[r])[:LAYOUT] for r in chosen]
    return out


def style_scale(all_sets: list) -> tuple:
    """Mean and spread of every style number over all endpoints' replies, for putting them on one scale."""
    import numpy as np
    rows = np.array([v for sets in all_sets for vectors in sets.values() for v in vectors], dtype=float)
    if not rows.size:
        return np.zeros(LAYOUT), np.ones(LAYOUT)
    return rows.mean(axis=0), rows.std(axis=0) + 1e-6


def style_test(a: dict, b: dict, scale: tuple, permutations: int = 300, seed: int = 0) -> dict:
    """Do two endpoints' replies look different in a consistent way?

    For every prompt both answered: the difference of their mean style vectors. The differences are averaged over
    prompts, so a shift that points the same way everywhere adds up while noise cancels. z is that averaged shift's
    size in units of what swapping replies between the two endpoints at random produces."""
    import numpy as np
    rng = np.random.default_rng(seed)
    mean, spread = scale
    blocks = []
    for pid in sorted(a.keys() & b.keys()):
        xs, ys = np.array(a[pid], dtype=float), np.array(b[pid], dtype=float)
        blocks.append(((np.vstack([xs, ys]) - mean) / spread, len(xs)))
    if not blocks:
        return {"prompts": 0, "z": 0.0, "shift": 0.0}

    # Keep the same permutation-major RNG sequence as the scalar test, but do the
    # means in NumPy. Sorting prompts above makes the seed stable across processes.
    orders = [[] for _ in blocks]
    for _ in range(permutations):
        for i, (rows, _) in enumerate(blocks):
            orders[i].append(rng.permutation(len(rows)))
    observed_total = np.zeros_like(blocks[0][0][0])
    null_total = np.zeros((permutations, len(observed_total)))
    for (rows, n_a), permutations_for_block in zip(blocks, orders):
        order = np.asarray(permutations_for_block)
        observed_total += rows[:n_a].mean(axis=0) - rows[n_a:].mean(axis=0)
        null_total += rows[order[:, :n_a]].mean(axis=1) - rows[order[:, n_a:]].mean(axis=1)
    observed = float(np.sum((observed_total / len(blocks)) ** 2))
    null = np.sum((null_total / len(blocks)) ** 2, axis=1)
    sd = null.std()
    return {"prompts": len(blocks), "z": float((observed - null.mean()) / sd) if sd else 0.0, "shift": observed}


# ---- Who the endpoint says it is: names shared across a group of endpoints ----

_NAME = re.compile(r"(?<=[a-z,] )([A-Z][A-Za-z]{2,})")          # a capitalised word that does not start a sentence


def names_used(text: str) -> set:
    return set(_NAME.findall(text or ""))


def who_replies(answers: dict) -> list:
    """Replies to "who are you, and who made you?"."""
    return [text for _, text in sorted(answers.get("f2-who", {}).items(), key=lambda kv: str(kv[0]))]


def usual_names(data: dict, members: list, share: float = 0.5) -> set:
    """Proper names (maker, model name) that at least `share` of the endpoints mention when asked who made them."""
    counts = Counter()
    for e in members:
        seen = set()
        for text in who_replies(data[e]):
            seen |= names_used(text)
        counts.update(seen)
    return {name for name, n in counts.items() if n >= max(2, share * len(members))}


def name_rate(answers: dict, usual: set) -> float:
    """Share of an endpoint's "who made you" replies that mention one of the group's usual names."""
    replies = who_replies(answers)
    return sum(bool(names_used(text) & usual) for text in replies) / len(replies) if replies else 1.0


def admits_rate(answers: dict) -> float | None:
    """Share of replies to the instruction audit that start with YES. None if the audit was not asked."""
    replies = [text for _, text in sorted(answers.get("a1-audit", {}).items(), key=lambda kv: str(kv[0]))]
    if not replies:
        return None
    first = [(_WORD.findall(text) or [""])[0].lower() for text in replies]
    return sum(word == "yes" for word in first) / len(first)
