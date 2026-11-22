"""Rule v3 (session hackathon-82): black-box disguise typing + local-weights evidence.

  variant   the endpoint's text fits a known adapter at some strength better than the plain base (log-likelihood gain)
  disguise  black-box tags (prompt / hot / repeat / skew) or local decoding read-out (fitted temperature off the family's,
            or no token beyond rank 50 where siblings have some)
Labels: variants are grouped by (adapter, strength bucket); non-variant disguised endpoints share the base label;
non-variant undisguised endpoints are each their own model (plain base and quantized copies cannot be told apart).
"""
import json, statistics
from collections import defaultdict
import typed2 as T

BUCKET = {0.15: 'weak', 0.3: 'weak', 0.5: 'mid', 0.7: 'mid', 1.0: 'strong', 1.5: 'xstrong'}
CFG3 = {'gain': 4.0, 'gain_prompted': 15.0, 'dT': 0.15, 'topk_expected': 4.0, 'merge_strong': False}


def lora_gains(paths, keep=None):
    """{adapter: {endpoint: {scale: gain}}} and tokens per endpoint, from wb_score outputs ({adapter: path})."""
    out, toks = {}, defaultdict(int)
    for adapter, path in paths.items():
        tot = {}
        for line in open(path):
            j = json.loads(line)
            e = j['endpoint']
            if keep is not None and (e, j['id'], j['repeat']) not in keep:
                continue
            if e not in tot:
                tot[e] = {str(s): [0.0] * len(j['temps']) for s in j['scales']}
            for s in j['scales']:
                for i, v in enumerate(j['ll'][str(s)]):
                    tot[e][str(s)][i] += v
            if adapter == next(iter(paths)):
                toks[e] += j['tokens']
        out[adapter] = {e: {float(s): max(v) - max(t['0.0']) for s, v in t.items() if s != '0.0'} for e, t in tot.items()}
    return out, toks


def label(fam, gains=None, decode=None, cfg=None, tags=None):
    c = {**CFG3, **(cfg or {})}
    if tags is None:
        tags = T.types(T.features(fam))
    tags = {e: set(t) for e, t in tags.items()}
    info = {}
    if decode:
        ts = [decode[e]['T'] for e in fam if e in decode]
        t0 = statistics.median(ts) if ts else None
        r50 = statistics.median([decode[e]['rank>=50'] for e in fam if e in decode]) if ts else 0
        for e in fam:
            if e not in decode:
                continue
            d = decode[e]
            if d['T'] >= t0 + c['dT']:
                tags[e].add('hotT')
            if d['T'] <= t0 - c['dT']:
                tags[e].add('coldT')
            if d['rank>=50'] == 0 and r50 * d['tokens'] >= c['topk_expected']:
                tags[e].add('topk')
            info[e] = f"T {d['T']:.2f} r50 {d['rank>=50']:.4f}"
    y = {}
    for e in fam:
        best = None
        if gains:
            for a, g in gains.items():
                if e in g:
                    s = max(g[e], key=g[e].get)
                    if best is None or g[e][s] > best[2]:
                        best = (a, s, g[e][s])
        prompted = bool(tags[e] & {'prompt', 'unnamed'})
        thr = c['gain_prompted'] if prompted else c['gain']
        if best and best[2] >= thr:
            b = BUCKET.get(best[1], str(best[1]))
            if c['merge_strong'] and b == 'xstrong':
                b = 'strong'
            y[e] = f"LORA:{best[0]}:{b}"
            info[e] = info.get(e, '') + f" | {best[0]}@{best[1]} +{best[2]:.1f}"
        elif tags[e]:
            y[e] = 'BASE'
            if best:
                info[e] = info.get(e, '') + f" | best {best[0]}@{best[1]} {best[2]:+.1f}"
        else:
            y[e] = e
            if best:
                info[e] = info.get(e, '') + f" | best {best[0]}@{best[1]} {best[2]:+.1f}"
    return y, tags, info
