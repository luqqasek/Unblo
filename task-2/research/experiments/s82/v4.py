"""Rule v4 (session hackathon-82): identify the model from its weights where the base is available locally.

Per family:
  LORA      text fits a known adapter at some strength better than the base (gain >= threshold)
  QUANT     text fits a rebuilt quantized copy better than the base (gain > threshold)
  BASE      base-strength weights and a disguise typed from the text (prompt, raised temperature, repeats, digit skew)
  variant   base-strength weights, no disguise, but fitted temperature inflated (unknown variant): own model
  plain     among the remaining undisguised endpoints the one that fits the base best (lowest fitted temperature,
            then most negative rebuilt-copy gain per token) joins BASE; the others are their own models
"""
import statistics

CFG = {'lora': 4.0, 'lora_prompted': 15.0, 'quant': 5.0, 'T_inflate': 0.08, 'attach': True}
BUCKET = {0.15: 'weak', 0.3: 'weak', 0.5: 'mid', 0.7: 'mid', 1.0: 'strong', 1.5: 'xstrong'}


def label(eps, tags, lora=None, quant=None, cfg=None, exclude_plain=()):
    """eps: endpoint names; tags: {e: set}; lora: {adapter: {e: {scale: gain}}}; quant: {e: {'T', 'tokens', 'gain': {kind: g}}}."""
    c = {**CFG, **(cfg or {})}
    y, why, gain_of = {}, {}, {}
    cand = []
    ts = [quant[e]['T'] for e in eps if quant and e in quant and not tags[e]]
    t0 = statistics.median_low(ts) if ts else None            # the family's default decoding temperature as fitted
    for e in eps:
        prompted = bool(tags[e] & {'prompt', 'unnamed'})
        best = None
        for a, g in (lora or {}).items():
            if e in g:
                s = max(g[e], key=g[e].get)
                if best is None or g[e][s] > best[2]:
                    best = (a, s, g[e][s])
        q = quant.get(e) if quant else None
        qbest = max(q['gain'].items(), key=lambda kv: kv[1]) if q else None
        if best and best[2] >= (c['lora_prompted'] if prompted else c['lora']):
            # an undisguised LoRA endpoint is its own model (no configuration is deployed twice); disguised copies group by strength
            y[e] = f"LORA:{best[0]}:{BUCKET.get(best[1], best[1])}"; gain_of[e] = best[2]
            why[e] = f"{best[0]}@{best[1]} +{best[2]:.1f}"
        elif qbest and qbest[1] >= c['quant'] and not prompted:
            y[e] = f"QUANT:{qbest[0]}"; why[e] = f"{qbest[0]} +{qbest[1]:.1f}"
        elif tags[e]:
            y[e] = 'BASE'; why[e] = 'disguise: ' + ','.join(sorted(tags[e]))
        elif q and t0 is not None and q['T'] <= t0 - c['T_inflate']:
            y[e] = 'BASE'; why[e] = f"fitted T {q['T']:.2f} vs {t0:.2f}: colder sampling (decoding disguise)"
        elif q and t0 is not None and q['T'] >= t0 + c['T_inflate']:
            y[e] = e; why[e] = f"fitted T {q['T']:.2f} vs {t0:.2f}: unknown variant"
        else:
            y[e] = e; cand.append(e)
            why[e] = 'plain-like' + (f" (T {q['T']:.2f}, fp8 {qbest[1]:+.1f})" if q else '')
    # no configuration is deployed twice: two UNDISGUISED endpoints in one LoRA bucket are different strengths
    seen = {}
    for e in sorted(gain_of, key=lambda x: -gain_of[x]):
        if not tags[e]:
            if y[e] in seen:
                y[e] = y[e] + ':' + e
            else:
                seen[y[e]] = e
    cand = [e for e in cand if e not in exclude_plain and quant and e in quant]      # only pick a plain base on evidence
    if c['attach'] and cand:
        def key(e):
            q = quant.get(e) if quant else None
            return (q['T'], min(q['gain'].values()) / max(1, q['tokens'])) if q else (9, 0)
        pick = min(cand, key=key)
        y[pick] = 'BASE'; why[pick] += ' -> taken as the plain base'
    return y, why
