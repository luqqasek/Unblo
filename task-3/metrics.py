"""Challenge formula. Requires REAL labels and gold logits; never infers them.

CE uses T=1; KL uses T=2. Gold loss mean/std is averaged across models,
not computed from ensemble logits. ddof is explicit because the brief omits it.
Zero original distances are undefined in the brief and raise instead of guessing.
"""
import argparse
import json
import numpy as np


def log_softmax(z, temperature=1.):
    z = np.asarray(z, dtype=np.float64) / temperature
    if not np.isfinite(z).all():
        raise ValueError('Logits must be finite')
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kl(p, log_q):
    p = np.asarray(p, dtype=np.float64)
    return (p * (np.log(np.maximum(p, 1e-300)) - log_q)).sum(-1)


def ce(z, labels):
    labels = np.asarray(labels)
    if labels.shape != (len(z),) or labels.dtype.kind not in 'iu':
        raise ValueError('Expected one integer true label per image')
    if np.any(labels < 0) or np.any(labels >= np.shape(z)[-1]):
        raise ValueError('Labels outside class range')
    return -log_softmax(z)[np.arange(len(z)), labels]


def improvement(distance, baseline):
    if baseline <= 1e-12:
        raise ValueError('Original-to-gold denominator is zero; server convention unknown')
    return float(1 - np.clip(distance / baseline, 0, 1))


def forget_group(orig, sub, gold, labels, ddof=0):
    orig, sub, gold = map(np.asarray, (orig, sub, gold))
    if gold.ndim != 3 or gold.shape[1:] != orig.shape or sub.shape != orig.shape:
        raise ValueError('Expected orig/sub [N,C], gold [models,N,C]')
    losses = np.stack([ce(g, labels) for g in gold])
    gm, gs = losses.mean(axis=1).mean(), losses.std(axis=1, ddof=ddof).mean()
    lo, ls = ce(orig, labels), ce(sub, labels)
    do = abs(lo.mean()-gm) + abs(lo.std(ddof=ddof)-gs)
    ds = abs(ls.mean()-gm) + abs(ls.std(ddof=ddof)-gs)
    pg = np.exp(log_softmax(gold, 2.)).mean(axis=0)
    ko, ks = kl(pg, log_softmax(orig, 2.)).mean(), kl(pg, log_softmax(sub, 2.)).mean()
    return dict(mean_std_score=improvement(ds, do), prediction_score=improvement(ks, ko),
                orig_loss_distance=float(do), sub_loss_distance=float(ds),
                orig_gold_kl=float(ko), sub_gold_kl=float(ks), gold_mean=float(gm), gold_std=float(gs))


def score(exact, altered, control_orig, control_sub, control_labels, validation_accuracy, ddof=0):
    if not 0 <= validation_accuracy <= 1:
        raise ValueError('validation_accuracy must be in [0,1]')
    if validation_accuracy < .7:
        return dict(final_score=0., forget_score=0., retain_score=0., utility_gate_passed=False)
    a, b = (forget_group(*group, ddof=ddof) for group in (exact, altered))
    ms = (a['mean_std_score'] + b['mean_std_score']) / 2
    ps = (a['prediction_score'] + b['prediction_score']) / 2
    ao = np.mean(np.argmax(control_orig, -1) == control_labels)
    ass = np.mean(np.argmax(control_sub, -1) == control_labels)
    if ao == 0:
        raise ValueError('Original control accuracy zero; server convention unknown')
    acc = float(np.clip(ass / ao, 0, 1))
    drift = float(kl(np.exp(log_softmax(control_orig, 2)), log_softmax(control_sub, 2)).mean())
    retain = acc / (1 + drift)
    return dict(final_score=ms*ps*retain, forget_score=ms*ps, retain_score=retain,
                mean_std_score=ms, prediction_score=ps, retain_accuracy_score=acc,
                retain_kl_similarity=1/(1+drift), utility_gate_passed=True,
                exact=a, altered=b, ddof=ddof)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions', help='NPZ with exact/altered_{orig,sub,gold,labels}, control_{orig,sub,labels}, validation_accuracy')
    p.add_argument('--ddof', type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    with np.load(args.predictions, allow_pickle=False) as d:
        groups = [tuple(d[f'{g}_{k}'] for k in ('orig','sub','gold','labels')) for g in ('exact','altered')]
        result = score(*groups, d['control_orig'], d['control_sub'], d['control_labels'], float(d['validation_accuracy']), args.ddof)
    print(json.dumps(result, indent=2, allow_nan=False))
