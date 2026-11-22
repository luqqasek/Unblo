"""Target construction for forget images from the original model only (no gold, no retain data, pseudo-labels)."""
import math
import torch, torch.nn.functional as F


def split(z, y):
    """-> delta (T=1 log-odds of class y vs the rest), log r (log-probs over the other classes; -1e9 at y)."""
    oh = F.one_hot(y, z.shape[1]).bool(); zy = z[oh]; zw = z.masked_fill(oh, -1e9); lse = zw.logsumexp(1)
    return zy - lse, zw - lse[:, None]


def compose(delta, logr, y):
    oh = F.one_hot(y, logr.shape[1]).bool(); z = logr.masked_fill(oh, 0.0).clone(); z[oh] = delta; return z


def mix_altered_wrong_classes(target_logits, teacher_view_logits, y, weight=0.5, temperature=1.5):
    """Adapt non-label probabilities to the current view, keeping target label log-odds fixed.

    This is an experimental target estimator, not a gold-model prediction.
    A zero weight returns the original logits exactly.
    """
    if not 0 <= weight <= 1 or temperature <= 0:
        raise ValueError("weight must be in [0,1] and temperature must be positive")
    if weight == 0:
        return target_logits
    delta, original_logr = split(target_logits.float(), y)
    _, view_logr = split(teacher_view_logits.float() / temperature, y)
    oh = F.one_hot(y, target_logits.shape[1]).bool()
    r = ((1-weight)*original_logr.exp() + weight*view_logr.exp()).masked_fill(oh, 0)
    logr = r.clamp_min(1e-30).log().masked_fill(oh, -1e9).log_softmax(1)
    return compose(delta, logr, y)


def hardness(z_exact, probes, y, names, weights=None):
    """Average true-class log-odds over the chosen probe views (lower = harder / more memorised)."""
    ds = [split(probes[n], y)[0] for n in names]
    w = torch.ones(len(ds)) if weights is None else torch.tensor(weights).float()
    return (torch.stack(ds) * w[:, None]).sum(0) / w.sum()


def wrong_dist(z_exact, probes, y, names, T=2.0, prior_w=0.5, eps=1e-4):
    """log r: blend of per-sample probe-view average and class-level prior, restricted to wrong classes."""
    C = z_exact.shape[1]; oh = F.one_hot(y, C).bool()
    ps = torch.stack([(probes[n] / T).masked_fill(oh, -1e9).softmax(1) for n in names]).mean(0)
    prior = torch.zeros(C, C)
    for c in y.unique(): prior[c] = ps[y == c].mean(0)
    r = prior_w * prior[y] + (1 - prior_w) * ps
    r = (r + eps / C).masked_fill(oh, 0.0); r = r / r.sum(1, keepdim=True)
    return r.clamp_min(1e-12).log().masked_fill(oh, -1e9)


def marginal_quantiles(n, mu, s, kind="logistic", skew=0.0):
    """Sorted target deltas (ascending) for n samples from a parametric marginal."""
    u = (torch.arange(n).float() + 0.5) / n
    if kind == "logistic": q = torch.log(u / (1 - u)) * (s * math.sqrt(3) / math.pi)
    else: q = torch.erfinv(2 * u - 1) * math.sqrt(2) * s
    return mu + q


def rank_match(signal, sorted_vals):
    return sorted_vals[signal.argsort().argsort()]


MILDISH = ["hflip", "con0.7", "con1.3", "bri0.7", "bri1.3", "blur1.5", "zoomout1.3", "mild0", "mild1", "mild2", "mild3"]


def nscore(v):
    """rank -> standard normal score"""
    u = (v.argsort().argsort().float() + 0.5) / len(v); return torch.erfinv(2 * u - 1) * 2 ** 0.5


def hardness_v2(z_exact, probes, y):
    """Mirror-validated hardness: normal-score average of (mean true-class log-odds over all probe views) and (exact margin)."""
    allp = torch.stack([split(probes[k], y)[0] for k in probes]).mean(0)
    return (nscore(allp) + nscore(split(z_exact, y)[0])) / 2


def rdist_v2(z_exact, probes, y, T=1.5, with_exact=False):
    """Mirror-validated wrong-class distribution: logit average over mild-ish views, temperature T."""
    oh = F.one_hot(y, z_exact.shape[1]).bool()
    zs = [probes[k] for k in MILDISH if k in probes] + ([z_exact] if with_exact else [])
    return (torch.stack(zs).mean(0) / T).masked_fill(oh, -1e9).log_softmax(1)


def class_count_adjusted_hardness(h, y, weight):
    """Hypothesis: deleting more examples of a class makes its images harder.

    Counts do not establish that an entire class is removed. This changes only
    the assignment of target difficulty; final rank matching keeps its marginal.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("class-count weight must be finite and nonnegative")
    if weight == 0:
        return h
    log_counts = torch.bincount(y).float()[y].log()
    spread = log_counts.std(unbiased=False)
    if spread <= 1e-8:
        return h
    return nscore(h) - weight * (log_counts-log_counts.mean()) / spread


def target_v2(z_exact, probes, y, mu, s, T=1.5, class_count_weight=0.0):
    h = class_count_adjusted_hardness(hardness_v2(z_exact, probes, y), y, class_count_weight)
    return compose(mu + s * nscore(h), rdist_v2(z_exact, probes, y, T), y)


def hardness_v3(z_exact, probes, y, extra):
    """v2 + LOO-kNN true-class votes at the two penultimate blocks + weight-noise log-odds (mirror: spearman .61 -> ~.69)."""
    allp = torch.stack([split(probes[k], y)[0] for k in probes]).mean(0)
    oh = F.one_hot(y, z_exact.shape[1]).bool()
    parts = [nscore(allp), nscore(split(z_exact, y)[0]), nscore(extra["knn_a"][oh] + 1e-6 * torch.rand(len(y))), nscore(extra["knn_b"][oh] + 1e-6 * torch.rand(len(y))), nscore(extra["wnoise"])]
    w = torch.tensor([1.0, 1.0, 0.5, 0.5, 1.0]); return (torch.stack(parts) * w[:, None]).sum(0) / w.sum()


def target_v3(z_exact, probes, y, extra, mu, s, T=1.5, sib=0.0, sib_mat=None):
    logr = rdist_v2(z_exact, probes, y, T)
    if sib > 0 and sib_mat is not None:   # boost classes the model itself treats as siblings of y
        oh = F.one_hot(y, z_exact.shape[1]).bool(); logr = (logr + sib * sib_mat[y]).masked_fill(oh, -1e9).log_softmax(1)
    return compose(mu + s * nscore(hardness_v3(z_exact, probes, y, extra)), logr, y)


def sibling_matrix(z_exact, y, C=100, T=4.0, top=4):
    """[C, C] 0/1 matrix: for each class present, its `top` nearest classes by mean dark knowledge of the original model."""
    M = torch.zeros(C, C)
    for c in y.unique():
        p = (z_exact[y == c] / T).softmax(1).mean(0); p[c] = 0; M[c, p.topk(top).indices] = 1.0
    return M
