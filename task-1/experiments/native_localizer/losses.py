"""Per-image objectives, with explicit pair eligibility supplied by provenance."""
import torch
from torch.nn import functional as F


def segmentation(logits, target):
    z, y = logits.float(), target.float()
    p = z.sigmoid()
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none").mean((-2, -1))
    dice = []
    for pc, yc in ((p, y), (1-p, 1-y)):
        # epsilon makes absent-class/all-zero prediction well defined.
        dice.append(1 - (2*(pc*yc).sum((-2, -1)) + 1e-6) /
                    (pc.sum((-2, -1)) + yc.sum((-2, -1)) + 1e-6))
    return (.5*bce + .25*(dice[0]+dice[1])).mean()


def local_balanced(logits, target):
    y = target.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
    losses, present = [], []
    for c in (0, 1):
        mask = y == c
        n = mask.sum((-2, -1))
        losses.append((bce*mask).sum((-2, -1))/n.clamp_min(1))
        present.append(n > 0)
    return ((losses[0]+losses[1])/(present[0].float()+present[1].float())).mean()


def pair_loss(logits, target, pairs=(), consistency=None):
    """Pairs index corresponding views; consistency is an explicitly safe 128-grid mask.

    Never infer safe consistency merely from equal labels. Default is NO consistency.
    """
    z, y = logits.float(), target.float()
    terms = []
    for k, (a, b) in enumerate(pairs):
        changed = y[a] != y[b]
        parts = []
        if changed.any():
            direction = torch.sign(y[a]-y[b])
            parts.append(F.softplus(1-direction[changed]*(z[a]-z[b])[changed]).mean())
        if consistency is not None:
            safe = consistency[k].bool() & ~changed
            if safe.any():
                parts.append((z[a].sigmoid()-z[b].sigmoid()).square()[safe].mean())
        if parts:
            terms.append(torch.stack(parts).mean())
    return torch.stack(terms).mean() if terms else z.sum()*0


def objective(output, target, update, pairs=(), consistency=None, paired=True, local=True):
    seg = segmentation(output["logits"], target)
    loc = local_balanced(output["local_logits"], target)
    pair = pair_loss(output["logits"], target, pairs, consistency)
    weight = .1 * min(1., update / 2000)
    total = seg + (weight*loc if local else 0) + (weight*pair if paired else 0)
    return total, {"seg": seg.detach(), "local": loc.detach(), "pair": pair.detach(), "aux_weight": weight}
