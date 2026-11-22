"""Numerical and protocol primitives shared by training and local tests."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_torch(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def d4(x, rotation, flip=False):
    return torch.rot90(x.flip(-1) if flip else x, rotation, (-2, -1))


def undo_d4(x, rotation, flip=False):
    x = torch.rot90(x, -rotation, (-2, -1))
    return x.flip(-1) if flip else x


def majority(support):
    """Competition convention: ties (8/16 pixels) are positive."""
    return (F.avg_pool2d(support[:, None].float(), 4)[:, 0] >= .5).float()


def lovasz(logits, target):
    z, y = logits.float().flatten(1), target.float().flatten(1)
    errors, order = (1 - z * (2 * y - 1)).sort(dim=1, descending=True)
    labels = y.gather(1, order)
    positives = labels.sum(1, keepdim=True)
    intersection = positives - labels.cumsum(1)
    union = positives + (1 - labels).cumsum(1)
    jaccard = 1 - intersection / union.clamp_min(1)
    gradient = torch.cat((jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]), 1)
    return (errors.relu() * gradient).sum(1).mean()


def loss_fn(logits, target):
    """Per-image class-balanced BCE + symmetric Lovasz, entirely FP32."""
    with torch.autocast(logits.device.type, enabled=False):
        z, y = logits.float().flatten(1), target.float().flatten(1)
        bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        pos, neg = y.sum(1), (1 - y).sum(1)
        cb = ((bce * y).sum(1) / pos.clamp_min(1) +
              (bce * (1 - y)).sum(1) / neg.clamp_min(1))
        cb = cb / ((pos > 0).float() + (neg > 0).float())
        return .5 * cb.mean() + .25 * (lovasz(z, y) + lovasz(-z, 1 - y))


def training_loss(logits, target, recipe="class_balanced_lovasz"):
    if recipe == "class_balanced_lovasz":
        return loss_fn(logits, target)
    if recipe == "bce_dice":
        z, y = logits.float(), target.float()
        probability = z.sigmoid().flatten(1)
        labels = y.flatten(1)
        dice = 1 - ((2 * (probability * labels).sum(1) + 1) /
                    (probability.sum(1) + labels.sum(1) + 1)).mean()
        return .5 * F.binary_cross_entropy_with_logits(z, y) + .5 * dice
    raise ValueError(f"Unknown loss recipe: {recipe}")


def verify_split(train, validation, expected=(113, 57)):
    tr, va = list(map(str, train)), list(map(str, validation))
    if len(tr) != len(set(tr)) or len(va) != len(set(va)) or set(tr) & set(va):
        raise ValueError("Duplicate names or training/validation leakage")
    if expected and (len(tr), len(va)) != expected:
        raise ValueError(f"Expected split {expected}, got {(len(tr), len(va))}")
    return tr, va


def aligned(archive, names, key):
    stored = list(map(str, archive["names"]))
    if len(stored) != len(set(stored)) or len(names) != len(set(names)):
        raise ValueError("Duplicate prediction names")
    lookup = {n: i for i, n in enumerate(stored)}
    missing = set(names) - set(lookup)
    if missing:
        raise ValueError(f"Missing {len(missing)} named predictions")
    return archive[key][[lookup[n] for n in names]]


def cosine(start, end, fraction):
    return end + .5 * (start - end) * (1 + math.cos(math.pi * min(1., max(0., fraction))))


def gate_decision(score, synthetic_score):
    if score >= .75:
        return "pass"
    if synthetic_score >= .90 and score < .72:
        return "stop_transfer_gap"
    return "stop_inconclusive"


def budget_steps(remaining, seconds_per_step, evaluation_seconds, interval, cap):
    """Reserve final evaluation and checkpoint I/O; include periodic evaluation cost."""
    usable = remaining - evaluation_seconds - 30
    return max(0, min(cap, int(usable / (seconds_per_step + evaluation_seconds / interval))))


@torch.no_grad()
def update_ema(ema, model, decay=.995):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)
