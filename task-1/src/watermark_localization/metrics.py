"""Metric from the supplied brief; compare downsampling with the official kit."""
import numpy as np


def binary_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim < 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected binary masks with at least two dimensions")
    return mask.astype(bool)


def downsample_mask(mask):
    """512x512 -> 128x128, including ties (8 of 16 pixels) as watermarked."""
    mask = binary_mask(mask)
    if mask.shape[-2:] != (512, 512):
        raise ValueError("Expected trailing dimensions (512, 512)")
    blocks = mask.reshape(*mask.shape[:-2], 128, 4, 128, 4)
    return (blocks.sum(axis=(-3, -1)) >= 8).astype(np.uint8)


def mean_iou(prediction, target):
    """Average both class IoUs per image, then average images equally."""
    p, g = binary_mask(prediction), binary_mask(target)
    if p.shape != g.shape or p.ndim not in (2, 3) or p.size == 0:
        raise ValueError("Expected matching nonempty [H,W] or [N,H,W] masks")
    scores = []
    for c in (False, True):
        intersection = ((p == c) & (g == c)).sum(axis=(-2, -1))
        union = ((p == c) | (g == c)).sum(axis=(-2, -1))
        scores.append(np.divide(intersection, union, out=np.ones_like(union, dtype=float), where=union != 0))
    return float(np.mean(scores))
