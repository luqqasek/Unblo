"""Exact cached pixels; synthetic geometry never reads real masks."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from .core import d4, majority


class PairPool:
    def __init__(self, path, device="cpu"):
        path = Path(path)
        self.rows = json.loads((path / "rows.json").read_text())
        self.clean = torch.from_numpy(np.load(path / "clean.npy")).to(device)
        self.marked = torch.from_numpy(np.load(path / "marked.npy")).to(device)
        self.rectangles = torch.tensor([r["rectangle"] for r in self.rows], device=device)
        self.bits = torch.tensor([r["message"] for r in self.rows], dtype=torch.float32, device=device)
        self.side = self.clean.shape[-1]
        if self.clean.dtype != torch.uint8 or self.clean.shape != self.marked.shape:
            raise ValueError("Pair pool must contain matching uint8 tensors")
        full, partial = [], []
        for i, row in enumerate(self.rows):
            x0, y0, x1, y1 = row["rectangle"]
            if row["rectangle"] == [0, 0, self.side, self.side]:
                full.append(i)
            elif max((x1 - x0) / (y1 - y0), (y1 - y0) / (x1 - x0)) <= 2:
                partial.append(i)
        if not full or not partial:
            raise ValueError("Need both full-image and valid crop-only encoded pairs")
        self.full = torch.tensor(full, device=device)
        self.partial = torch.tensor(partial, device=device)
        self.device = torch.device(device)

    def sample(self, count, rng, augment=True):
        if count < 4 or count % 4:
            raise ValueError("Synthetic count must be a positive multiple of four")
        nb, nc = count // 2, count // 4
        # Exactly 50% clean holes, 25% exact local crops, 25% full-image remnants.
        fi = torch.randint(len(self.full), (count,), device=self.device, generator=rng)
        indices = self.full[fi]
        ci = torch.randint(len(self.partial), (nc,), device=self.device, generator=rng)
        indices[nb:nb + nc] = self.partial[ci]
        clean, marked = self.clean[indices], self.marked[indices]
        support = shapes(count, self.side, self.device, rng)
        holes = shapes(nb, self.side, self.device, rng)
        # A second connected shape, present in half the samples, gives at most two holes.
        second = shapes(nb, self.side, self.device, rng, area_scale=.5)
        add = torch.rand((nb, 1, 1), device=self.device, generator=rng) < .5
        holes |= second & add
        support[:nb] = ~holes
        rect = self.rectangles[indices[nb:nb + nc]]
        yy = torch.arange(self.side, device=self.device)[None, :, None]
        xx = torch.arange(self.side, device=self.device)[None, None, :]
        support[nb:nb + nc] = ((xx >= rect[:, 0, None, None]) & (xx < rect[:, 2, None, None]) &
                             (yy >= rect[:, 1, None, None]) & (yy < rect[:, 3, None, None]))
        rgb = torch.where(support[:, None], marked, clean)
        target = majority(support)
        # Use CPU RNG for augmentation controls: no CUDA scalar synchronization.
        if augment:
            rotation, flip = int(torch.randint(4, ())), bool(torch.randint(2, ()))
            rgb, target = d4(rgb, rotation, flip), d4(target, rotation, flip)
        return rgb.float().div_(255), target, self.bits[indices]


def shapes(count, side, device, generator, area_scale=1.):
    """Rectangles, ellipses and connected radial polygons; small areas oversampled."""
    def rand(*shape):
        return torch.rand(shape, device=device, generator=generator)
    small = rand(count) < .75
    lo = torch.where(small, .01, .005)
    hi = torch.where(small, .10, .30)  # second shape keeps total requested area <= .45
    area = (lo.log() + rand(count) * (hi.log() - lo.log())).exp() * area_scale
    kind = torch.randint(3, (count,), device=device, generator=generator)
    aspect = ((rand(count) * 2 - 1) * math.log(2)).exp()
    fill = torch.where(kind == 0, 1., math.pi / 4)
    width = (area * aspect / fill).sqrt().clamp(max=.95)
    height = (area / aspect / fill).sqrt().clamp(max=.95)
    cx = width / 2 + rand(count) * (1 - width)
    cy = height / 2 + rand(count) * (1 - height)
    grid = (torch.arange(side, device=device) + .5) / side
    x = (grid[None, None, :] - cx[:, None, None]) / (width[:, None, None] / 2)
    y = (grid[None, :, None] - cy[:, None, None]) / (height[:, None, None] / 2)
    rectangle = (x.abs() <= 1) & (y.abs() <= 1)
    radius = (x.square() + y.square()).sqrt()
    # A positive radial boundary is connected and cannot create isolated fragments.
    phase = rand(count, 1, 1) * (2 * math.pi)
    boundary = .87 + .13 * (5 * torch.atan2(y, x) + phase).cos()
    return torch.where(kind[:, None, None] == 0, rectangle,
                       radius <= torch.where(kind[:, None, None] == 1, 1., boundary))


class RealPool:
    def __init__(self, path, device="cpu"):
        with np.load(path) as d:
            self.names = list(map(str, d["names"]))
            self.images = torch.from_numpy(d["images"].copy()).to(device)
            self.targets = torch.from_numpy(d["targets"].astype(np.float32)).to(device)
            self.bits = torch.from_numpy(d["bits"].astype(np.float32)).to(device)
        self.device = torch.device(device)

    def sample(self, count, rng):
        i = torch.randint(len(self.names), (count,), device=self.device, generator=rng)
        rotation, flip = int(torch.randint(4, ())), bool(torch.randint(2, ()))
        x = d4(self.images[i], rotation, flip).float().div_(255)
        return x, d4(self.targets[i], rotation, flip), self.bits[i]
