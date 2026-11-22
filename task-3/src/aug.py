"""Image alterations on uint8 NHWC tensors (CPU). Each returns uint8 NHWC. Used for eval suites and training views."""
import io, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_f(u8):  # NHWC uint8 -> NCHW float [0,1]
    return u8.permute(0, 3, 1, 2).float() / 255.0


def _to_u8(x):
    return (x.clamp(0, 1) * 255.0 + 0.5).floor().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def affine(x, scale=None, tx=None, ty=None, rot=None, flip=None, pad="reflection"):
    """x NCHW float. scale<1 zooms in (crop), tx/ty in [-1,1] units, rot in degrees. Per-sample tensors or None."""
    n = x.shape[0]
    one, zero = torch.ones(n), torch.zeros(n)
    s = one if scale is None else torch.as_tensor(scale).float().expand(n)
    tx = zero if tx is None else torch.as_tensor(tx).float().expand(n)
    ty = zero if ty is None else torch.as_tensor(ty).float().expand(n)
    r = (zero if rot is None else torch.as_tensor(rot).float().expand(n)) * math.pi / 180
    f = one if flip is None else torch.where(torch.as_tensor(flip).bool().expand(n), -one, one)
    cos, sin = torch.cos(r) * s, torch.sin(r) * s
    theta = torch.stack([torch.stack([cos * f, -sin, tx], 1), torch.stack([sin * f, cos, ty], 1)], 1)
    grid = F.affine_grid(theta.to(x.device), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=pad, align_corners=False)


def color(x, b=None, c=None, s=None):
    n = x.shape[0]
    def v(a): return torch.as_tensor(a).float().expand(n).view(n, 1, 1, 1).to(x.device)
    if b is not None: x = x * v(b)
    if c is not None:
        m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * v(c) + m
    if s is not None:
        g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]); x = (x - g) * v(s) + g
    return x


def blur(x, sigma):
    k = int(2 * math.ceil(3 * sigma) + 1)
    ax = torch.arange(k).float() - k // 2
    w = torch.exp(-ax ** 2 / (2 * sigma ** 2)); w = (w / w.sum()).to(x.device)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def jpeg(u8, q):
    out = []
    for im in u8.numpy():
        buf = io.BytesIO(); Image.fromarray(im).save(buf, format="JPEG", quality=q)
        out.append(np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    return torch.from_numpy(np.stack(out))


def downup(x, size):
    full = x.shape[-1]
    return F.interpolate(F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False), size=full, mode="bilinear", align_corners=False)


def mild_view(u8, g, p_color=0.8, p_blur=0.15, p_noise=0.25, smin=0.8):
    """Random slight alteration (training view): crop/shift/flip/small rotation + colour jitter (+ sometimes blur/noise)."""
    n = u8.shape[0]; x = _to_f(u8)
    U = lambda a, b: torch.rand(n, generator=g) * (b - a) + a
    s = U(smin, 1.0); room = 1 - s
    x = affine(x, scale=s, tx=(U(-1, 1)) * (room + 0.06), ty=(U(-1, 1)) * (room + 0.06), rot=U(-10, 10) * (U(0, 1) < 0.5), flip=U(0, 1) < 0.5)
    cm = (U(0, 1) < p_color).float()
    x = color(x, b=1 + cm * U(-.2, .2), c=1 + cm * U(-.2, .2), s=1 + cm * U(-.3, .3))
    if torch.rand(1, generator=g).item() < p_blur and x.shape[-1] >= 64: x = blur(x, (float(torch.rand(1, generator=g)) * 0.8 + 0.4) * x.shape[-1] / 224)
    nm = (U(0, 1) < p_noise).float().view(n, 1, 1, 1)
    x = x + (nm * 0.04 * torch.rand(1, generator=g).item() * torch.randn(x.shape, generator=g)).to(x.device)
    return _to_u8(x)


def strong_view(u8, g, other=None):
    """Memorisation-breaking alteration (anchor view): small crop, heavy colour change, optional cutmix with another image."""
    n = u8.shape[0]; x = _to_f(u8)
    U = lambda a, b: torch.rand(n, generator=g) * (b - a) + a
    s = U(0.35, 0.7); room = 1 - s
    x = affine(x, scale=s, tx=U(-1, 1) * room, ty=U(-1, 1) * room, rot=U(-20, 20), flip=U(0, 1) < 0.5)
    gm = (U(0, 1) < 0.25).float()
    x = color(x, b=U(.6, 1.4), c=U(.5, 1.5), s=(1 - gm) * U(.3, 1.7))
    if other is not None:  # cutmix a random rectangle from another image
        H = x.shape[-1]; y = _to_f(other.to(x.device))
        for i in range(n):
            if torch.rand(1, generator=g).item() < 0.5:
                h = int(H * (0.3 + 0.4 * torch.rand(1, generator=g).item())); w = int(H * (0.3 + 0.4 * torch.rand(1, generator=g).item()))
                t = int(torch.randint(0, H - h + 1, (1,), generator=g)); l = int(torch.randint(0, H - w + 1, (1,), generator=g))
                x[i, :, t:t + h, l:l + w] = y[i, :, t:t + h, l:l + w]
    return _to_u8(x)


# ---- fixed eval suite of "slight alterations" (deterministic given seed) ----
def suite(u8, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = u8.shape[0]; x = _to_f(u8)
    U = lambda a, b: torch.rand(n, generator=g) * (b - a) + a
    out = {}
    out["hflip"] = _to_u8(affine(x, flip=torch.ones(n)))
    out["shift8"] = _to_u8(affine(x, tx=torch.sign(U(-1, 1)) * 16 / 224, ty=torch.sign(U(-1, 1)) * 16 / 224))  # 8px
    out["crop90"] = _to_u8(affine(x, scale=0.9, tx=U(-.1, .1), ty=U(-.1, .1)))
    out["crop75"] = _to_u8(affine(x, scale=0.75, tx=U(-.25, .25), ty=U(-.25, .25)))
    out["crop50"] = _to_u8(affine(x, scale=0.5, tx=U(-.5, .5), ty=U(-.5, .5)))
    out["zoomout"] = _to_u8(affine(x, scale=1.15))
    out["rot5"] = _to_u8(affine(x, rot=torch.sign(U(-1, 1)) * 5))
    out["rot15"] = _to_u8(affine(x, rot=torch.sign(U(-1, 1)) * 15))
    out["bright"] = _to_u8(color(x, b=torch.where(U(0, 1) > .5, 1.2, 0.8)))
    out["contrast"] = _to_u8(color(x, c=torch.where(U(0, 1) > .5, 1.3, 0.7)))
    out["sat"] = _to_u8(color(x, s=torch.where(U(0, 1) > .5, 1.4, 0.6)))
    out["gray"] = _to_u8(color(x, s=0.0))
    out["blur1"] = _to_u8(blur(x, 1.0))
    out["blur2"] = _to_u8(blur(x, 2.0))
    out["noise.02"] = _to_u8(x + 0.02 * torch.randn(x.shape, generator=g))
    out["noise.05"] = _to_u8(x + 0.05 * torch.randn(x.shape, generator=g))
    out["noise.10"] = _to_u8(x + 0.10 * torch.randn(x.shape, generator=g))
    out["jpeg75"] = jpeg(u8, 75)
    out["jpeg30"] = jpeg(u8, 30)
    out["down160"] = _to_u8(downup(x, 160))
    out["down112"] = _to_u8(downup(x, 112))
    out["combo"] = _to_u8(color(affine(x, scale=U(.8, 1.0), tx=U(-.08, .08), ty=U(-.08, .08), rot=U(-7, 7), flip=U(0, 1) > .5), b=U(.85, 1.15), c=U(.85, 1.15), s=U(.8, 1.2)))
    return out


def medium_view(u8, g):
    n = u8.shape[0]; x = _to_f(u8)
    U = lambda a, b: torch.rand(n, generator=g) * (b - a) + a
    s = U(0.55, 0.8); room = 1 - s
    x = affine(x, scale=s, tx=U(-1, 1) * room, ty=U(-1, 1) * room, rot=U(-12, 12), flip=U(0, 1) < 0.5)
    return _to_u8(color(x, b=U(.75, 1.25), c=U(.7, 1.3), s=U(.6, 1.4)))


def probe_bank(seed=0):
    """Named deterministic probe views: list of (name, fn(u8)->u8). Random ones are seeded per name."""
    def R(fn, k):
        def f(u8):
            g = torch.Generator().manual_seed(seed * 1000 + k); return fn(u8, g)
        return f
    bank = []
    for k in range(4): bank.append((f"mild{k}", R(mild_view, k)))
    for k in range(4): bank.append((f"med{k}", R(medium_view, 10 + k)))
    for k in range(4): bank.append((f"strong{k}", R(lambda u, g: strong_view(u, g, None), 20 + k)))
    bank += [
        ("gray", lambda u: _to_u8(color(_to_f(u), s=0.0))),
        ("con0.7", lambda u: _to_u8(color(_to_f(u), c=0.7))),
        ("con1.3", lambda u: _to_u8(color(_to_f(u), c=1.3))),
        ("con0.5", lambda u: _to_u8(color(_to_f(u), c=0.5))),
        ("bri0.7", lambda u: _to_u8(color(_to_f(u), b=0.7))),
        ("bri1.3", lambda u: _to_u8(color(_to_f(u), b=1.3))),
        ("blur1.5", lambda u: _to_u8(blur(_to_f(u), 1.5 * u.shape[1] / 224 if u.shape[1] >= 64 else 0.5))),
        ("blur2.5", lambda u: _to_u8(blur(_to_f(u), 2.5 * u.shape[1] / 224 if u.shape[1] >= 64 else 0.8))),
        ("crop70c", lambda u: _to_u8(affine(_to_f(u), scale=0.7))),
        ("crop50c", lambda u: _to_u8(affine(_to_f(u), scale=0.5))),
        ("zoomout1.3", lambda u: _to_u8(affine(_to_f(u), scale=1.3))),
        ("rot20", lambda u: _to_u8(affine(_to_f(u), rot=20.0))),
        ("noise.08", R(lambda u, g: _to_u8(_to_f(u) + 0.08 * torch.randn(_to_f(u).shape, generator=g)), 40)),
        ("noise.15", R(lambda u, g: _to_u8(_to_f(u) + 0.15 * torch.randn(_to_f(u).shape, generator=g)), 41)),
        ("down.5", lambda u: _to_u8(downup(_to_f(u), u.shape[1] // 2))),
        ("hflip", lambda u: _to_u8(affine(_to_f(u), flip=torch.ones(u.shape[0])))),
    ]
    return bank
