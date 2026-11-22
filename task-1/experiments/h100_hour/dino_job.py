"""H3: cross-family DINOv2-LoRA edit localizer with two mode heads.

hole head   : trained on full_watermark=True images, target = clean region (1 - y)
partial head: trained on full_watermark=False images, target = watermarked region (y)
--mode fold : train on all real images except the held-out names, save held-out maps at --eval_steps
--mode refit: train on all 510 real images, save test maps at the last step
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PAD_LO, PAD_HI = 8, 12  # 512 -> 532 = 38 patches of 14 px; 532/4 = 133 cells, crop 2:130 -> 128


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__(); self.base = base; self.scale = alpha / r
        self.a = nn.Parameter(torch.empty(r, base.in_features)); self.b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale


def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2 if s == 1 else 0), nn.GroupNorm(8, o), nn.GELU())


class Net(nn.Module):
    def __init__(self, arch, layers, r, nout=2, hires=False):
        super().__init__(); self.hires = hires
        self.vit = timm.create_model(arch, pretrained=True, dynamic_img_size=True, num_classes=0)
        for p in self.vit.parameters(): p.requires_grad_(False)
        for blk in self.vit.blocks:
            blk.attn.qkv = LoRA(blk.attn.qkv, r, 2 * r); blk.attn.proj = LoRA(blk.attn.proj, r, 2 * r)
        self.layers = layers; d = self.vit.embed_dim
        self.lat = nn.ModuleList([cbr(d, 128, 1) for _ in layers])
        self.fuse = nn.Sequential(cbr(128 * len(layers), 256), cbr(256, 256))
        self.stem = nn.Sequential(cbr(3, 32), nn.Conv2d(32, 64, 4, 4), nn.GroupNorm(8, 64), nn.GELU(), cbr(64, 64))
        self.head = nn.Sequential(cbr(256 + 64, 128), cbr(128, 64), nn.Conv2d(64, nout, 1))

    def forward(self, x):  # x: float [B,3,512,512] in [0,1]
        xn = (x - MEAN.to(x)) / STD.to(x)
        if self.hires:  # 512 -> 896 = 64 patches of 14 px, aligned exactly with the 512 image
            xp = F.interpolate(xn, size=(896, 896), mode="bilinear", align_corners=False)
        else:
            xp = F.pad(xn, (PAD_LO, PAD_HI, PAD_LO, PAD_HI), mode="reflect")
        feats = self.vit.forward_intermediates(xp, indices=self.layers, output_fmt="NCHW", intermediates_only=True)
        f = torch.cat([l(t) for l, t in zip(self.lat, feats)], 1)
        f = self.fuse(f)
        if self.hires: f = F.interpolate(f, size=(128, 128), mode="bilinear", align_corners=False)
        else: f = F.interpolate(f, size=(133, 133), mode="bilinear", align_corners=False)[:, :, 2:130, 2:130]
        return self.head(torch.cat([f, self.stem(xn)], 1)).float()  # [B,nout,128,128]: (hole, partial) x families


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum(); inter = gts - gt_sorted.cumsum(0); union = gts + (1 - gt_sorted).cumsum(0)
    j = 1.0 - inter / union
    if len(gt_sorted) > 1: j[1:] = j[1:] - j[:-1]
    return j


def lovasz_hinge(logits, labels):  # per image mean; logits/labels [B,N]
    out = []
    for z, y in zip(logits, labels):
        signs = 2 * y - 1; err = 1 - z * signs
        e, perm = torch.sort(err, descending=True); g = lovasz_grad(y[perm])
        out.append(torch.dot(F.relu(e), g))
    return torch.stack(out).mean()


def loss_fn(z, y):  # z,y: [B,N]
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    pos, neg = y.sum(1).clamp(min=1), (1 - y).sum(1).clamp(min=1)
    cb = 0.5 * ((bce * y).sum(1) / pos + (bce * (1 - y)).sum(1) / neg)
    sym = 0.5 * lovasz_hinge(z, y) + 0.5 * lovasz_hinge(-z, 1 - y)
    return 0.5 * cb.mean() + 0.5 * sym


def d4(x, k, flip):  # x [...,H,W]
    if flip: x = x.flip(-1)
    return torch.rot90(x, k, (-2, -1))


def d4_inv(x, k, flip):
    x = torch.rot90(x, -k, (-2, -1))
    return x.flip(-1) if flip else x


FAMS = ["pixelseal", "trustmark", "stegastamp"]


def pick(z, fam):  # z [B,C,H,W], fam LongTensor [B] or None -> [B,2,H,W]
    if z.shape[1] == 2 or fam is None: return z[:, :2]
    idx = torch.stack([2 * fam, 2 * fam + 1], 1)[:, :, None, None].expand(-1, -1, *z.shape[2:])
    return torch.gather(z, 1, idx)


@torch.no_grad()
def predict(net, imgs, bs=16, fam=None):
    net.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs].cuda().permute(0, 3, 1, 2).float() / 255; acc = 0
        for k in range(4):
            for flip in (False, True):
                with torch.autocast("cuda", torch.bfloat16):
                    z = net(d4(x, k, flip))
                z = pick(z, None if fam is None else fam[i:i + bs].cuda())
                acc = acc + torch.sigmoid(d4_inv(z, k, flip))
        out.append((acc / 8).cpu())
    net.train(); return torch.cat(out).numpy()  # [n,2,128,128]


def auc(score, label):
    from numpy import argsort
    s = score.ravel(); l = label.ravel().astype(bool); npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0: return float("nan")
    r = np.empty(len(s)); r[argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    return float((r[l].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def evaluate(maps, gt, full):
    hole = [auc(maps[i, 0], 1 - gt[i]) for i in range(len(gt)) if full[i]]
    part = [auc(maps[i, 1], gt[i]) for i in range(len(gt)) if not full[i]]
    ciou = []
    for i in range(len(gt)):
        if full[i]:
            p = maps[i, 0] > 0.5; c = gt[i] == 0
            ciou.append((p & c).sum() / max((p | c).sum(), 1))
    return {"hole_auc": float(np.nanmean(hole)), "partial_auc": float(np.nanmean(part)), "hole_clean_iou05": float(np.mean(ciou))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/packed.npz"); ap.add_argument("--test", default="/root/packed_test.npz")
    ap.add_argument("--heldout", default="/root/job/heldout.json"); ap.add_argument("--mode", choices=["fold", "refit"], required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, nargs="*", default=[300, 1000]); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--arch", default="vit_large_patch14_reg4_dinov2.lvd142m"); ap.add_argument("--layers", type=int, nargs="*", default=[5, 11, 17, 23])
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--lr_lora", type=float, default=2e-4); ap.add_argument("--lr_dec", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--family_cond", action="store_true"); ap.add_argument("--hires", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
    d = np.load(a.data); names = list(d["names"]); imgs = torch.from_numpy(d["images"]); gt = d["mask128"].astype(np.float32); full = d["full"].astype(bool)
    held = set(json.load(open(a.heldout))) if a.mode == "fold" else set()
    tr = np.array([n not in held for n in names]); te = ~tr
    print(json.dumps({"mode": a.mode, "train": int(tr.sum()), "heldout": int(te.sum()), "full_train": int(full[tr].sum())}), flush=True)
    famid = torch.tensor([FAMS.index(m) for m in d["method"]]) if a.family_cond else None
    net = Net(a.arch, a.layers, a.r, 6 if a.family_cond else 2, a.hires).cuda()
    lora = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("vit.")]
    dec = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("vit.")]
    print("trainable lora/dec", sum(p.numel() for p in lora), sum(p.numel() for p in dec), flush=True)
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr_lora}, {"params": dec, "lr": a.lr_dec}], weight_decay=1e-2)
    warm = 50; sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    tri = np.where(tr)[0]; gimgs = imgs.cuda(); ggt = torch.from_numpy(gt).cuda(); gfull = torch.from_numpy(full).cuda()
    rng = np.random.default_rng(a.seed); t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        idx = torch.from_numpy(rng.choice(tri, a.bs, replace=False)).cuda()
        x = gimgs[idx].permute(0, 3, 1, 2).float() / 255; y = ggt[idx]; fw = gfull[idx]
        k = int(rng.integers(4)); flip = bool(rng.integers(2)); x, y = d4(x, k, flip), d4(y, k, flip)
        with torch.autocast("cuda", torch.bfloat16):
            z = net(x)
        z = pick(z, None if famid is None else famid[idx.cpu()].cuda())
        zsel = torch.where(fw[:, None, None], z[:, 0], z[:, 1]).flatten(1)
        ysel = torch.where(fw[:, None, None], 1 - y, y).flatten(1)
        loss = loss_fn(zsel, ysel)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        hist.append(float(loss))
        if step % 25 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(np.mean(hist[-25:])), "sec": round(time.time() - t0, 1),
                              "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
        if a.smoke and step >= 3: break
        if a.mode == "fold" and step in a.eval_steps:
            te_idx = np.where(te)[0]; maps = predict(net, imgs[te_idx], fam=None if famid is None else famid[te_idx])
            ev = evaluate(maps, gt[te_idx], full[te_idx]); ev["step"] = step; ev["sec"] = round(time.time() - t0, 1)
            print("EVAL", json.dumps(ev), flush=True)
            np.savez_compressed(out / f"heldout-{step}.npz", names=np.array(names)[te_idx], maps=maps.astype(np.float16))
            (out / f"eval-{step}.json").write_text(json.dumps(ev))
    if a.mode == "refit" or a.smoke:
        keep = {n for n, p in net.named_parameters() if p.requires_grad}
        torch.save({k: v for k, v in net.state_dict().items() if k in keep or not k.startswith("vit.")}, out / "trainable.pt")
        dt = np.load(a.test); tf = torch.tensor([FAMS.index(m) for m in dt["method"]]) if a.family_cond else None
        lim = 16 if a.smoke else None
        maps = predict(net, torch.from_numpy(dt["images"][:lim]), fam=None if tf is None else tf[:lim])
        np.savez_compressed(out / "test.npz", names=dt["names"][: len(maps)], maps=maps.astype(np.float16))
        print("TEST_DONE", maps.shape, round(time.time() - t0, 1), flush=True)
    (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
