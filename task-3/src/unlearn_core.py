"""Model-agnostic retain-free unlearning core.

Given: frozen teacher (orig), student (copy of orig), forget images (uint8 NHWC), per-sample target distributions q* [N,C]
(as log-probabilities at T=1, i.e. target logits), the student is distilled toward q* on exact + slightly altered forget
images, while anchored to the teacher on memorisation-breaking views (strong alterations / cutmix) and in weight space.
"""
import copy, math, time
import torch, torch.nn.functional as F
import aug
from targets import mix_altered_wrong_classes

torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


def _ac(device):   # bf16 autocast on CUDA (weights stay fp32); no-op elsewhere
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(str(device) == "cuda"))


def kl_t(target_logits, student_logits, T):
    p = (target_logits / T).log_softmax(1); q = (student_logits / T).log_softmax(1)
    return (p.exp() * (p - q)).sum(1)


def run_unlearn(student, teacher, xf, tgt_logits, prep, cfg, device, log=print, eval_fn=None):
    """cfg keys: steps, bs, lr, wd_to_orig, lam_anchor, anchor ('strong'|'none'), train ('all'|'last<k>'|'head'),
    T_list (e.g. [1,2]), p_exact, sample_w (optional [N] weights), anchor_bs, tgt_alt (optional fn), warmup."""
    g = torch.Generator().manual_seed(cfg.get("seed", 0)); N = xf.shape[0]
    params = select_params(student, cfg.get("train", "all"))
    for p in student.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    orig = [p.detach().clone() for p in params]
    head_guard = None
    if cfg.get("freeze_absent_head", False):
        from model_protection import HeadRowGuard
        head_guard = HeadRowGuard(student.head, cfg["labels"].unique())
        cfg["frozen_head_classes"] = head_guard.rows.cpu().tolist()
        log(f"Protecting {len(head_guard.rows)} absent pseudo-class head rows")
    ewc_F = fisher_diag(student, xf, prep, device, params, g, n_batches=cfg.get("ewc_batches", 150), kind=cfg.get("ewc_views", "med+strong+mix")) if cfg.get("lam_ewc", 0) > 0 else None
    if cfg.get("opt", "adamw") == "sgd": opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9))
    else: opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0, betas=(0.9, 0.99))
    steps, bs = cfg["steps"], cfg["bs"]; abs_ = cfg.get("anchor_bs", bs); Ts = cfg.get("T_list", [1.0, 2.0])
    warm = cfg.get("warmup", max(1, steps // 20)); t0 = time.time(); perm = torch.randperm(N, generator=g); ptr = 0
    # Staggered release: sample i switches from "hold at the original logits" to "move to its target" at step rel[i].
    # At any time the already-released and the still-held images of a class pull in opposite directions on a class-level
    # shift, so the only consistent solution is an image-specific change.
    rf = cfg.get("release_frac", 0.0); lam_hold = cfg.get("lam_hold", 2.0); hold_logits = cfg.get("hold_logits", None)
    if rf > 0:
        rel = torch.empty(N); yy = cfg["labels"]
        for c in yy.unique():
            idx = (yy == c).nonzero().squeeze(1); k = len(idx)
            rel[idx[torch.randperm(k, generator=g)]] = (torch.arange(k).float() + torch.rand(k, generator=g)) / k
        rel = (rel * rf * steps).long()
    else: rel = torch.zeros(N, dtype=torch.long)
    bank = None
    if cfg.get("anchor_bank_fraction", 0) > 0:
        from anchor_bank import AnchorBank
        bank = AnchorBank(cfg["anchor_bank"], seed=cfg.get("seed", 0) + 123456)
        log(f"Synthetic anchor bank: {len(bank.images)} images, {len(bank.classes)} classes")
    student.train(False)  # eval mode: no dropout/droppath; BN (mirror stem) stays frozen
    for s in range(steps):
        lr = cfg["lr"] * ((s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))) if cfg.get("cosine", True) else cfg["lr"]
        for gp in opt.param_groups: gp["lr"] = lr
        if ptr + bs > N: perm = torch.randperm(N, generator=g); ptr = 0
        ib = perm[ptr:ptr + bs]; ptr += bs
        xb = xf[ib].to(device); ne = int(round(bs * cfg.get("p_exact", 0.5)))
        if ne < bs: xb = torch.cat([xb[:ne], aug.mild_view(xb[ne:], g, smin=cfg.get("smin", 0.8))])
        if cfg.get("broad_views", False) and ne < bs and s % 2 == 0:
            xb[ne:] = aug.medium_view(xf[ib[ne:]].to(device), g)
        zt = tgt_logits[ib].to(device)
        if cfg.get("alt_shift", 0) > 0 and ne < bs:   # unseen images get slightly harder when altered: lower the true-class logit on altered views
            ya = cfg["labels"][ib[ne:]].to(device); zt = zt.clone(); zt[torch.arange(ne, bs, device=device), ya] -= cfg["alt_shift"]
        if cfg.get("alt_rmix", 0) > 0 and ne < bs:
            with torch.no_grad(), _ac(device):
                view_teacher = teacher(prep(xb[ne:], device))
            zt = zt.clone()
            zt[ne:] = mix_altered_wrong_classes(
                zt[ne:], view_teacher.float(), cfg["labels"][ib[ne:]].to(device),
                weight=cfg["alt_rmix"], temperature=cfg.get("alt_rtemp", 1.5))
        with _ac(device): zs = student(prep(xb, device))
        zs = zs.float()
        lf = sum(kl_t(zt, zs, T) * (T * T if cfg.get("t2_scale", False) else 1.0) for T in Ts) / len(Ts)
        if cfg.get("lam_delta", 0) > 0:
            yb = cfg["labels"][ib].to(device); ohb = F.one_hot(yb, zs.shape[1]).bool()
            d_s = zs[ohb] - zs.masked_fill(ohb, -1e9).logsumexp(1); d_t = zt[ohb] - zt.masked_fill(ohb, -1e9).logsumexp(1)
            lf = lf + cfg["lam_delta"] * F.smooth_l1_loss(d_s, d_t, reduction="none")
        if cfg.get("sala_ratio", 0) > 0:
            sigma=cfg["sala_sigma"][ib].to(device)
            epsilon=cfg["sala_eps"][ib].to(device)
            error=d_s-d_t-cfg.get("sala_offset",0)*epsilon*sigma
            # Cap inverse-variance weights: at most four-fold spread.
            weights=sigma.square().reciprocal().clamp(.25,1.)
            alignment=weights*error.square()
            if "sala_lambda" not in cfg:
                gb=torch.autograd.grad(lf.mean(),params,retain_graph=True)
                ga=torch.autograd.grad(alignment.mean(),params,retain_graph=True)
                nb=sum(v.float().square().sum() for v in gb).sqrt()
                na=sum(v.float().square().sum() for v in ga).sqrt()
                cfg["sala_lambda"]=float((cfg["sala_ratio"]*nb/(na+1e-12)).clamp(max=1.))
                del gb,ga
                log(f"SALA lambda {cfg['sala_lambda']:.6g}; initial gradient ratio {cfg['sala_ratio']}")
            lf=lf+cfg["sala_lambda"]*alignment
        if "sample_w" in cfg: lf = lf * cfg["sample_w"][ib].to(device)
        released = (rel[ib] <= s).to(device); lh = torch.zeros((), device=device)
        if rf > 0 and (~released).any():
            zh = hold_logits[ib].to(device); dz = (zs - zs.mean(1, keepdim=True)) - (zh - zh.mean(1, keepdim=True))
            lh_i = (dz ** 2).sum(1) if cfg.get("hold_kind", "mse") == "mse" else torch.nn.functional.smooth_l1_loss(dz, torch.zeros_like(dz), reduction="none").sum(1)
            lh = (lh_i * (~released).float()).sum() / bs
            lf = lf * released.float()
        loss = lf.sum() / bs + lam_hold * lh; la = torch.zeros((), device=device)
        forget_loss = loss
        if cfg.get("anchor", "strong") != "none" and cfg.get("lam_anchor", 0) > 0:
            ia = torch.randint(0, N, (abs_,), generator=g); io = torch.randint(0, N, (abs_,), generator=g)
            xa = make_anchor(xf, ia, io, g, cfg, device)
            if bank is not None:
                nbank = int(round(len(xa) * cfg["anchor_bank_fraction"]))
                if nbank:
                    xa = xa.clone()
                    xa[-nbank:] = bank.sample(nbank).to(device)
            xin = prep(xa, device)
            lam_feat = cfg.get("lam_feat", 0.0)
            if lam_feat > 0:
                if cfg.get("feat_kind", "all") == "cls":
                    from model_protection import forward_with_cls, normalized_cls_loss
                    with torch.no_grad(), _ac(device): za_t, ft = forward_with_cls(teacher, xin)
                    with _ac(device): za, fs = forward_with_cls(student, xin)
                    lfeat = normalized_cls_loss(fs, ft)
                else:
                    with torch.no_grad(), _ac(device): za_t, ft = fwd_feats(teacher, xin)
                    with _ac(device): za, fs = fwd_feats(student, xin)
                    lfeat = sum(((a.float() - b.float()) ** 2).mean() / (b.float().var() + 1e-6) for a, b in zip(fs, ft)) / len(ft)
                loss = loss + lam_feat * lfeat
            else:
                with torch.no_grad(), _ac(device): za_t = teacher(xin)
                with _ac(device): za = student(xin)
            za_t, za = za_t.float(), za.float()
            la = sum(kl_t(za_t, za, T) for T in cfg.get("T_anchor", [1.0, 2.0])).mean() / len(cfg.get("T_anchor", [1.0, 2.0]))
            loss = loss + cfg["lam_anchor"] * la
        if ewc_F is not None:
            loss = loss + cfg["lam_ewc"] * sum((f * (p - o) ** 2).sum() for f, p, o in zip(ewc_F, params, orig))
        if cfg.get("wd_to_orig", 0) > 0:
            l2 = sum(((p - o) ** 2).sum() for p, o in zip(params, orig)); loss = loss + cfg["wd_to_orig"] * l2
        opt.zero_grad(set_to_none=True)
        if cfg.get("pcgrad", False):
            gf = torch.autograd.grad(forget_loss, params, retain_graph=True)
            gp = torch.autograd.grad(loss - forget_loss, params)
            for p, f, a in zip(params, gf, gp):
                dot = (f * a).sum()
                p.grad = f - dot.clamp(max=0) / (a.square().sum() + 1e-12) * a + a
        else:
            loss.backward()
        if head_guard is not None: head_guard.mask_gradients()
        if cfg.get("clip", 1.0): torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 1.0))
        opt.step()
        if head_guard is not None: head_guard.restore()
        if (s + 1) % cfg.get("log_every", 100) == 0 or s == steps - 1:
            log(f"    step {s+1}/{steps} lf {lf.mean().item():.4f} lh {float(lh.detach()):.4f} la {la.item():.4f} released {(rel <= s).float().mean().item():.2f} lr {lr:.2e} t {time.time()-t0:.0f}s")
        if eval_fn is not None and (s + 1) % cfg.get("eval_every", 10**9) == 0 and s + 1 < steps: eval_fn(student)
    if head_guard is not None:
        assert head_guard.verified(), "Protected head rows changed"
        log("Protected head rows verified bitwise unchanged")
    for p in student.parameters(): p.requires_grad_(False)
    return student


def fwd_feats(m, x):
    """timm ViT forward returning logits and every block's token output."""
    h = m.patch_embed(x); h = m._pos_embed(h); h = m.patch_drop(h); h = m.norm_pre(h); feats = []
    for blk in m.blocks:
        h = blk(h); feats.append(h)
    return m.forward_head(m.norm(h)), feats


def fisher_diag(model, xf, prep, device, params, g, n_batches=150, bs=16, kind="med+strong+mix", T=1.0):
    """Diagonal Fisher of the model's own predictive distribution on surrogate (memorisation-breaking) views."""
    F_ = [torch.zeros_like(p) for p in params]; N = xf.shape[0]
    for p in model.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    for _ in range(n_batches):
        ia = torch.randint(0, N, (bs,), generator=g); io = torch.randint(0, N, (bs,), generator=g)
        xa = make_anchor(xf, ia, io, g, {"anchor": kind}, device)
        with _ac(device): z = model(prep(xa, device))
        z = z.float() / T; ys = torch.multinomial(z.softmax(1).detach().cpu(), 1, generator=g).squeeze(1).to(device)
        gr = torch.autograd.grad(F.cross_entropy(z, ys), params)
        for f, g_ in zip(F_, gr): f += g_.detach() ** 2
    tot = sum(f.sum() for f in F_) / sum(f.numel() for f in F_)
    return [f / tot for f in F_]   # normalised to mean 1


def make_anchor(xf, ia, io, g, cfg, device="cpu"):
    kind = cfg.get("anchor", "strong"); xa, xo = xf[ia].to(device), xf[io].to(device); n = len(ia)
    def mix(a, b):
        lam = (torch.rand(len(a), generator=g).view(-1, 1, 1, 1) * 0.6 + 0.2).to(a.device)
        return (a.float() * lam + b.float() * (1 - lam)).round().clamp(0, 255).to(torch.uint8)
    if kind == "strong": return aug.strong_view(xa, g, other=xo if cfg.get("cutmix", True) else None)
    if kind == "mix": return mix(xa, xo)
    if kind == "med": return aug.medium_view(xa, g)
    if kind == "strong+mix":
        h = n // 2; return torch.cat([aug.strong_view(xa[:h], g, other=xo[:h]), mix(xa[h:], xo[h:])])
    if kind == "med+strong+mix":
        h = n // 3; return torch.cat([aug.medium_view(xa[:h], g), aug.strong_view(xa[h:2 * h], g, other=xo[h:2 * h]), mix(xa[2 * h:], xo[2 * h:])])
    raise ValueError(kind)


def select_params(model, spec):
    named = list(model.named_parameters())
    if spec == "all": return [p for _, p in named]
    if spec == "head": return [p for n, p in named if n.startswith("head")]
    if spec.startswith("last"):
        k = int(spec[4:]); depth = len(model.blocks); keep = tuple(f"blocks.{i}." for i in range(depth - k, depth))
        return [p for n, p in named if n.startswith(keep) or n.startswith("head") or n.startswith("norm.")]
    raise ValueError(spec)
