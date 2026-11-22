"""Diagnostics against original-model pseudo-labels and transformed forget images.

These measurements do not establish real retain accuracy, gold-model agreement,
or the hidden utility gate. E30 uses all forget images (holdout=0).
"""
import torch, torch.nn.functional as F
from common import *
import aug
from unlearn_core import kl_t

MILD = ["hflip", "shift8", "crop90", "rot5", "bright", "sat", "blur1", "noise.05", "jpeg75", "combo"]


def split_holdout(y, n_hold=500, seed=11):
    g = torch.Generator().manual_seed(seed); N = len(y); hold = torch.zeros(N, dtype=torch.bool)
    frac = n_hold / N
    for c in y.unique():
        idx = (y == c).nonzero().squeeze(1); k = max(1, int(round(len(idx) * frac)))
        hold[idx[torch.randperm(len(idx), generator=g)[:k]]] = True
    return (~hold).nonzero().squeeze(1), hold.nonzero().squeeze(1)


class RealEval:
    def __init__(self, X, y, zo, teacher, ia, ib, n_sub=300, seed=7):
        self.X, self.y, self.zo, self.ia, self.ib = X, y, zo, ia, ib
        g = torch.Generator().manual_seed(seed)
        self.sub = ia[torch.randperm(len(ia), generator=g)[:n_sub]]
        S = aug.suite(X[self.sub], seed=seed); self.suite = {k: S[k] for k in MILD}
        XB = X[ib]
        self.B = {"exact": XB, "mild": aug.mild_view(XB, g), "med0": aug.medium_view(XB, g), "med1": aug.medium_view(XB, g),
                  "strong": aug.strong_view(XB, g, None), "con0.7": aug._to_u8(aug.color(aug._to_f(XB), c=0.7))} if len(ib) else {}
        self.B_t = {k: predict(teacher, v, bs=50) for k, v in self.B.items()}
        XA = X[self.sub]
        self.Aanch = {"A-med": aug.medium_view(XA, g), "A-strong": aug.strong_view(XA, g, None)}
        self.Aanch_t = {k: predict(teacher, v, bs=50) for k, v in self.Aanch.items()}

    def __call__(self, student, tgt, log=print):
        y, ia = self.y, self.ia; out = {}
        zf = predict(student, self.X[ia], bs=50); yy = y[ia]; tt = tgt[ia]
        L = F.cross_entropy(zf, yy, reduction="none"); Lt = F.cross_entropy(tt, yy, reduction="none")
        out["exact"] = dict(Lmean=L.mean().item(), Lstd=L.std().item(), acc=(zf.argmax(1) == yy).float().mean().item(), kl_tgt=kl_t(tt, zf, 2.0).mean().item(),
                            kl_tgt_orig=kl_t(tt, self.zo[ia], 2.0).mean().item())
        log(f"  A exact : L {L.mean():.3f}/{L.std():.3f} (target {Lt.mean():.3f}/{Lt.std():.3f}) acc {out['exact']['acc']:.3f} | KL2(tgt||sub) {out['exact']['kl_tgt']:.4f} vs KL2(tgt||orig) {out['exact']['kl_tgt_orig']:.3f} -> fit {1-out['exact']['kl_tgt']/out['exact']['kl_tgt_orig']:.3f}")
        ys, ts = y[self.sub], tgt[self.sub]; Ls, Ks, K0, accs = [], [], [], []
        for k, v in self.suite.items():
            z = predict(student, v, bs=50); l = F.cross_entropy(z, ys, reduction="none")
            Ls.append((l.mean().item(), l.std().item())); Ks.append(kl_t(ts, z, 2.0).mean().item()); accs.append((z.argmax(1) == ys).float().mean().item())
        log("  A altered (held-out alteration types) L: " + " ".join(f"{k}:{a:.2f}/{b:.2f}" for k, (a, b) in zip(self.suite, Ls)))
        out["alt"] = dict(Lmean=sum(a for a, _ in Ls) / len(Ls), Lstd=sum(b for _, b in Ls) / len(Ls), kl_tgt=sum(Ks) / len(Ks), acc=sum(accs) / len(accs), per={k: v for k, v in zip(self.suite, Ls)})
        log(f"  A altered avg: L {out['alt']['Lmean']:.3f}/{out['alt']['Lstd']:.3f} acc {out['alt']['acc']:.3f} KL2(tgt||sub) {out['alt']['kl_tgt']:.4f}")
        dr = {}
        for name, views, tz in [("B", self.B, self.B_t), ("A", self.Aanch, self.Aanch_t)]:
            for k, v in views.items():
                z = predict(student, v, bs=50); t = tz[k]; key = k if name == "A" else f"B-{k}"
                yb = y[self.ib] if name == "B" else ys
                dm = (z.gather(1, yb[:, None]) - t.gather(1, yb[:, None])).mean().item()
                dr[key] = dict(kl=kl_t(t, z, 2.0).mean().item(), agree=(z.argmax(1) == t.argmax(1)).float().mean().item(),
                               acc_o=(t.argmax(1) == yb).float().mean().item(), acc_s=(z.argmax(1) == yb).float().mean().item(), dlogit=dm)
        log("  drift KL2(orig||sub) [agree | acc orig->sub | d true-logit]: ")
        for k, v in dr.items(): log(f"      {k:9s} KL {v['kl']:.4f}  agree {v['agree']:.3f}  acc {v['acc_o']:.3f}->{v['acc_s']:.3f}  dlogit {v['dlogit']:+.2f}")
        out["drift"] = dr
        return out


    def light(self, student, tgt, log=print, tag=""):
        """Cheap trade-off snapshot: fit on a subset of A (exact + 1 mild view) vs drift on held-out B (med view, exact)."""
        ys, ts = self.y[self.sub], tgt[self.sub]
        z = predict(student, self.X[self.sub], bs=50); l = F.cross_entropy(z, ys, reduction="none")
        fit = 1 - kl_t(ts, z, 2.0).mean().item() / kl_t(ts, self.zo[self.sub], 2.0).mean().item()
        za = predict(student, self.suite["combo"], bs=50); la = F.cross_entropy(za, ys, reduction="none")
        fit_a = 1 - kl_t(ts, za, 2.0).mean().item() / kl_t(ts, self.zo[self.sub], 2.0).mean().item()
        msg = f"  [{tag}] A: L {l.mean():.3f}/{l.std():.3f} fit {fit:.3f} | combo L {la.mean():.3f}/{la.std():.3f} fit {fit_a:.3f}"
        res = dict(L=l.mean().item(), Lstd=l.std().item(), fit=fit, fit_alt=fit_a)
        for k in ["med0", "exact", "strong"]:
            if k in self.B:
                zb = predict(student, self.B[k], bs=50); t = self.B_t[k]; yb = self.y[self.ib]
                kl = kl_t(t, zb, 2.0).mean().item(); dm = (zb.gather(1, yb[:, None]) - t.gather(1, yb[:, None])).mean().item()
                msg += f" | B-{k} KL {kl:.4f} acc {(t.argmax(1)==yb).float().mean():.3f}->{(zb.argmax(1)==yb).float().mean():.3f} dz {dm:+.2f}"
                res[f"B-{k}"] = kl
        log(msg); return res
