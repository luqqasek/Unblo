"""Unlearning run on the real ViT-B/16. Usage: python run_real.py <name> key=value ..."""
import sys, json, time, torch, torch.nn.functional as F
from common import *
import aug, targets as TG
from unlearn_core import run_unlearn, kl_t
from real_eval import RealEval, split_holdout

name = sys.argv[1]; kv = dict(a.split("=", 1) for a in sys.argv[2:])
def get(k, d, t=float): return t(kv[k]) if k in kv else d
cfg = dict(steps=get("steps", 400, int), bs=get("bs", 16, int), anchor_bs=get("anchor_bs", 16, int), lr=get("lr", 2e-5), lam_anchor=get("lam_anchor", 1.0),
           anchor=get("anchor", "strong", str), train=get("train", "all", str), wd_to_orig=get("wd_to_orig", 0.0), p_exact=get("p_exact", 0.5),
           smin=get("smin", 0.8), seed=get("seed", 0, int), log_every=50, eval_every=get("eval_every", 100, int),
           opt=get("opt", "adamw", str), momentum=get("momentum", 0.9), T_list=[float(t) for t in get("T_list", "1,2", str).split(",")],
           lam_delta=get("lam_delta", 0.0), lam_feat=get("lam_feat", 0.0), alt_shift=get("alt_shift", 0.0), release_frac=get("release_frac", 0.0), lam_hold=get("lam_hold", 2.0), hold_kind=get("hold_kind", "mse", str), cosine=bool(get("cosine", 1, int)))
cfg["sala_ratio"] = get("sala_ratio", 0.0)
cfg["sala_offset"] = get("sala_offset", 0.0)
cfg["pcgrad"] = bool(get("pcgrad", 0, int))
cfg["broad_views"] = bool(get("broad_views", 0, int))
cfg["freeze_absent_head"] = bool(get("freeze_absent_head", 0, int))
cfg["feat_kind"] = get("feat_kind", "all", str)
if cfg["feat_kind"] not in ("all", "cls"):
    raise ValueError("feat_kind must be all or cls")
cfg["anchor_bank"] = get("anchor_bank", "", str)
cfg["anchor_bank_fraction"] = get("anchor_bank_fraction", 0.0)
if not 0 <= cfg["anchor_bank_fraction"] <= 1:
    raise ValueError("anchor_bank_fraction must lie in [0,1]")
cfg["alt_rmix"] = get("alt_rmix", 0.0)
cfg["alt_rtemp"] = get("alt_rtemp", 1.5)
if not 0 <= cfg["alt_rmix"] <= 1 or cfg["alt_rtemp"] <= 0:
    raise ValueError("Invalid altered-view wrong-class mixture configuration")
mu, s = get("mu", 2.5), get("s", 2.5); prior_w = get("prior_w", 0.5); Tr = get("Tr", 2.0)
hard_names = get("hard", "med", str); r_names = get("rviews", "med,strong", str)

X = load_forget(); zo = torch.load(OUT / "eda" / "orig_logits.pt"); y = zo.argmax(1)
P = torch.load(OUT / "eda" / "probe_logits.pt")
if "classes" in kv:   # quick experiments on a class range, e.g. classes=0-8
    lo, hi = [int(v) for v in kv["classes"].split("-")]; sel = ((y >= lo) & (y <= hi)).nonzero().squeeze(1)
    X, zo, y, P = X[sel], zo[sel], y[sel], {k: v[sel] for k, v in P.items()}; print(f"subset classes {lo}-{hi}: {len(y)} images", flush=True)
def expand(spec): return [k for k in P if any(k.startswith(p) for p in spec.split(","))]
FULL = None
if "cf" in kv:   # targets from cross-fitted held-out predictions (src/crossfit.py)
    tags = kv["cf"].split(","); zs_cf = [torch.load(OUT / "eda" / f"crossfit_{t}.pt")["logits"] for t in tags]
    oh_ = F.one_hot(y, 100).bool(); mode = get("cf_mode", "raw", str)
    # ensemble of cross-fit runs: average probabilities at T=2 (like the gold reference), back to logits
    pcf = torch.stack([(z_ / 2).softmax(1) for z_ in zs_cf]).mean(0); zcf = 2 * pcf.clamp_min(1e-9).log()
    d_cf, lr_cf = TG.split(zcf, y)
    if mode == "raw": tgt = get("cf_scale", 1.0) * (zcf - zcf.mean(1, keepdim=True))
    else:
        hsig = TG.nscore(d_cf) if mode == "rank" else (TG.nscore(d_cf) + get("cf_hw", 0.5) * TG.nscore(TG.hardness_v2(zo, P, y))) / (1 + get("cf_hw", 0.5))
        rw = get("cf_rw", 1.0)   # weight of the cross-fit wrong-class distribution vs the original model's dark knowledge
        lr_o = TG.rdist_v2(zo, P, y, get("Tr", 1.5))
        lr = (rw * (lr_cf * get("cf_rsharp", 1.0)).masked_fill(oh_, -1e9).log_softmax(1).exp() + (1 - rw) * lr_o.exp()).clamp_min(1e-12).log().masked_fill(oh_, -1e9).log_softmax(1)
        tgt = TG.compose(mu + s * TG.nscore(hsig), lr, y)
    print(f"cross-fit targets ({tags}, mode {mode}): cf acc {(zcf.argmax(1) == y).float().mean():.3f}", flush=True)
elif "full_delta" in kv:   # classes whose whole training set is in the forget set: removed at class level by a head-bias edit
    fd = float(kv["full_delta"]); cnt = torch.bincount(y, minlength=100); FULL = (cnt >= 129).nonzero().squeeze(1)
    zo = zo.clone(); zo[:, FULL] -= fd; P = {k: v.clone() for k, v in P.items()}
    for k in P: P[k][:, FULL] -= fd
    isf = torch.isin(y, FULL); pi = (~isf).nonzero().squeeze(1)
    tgt = zo.clone()                                   # fully-removed classes: hold the edited outputs
    Pp = {k: v[pi] for k, v in P.items()}
    tgt[pi] = TG.target_v2(zo[pi], Pp, y[pi], mu, s, get("Tr", 1.5))
    sharp = get("full_sharp", 1.0)
    if sharp != 1.0:                                   # optionally sharpen the sibling distribution of removed-class images
        fi = isf.nonzero().squeeze(1); d_f, lr_f = TG.split(zo[fi], y[fi]); tgt[fi] = TG.compose(d_f, (lr_f * sharp).masked_fill(F.one_hot(y[fi], 100).bool(), -1e9).log_softmax(1), y[fi])
    print(f"class-level removal of {FULL.tolist()} (delta {fd}); per-image targets for {len(pi)} partial-class images", flush=True)
elif get("tv", 2, int) == 3:
    extra = torch.load(OUT / "eda" / "sig_extra.pt")
    if "classes" in kv: extra = {k: v[sel] for k, v in extra.items()}
    tgt = TG.target_v3(zo, P, y, extra, mu, s, get("Tr", 1.25), sib=get("sib", 1.0), sib_mat=TG.sibling_matrix(zo, y))
elif get("tv", 2, int) == 2: tgt = TG.target_v2(zo, P, y, mu, s, get("Tr", 1.5), class_count_weight=get("class_count_weight", 0.0))
else:
    h = TG.hardness(zo, P, y, expand(hard_names))
    delta = TG.rank_match(h, TG.marginal_quantiles(len(y), mu, s, kind=get("marg", "logistic", str)))
    logr = TG.wrong_dist(zo, P, y, expand(r_names), T=Tr, prior_w=prior_w)
    tgt = TG.compose(delta, logr, y)
if "target_file" in kv:
    tgt = torch.load(kv["target_file"], weights_only=True)
    assert tgt.shape == zo.shape and torch.isfinite(tgt).all()
Lt = F.cross_entropy(tgt, y, reduction="none"); print(f"target: L {Lt.mean():.3f}/{Lt.std():.3f} acc {(tgt.argmax(1)==y).float().mean():.3f}", flush=True)

teacher = load_model(); student = load_model()
if FULL is not None:
    with torch.no_grad():
        for m_ in (teacher, student): m_.head.bias[FULL.to(DEVICE)] -= fd
n_hold = get("holdout", 500, int)
ia, ib = split_holdout(y, int(round(n_hold * len(y) / 3500))) if n_hold > 0 else (torch.arange(len(y)), torch.arange(0))
print(f"train on {len(ia)} forget images, hold out {len(ib)}", flush=True)
ev = RealEval(X, y, zo, teacher, ia, ib)
log = lambda m: print(m, flush=True)
t0 = time.time()
step_box = [0]
def light(st):
    step_box[0] += cfg["eval_every"]; ev.light(st, tgt, log=log, tag=f"step {step_box[0]}")
cfg["labels"] = y[ia]; cfg["hold_logits"] = zo[ia]
if cfg["sala_ratio"] > 0:
    names = [k for k in TG.MILDISH if k in P]
    ds = torch.stack([TG.split(P[k],y)[0] for k in names])
    med = ds.median(0).values
    sigma = (1.4826*(ds-med).abs().median(0).values).clamp(.5,3.)
    eps = torch.randn(len(y),generator=torch.Generator().manual_seed(902)).clamp(-2,2)
    cfg["sala_sigma"] = sigma[ia]
    cfg["sala_eps"] = eps[ia]

run_unlearn(student, teacher, X[ia], tgt[ia], preprocess, cfg, DEVICE, log=log, eval_fn=light if n_hold > 0 else None)
cfg.pop("labels"); cfg.pop("hold_logits"); cfg.pop("sala_sigma", None); cfg.pop("sala_eps", None)
print(f"[{name}] cfg={cfg} time {time.time()-t0:.0f}s")
res = ev(student, tgt, log=log)
if cfg["sala_ratio"] > 0:
    def sala_metrics(images, ids, shift=0):
        z = predict(student, images, bs=50)
        delta = TG.split(z,y[ids])[0]
        mu = TG.split(tgt[ids],y[ids])[0] - shift + cfg["sala_offset"]*eps[ids]*sigma[ids]
        score=(delta-mu)/sigma[ids]
        return dict(indist=float((score.abs()<2).float().mean()),mean_abs_z=float(score.abs().mean()),p95_abs_z=float(score.abs().quantile(.95)))
    res["sala_exact"] = sala_metrics(X,torch.arange(len(y)))
    ids=ev.sub
    res["sala_altered"] = sala_metrics(ev.suite["combo"],ids,cfg["alt_shift"])
    print("SALA proxy metrics",res["sala_exact"],res["sala_altered"],flush=True)
sd = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}
(OUT / "runs").mkdir(exist_ok=True)
if get("save", 0, int): torch.save(sd, OUT / "runs" / f"{name}.pt"); torch.save(tgt, OUT / "runs" / f"{name}_tgt.pt")
json.dump({"cfg": cfg, "kv": kv, "res": res}, open(OUT / "runs" / f"{name}.json", "w"), indent=1)
