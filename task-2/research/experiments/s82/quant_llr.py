"""Does a locally rebuilt quantized copy of the base explain an endpoint's text better than the plain base?

For each endpoint: summed log-likelihood of its answers under the plain base and under each rebuilt copy
(weights-only emulation), each at its best temperature. gain > 0 means the quantized copy fits better.

  HF_HOME=external/hf testbed/.venv/bin/python research/experiments/s82/quant_llr.py MODEL OUT ANSWERS --prefix ref- --endpoints a,b
        [--kinds fp8t,fp8m,int8r,nf4,fp4] [--limit N] [--device cpu|mps] [--date "21 Nov 2026"]
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("answers", nargs="+")
ap.add_argument("--prefix", default="ep-"); ap.add_argument("--endpoints", default=None)
ap.add_argument("--kinds", default="fp8t,fp8m,int8r,nf4,fp4"); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--ids-prefix", default="open-,f2,k4,k6,m,recon,d,t-"); ap.add_argument("--device", default="cpu")
ap.add_argument("--revision", default=None); ap.add_argument("--batch", type=int, default=1); ap.add_argument("--half", action="store_true"); ap.add_argument("--date", default="21 Nov 2026")
args = ap.parse_args()
TEMPS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1]
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
                    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
FP4 = torch.tensor([0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0]); FP4 = torch.cat([FP4, -FP4[1:]]) / 12.0


def e4m3(w, scale):
    return (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float32) * scale


def block4(w, code, block=64):
    flat = w.reshape(-1)
    pad = (-flat.numel()) % block
    x = torch.cat([flat, flat.new_zeros(pad)]).view(-1, block)
    absmax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = ((x / absmax).unsqueeze(-1) - code.to(x)).abs().argmin(-1)
    return (code.to(x)[idx] * absmax).reshape(-1)[:flat.numel()].view_as(w)


def quantize(named, kind):
    """named: {module name: original weight}. Returns {name: quantized weight}."""
    out = {}
    if kind == "fp8t":
        for n, w in named.items():
            out[n] = e4m3(w, w.abs().max() / 448)
    elif kind == "fp8m":                       # vLLM stores q,k,v as one tensor and gate,up as one tensor: one scale each
        groups = defaultdict(list)
        for n in named:
            stem, leaf = n.rsplit(".", 1)
            key = stem + (".qkv" if leaf in ("q_proj", "k_proj", "v_proj") else ".gateup" if leaf in ("gate_proj", "up_proj") else "." + leaf)
            groups[key].append(n)
        for key, names in groups.items():
            s = max(float(named[n].abs().max()) for n in names) / 448
            for n in names:
                out[n] = e4m3(named[n], s)
    elif kind == "int8r":
        for n, w in named.items():
            s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            out[n] = torch.round(w / s * 127) * s / 127
    elif kind in ("nf4", "fp4"):
        code = NF4 if kind == "nf4" else FP4
        for n, w in named.items():
            out[n] = block4(w, code)
    return out


tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=(torch.float16 if args.half else torch.float32)).to(args.device).eval()
lin = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in
       ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
orig = {n: m.weight.detach().to("cpu", copy=True).float() for n, m in lin.items()}
print(f"{len(lin)} linear layers to quantize", flush=True)

want = set(args.endpoints.split(",")) if args.endpoints else None
idp = tuple(args.ids_prefix.split(","))
rows, per = [], defaultdict(int)
for path in args.answers:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e = r["endpoint"]
        if not e.startswith(args.prefix) or (want and e not in want) or not (r.get("response") or "").strip() or not r["id"].startswith(idp):
            continue
        if args.limit and per[e] >= args.limit:
            continue
        per[e] += 1; rows.append(r)
enc = []
for r in rows:
    try:
        head = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False, date_string=args.date)
    except TypeError:
        head = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False)
    h = tok(head, return_tensors="pt", add_special_tokens=False).input_ids
    a = tok(r["response"], return_tensors="pt", add_special_tokens=False).input_ids[:, :256]
    if a.shape[1] >= 2:
        enc.append((r["endpoint"], torch.cat([h, a], 1), h.shape[1], a[0]))
print(f"{len(enc)} answers, kinds {args.kinds}", flush=True)
tot = {}
kinds = ["base"] + args.kinds.split(",")
t0 = time.time()
with torch.no_grad():
    for kind in kinds:
        if kind != "base":
            tq = time.time()
            q = quantize(orig, kind)                       # on the CPU; the GPU only ever holds one copy of the model
            num = den = 0.0
            for n, m in lin.items():
                num += float((q[n] - orig[n]).norm()); den += float(orig[n].norm())
                m.weight.copy_(q[n].to(args.device, dtype=m.weight.dtype)); q[n] = None
            del q
            print(f"{kind}: relative weight change {num / den:.4f} (rebuilt in {time.time() - tq:.0f} s)", flush=True)
        acc = defaultdict(lambda: [0.0] * len(TEMPS)); ntok = defaultdict(int)
        order = sorted(range(len(enc)), key=lambda i: enc[i][1].shape[1])
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for c0 in range(0, len(order), args.batch):
            chunk = [enc[i] for i in order[c0:c0 + args.batch]]
            L = max(x[1].shape[1] for x in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for bi, (_, x, _, _) in enumerate(chunk):
                ids[bi, :x.shape[1]] = x[0]; mask[bi, :x.shape[1]] = 1
            out = model(input_ids=ids.to(args.device), attention_mask=mask.to(args.device)).logits
            for bi, (e, x, start, a) in enumerate(chunk):
                logits = out[bi, start - 1:start - 1 + len(a)].float()
                tgt = a.to(args.device).view(-1, 1)
                for i, T_ in enumerate(TEMPS):
                    z = logits / T_
                    acc[e][i] += float((z.gather(-1, tgt).squeeze(-1) - torch.logsumexp(z, -1)).sum())
                ntok[e] += len(a)
            del out
            if args.device == 'mps':
                torch.mps.empty_cache()
        tot[kind] = {e: v for e, v in acc.items()}
        print(f"{kind} done at {time.time() - t0:.0f} s", flush=True)
        if kind != "base":
            print("   " + "  ".join(f"{e}: {max(tot[kind][e]) - max(tot['base'][e]):+.1f} ({ntok[e]} tok)" for e in sorted(tot[kind])), flush=True)
        if kind != "base":
            for n, m in lin.items():
                m.weight.copy_(orig[n].to(args.device, dtype=m.weight.dtype))
res = {}
print(f"\n{'endpoint':8s} {'tokens':>6s} {'T':>5s}  gain of each rebuilt copy over the plain base (nats; each at its best temperature)")
for e in sorted(tot["base"]):
    b = tot["base"][e]; ib = max(range(len(TEMPS)), key=lambda i: b[i])
    g = {k: max(tot[k][e]) - max(b) for k in kinds[1:]}
    gfix = {k: tot[k][e][ib] - b[ib] for k in kinds[1:]}
    res[e] = {"tokens": ntok[e], "T": TEMPS[ib], "gain": g, "gain_fixedT": gfix}
    print(f"{e:8s} {ntok[e]:6d} {TEMPS[ib]:5.2f}  " + "  ".join(f"{k}: {g[k]:+7.1f}" for k in kinds[1:]), flush=True)
json.dump(res, open(args.out, "w"), indent=1)
print("QLLR-DONE", flush=True)
