"""Read each endpoint's decoding from its own text using the local base model (no adapter).

Per endpoint: fitted temperature (fine grid), and where the sampled tokens sit in the base distribution at that
temperature: share of tokens beyond the top-0.9 / 0.95 / 0.99 probability mass, and beyond rank 20 / 50. Plain sampling
puts about 10% / 5% / 1% of tokens beyond those masses; top-p or top-k sampling puts none there.
Also the standardized excess surprisal (text less typical of the base than its own samples would be).

  HF_HOME=external/hf testbed/.venv/bin/python research/experiments/s82/wb_decode.py FAMILY OUT.json ANSWERS --prefix ep- --endpoints a,b
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "testbed"))
import endpoints as tb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("family"); ap.add_argument("out"); ap.add_argument("answers", nargs="+")
ap.add_argument("--prefix", default="ep-"); ap.add_argument("--endpoints", default=None)
ap.add_argument("--ids-prefix", default="open-,f2,k4,k6,m,recon,d,t-"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--half", action="store_true"); ap.add_argument("--device", default=None)
args = ap.parse_args()
EXTRA_BASES = {"L1B": {"model": "unsloth/Llama-3.2-1B-Instruct", "revision": None}}
base = EXTRA_BASES.get(args.family) or tb.BASES[args.family]
device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(base["model"], revision=base["revision"])
model = AutoModelForCausalLM.from_pretrained(base["model"], revision=base["revision"], torch_dtype=(torch.float16 if args.half else torch.float32)).to(device).eval()
TEMPS = [round(0.4 + 0.05 * i, 2) for i in range(21)]          # 0.40 .. 1.40
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
print(f"{len(rows)} answers on {device}", flush=True)
acc = defaultdict(lambda: {"ll": [0.0] * len(TEMPS), "tok": 0, "logits": []})
tt = torch.tensor(TEMPS).view(-1, 1, 1)
with torch.no_grad():
    for n, r in enumerate(rows, 1):
        head = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False, date_string="21 Nov 2026")
        h = tok(head, return_tensors="pt", add_special_tokens=False).input_ids
        a = tok(r["response"], return_tensors="pt", add_special_tokens=False).input_ids[:, :256]
        if a.shape[1] < 2:
            continue
        ids = torch.cat([h, a], dim=1).to(device)
        logits = model(ids).logits[0, h.shape[1] - 1:-1].float().cpu()
        lp = torch.log_softmax(logits.unsqueeze(0) / tt, dim=-1).gather(-1, a[0].view(1, -1, 1).expand(len(TEMPS), -1, 1)).squeeze(-1)
        A = acc[r["endpoint"]]
        for i in range(len(TEMPS)):
            A["ll"][i] += float(lp[i].sum())
        A["tok"] += int(a.shape[1])
        A["logits"].append((logits, a[0].clone()))
        if n % 100 == 0:
            print(f"{n}/{len(rows)}", flush=True)
out = {}
for e, A in sorted(acc.items()):
    best = max(range(len(TEMPS)), key=lambda i: A["ll"][i])
    T = TEMPS[best]
    beyond = {0.9: 0, 0.95: 0, 0.99: 0}; r20 = r50 = n = 0; exc = var = 0.0
    for logits, a in A["logits"]:
        p = torch.softmax(logits / T, dim=-1)
        ptok = p.gather(-1, a.view(-1, 1)).squeeze(-1)
        above = (p * (p > ptok.view(-1, 1))).sum(-1)               # mass of tokens strictly more likely than the sampled one
        rank = (p > ptok.view(-1, 1)).sum(-1)
        for q in beyond:
            beyond[q] += int((above > q).sum())
        r20 += int((rank >= 20).sum()); r50 += int((rank >= 50).sum()); n += len(a)
        logp = torch.log(p.clamp_min(1e-30))
        H = -(p * logp).sum(-1)
        V = (p * logp ** 2).sum(-1) - H ** 2
        exc += float((-torch.log(ptok.clamp_min(1e-30)) - H).sum()); var += float(V.sum())
    out[e] = {"T": T, "tokens": n, "beyond_0.9": beyond[0.9] / n, "beyond_0.95": beyond[0.95] / n, "beyond_0.99": beyond[0.99] / n,
              "rank>=20": r20 / n, "rank>=50": r50 / n, "excess_z": exc / math.sqrt(var) if var else 0.0,
              "ll_per_tok": A["ll"][best] / A["tok"], "ll_curve": [round(x / A["tok"], 4) for x in A["ll"]]}
    o = out[e]
    print(f"{e} T {T:.2f} tokens {n:5d} beyond0.9 {o['beyond_0.9']:.3f} beyond0.95 {o['beyond_0.95']:.3f} beyond0.99 {o['beyond_0.99']:.4f} "
          f"rank>=20 {o['rank>=20']:.3f} rank>=50 {o['rank>=50']:.4f} excess z {o['excess_z']:+.1f} ll/tok {o['ll_per_tok']:.3f}", flush=True)
json.dump(out, open(args.out, "w"), indent=1)
print("DECODE-DONE", flush=True)
