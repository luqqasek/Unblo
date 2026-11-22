"""White-box check (local test bed research): how well does text from an endpoint fit the known base model versus
the base with a known LoRA adapter at several strengths?

For every saved answer, the answer's tokens are scored under the base and under each adapter strength, at several
temperatures. Output: one JSON line per answer with the summed log-likelihood per (strength, temperature).

  HF_HOME=external/hf testbed/.venv/bin/python research/experiments/s82/wb_score.py FAMILY ADAPTER OUT ANSWERS... \
        [--prefix tb-] [--endpoints a,b] [--limit N] [--device mps|cpu] [--ids-prefix open-,m]
FAMILY: Q15 | Q30 (keys of testbed/endpoints.py BASES); ADAPTER: general | persona | code | latex | math
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from peft import PeftModel
from peft.helpers import rescale_adapter_scale
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "testbed"))
import endpoints as tb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("family"); ap.add_argument("adapter"); ap.add_argument("out"); ap.add_argument("answers", nargs="+")
ap.add_argument("--prefix", default="tb-"); ap.add_argument("--endpoints", default=None)
ap.add_argument("--limit", type=int, default=None, help="answers per endpoint")
ap.add_argument("--device", default=None); ap.add_argument("--ids-prefix", default=None)
ap.add_argument("--scales", default=None); ap.add_argument("--half", action="store_true")
ap.add_argument("--system", default=None, help="JSON file {endpoint: system prompt text} to use instead of the template default")
args = ap.parse_args()

SCALES = [0.0] + [float(x) for x in args.scales.split(",")] if args.scales else [0.0, 0.15, 0.3, 0.5, 0.7, 1.0, 1.5]
TEMPS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3]
EXTRA_BASES = {"L1B": {"model": "unsloth/Llama-3.2-1B-Instruct", "revision": None}}
EXTRA_ADAPTERS = {("L1B", "general"): {"adapter": "zjudai/flowertune-general-nlp-lora-llama-3.2-1b-instruct", "revision": "27007a6617d13213b7892fbcfb1e853e3c49a29e"}}
base = EXTRA_BASES.get(args.family) or tb.BASES[args.family]
adapter = EXTRA_ADAPTERS.get((args.family, args.adapter)) or tb.ADAPTERS[(args.family, args.adapter)]
device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(base["model"], revision=base["revision"])
model = AutoModelForCausalLM.from_pretrained(base["model"], revision=base["revision"], torch_dtype=(torch.float16 if args.half else torch.float32)).to(device)
model = PeftModel.from_pretrained(model, adapter["adapter"], revision=adapter["revision"]).to(device)
model.eval()
systems = json.loads(Path(args.system).read_text()) if args.system else {}

rows, per = [], {}
want = set(args.endpoints.split(",")) if args.endpoints else None
idp = tuple(args.ids_prefix.split(",")) if args.ids_prefix else None
for path in args.answers:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e = r["endpoint"]
        if not e.startswith(args.prefix) or (want and e not in want) or not (r.get("response") or "").strip():
            continue
        if idp and not r["id"].startswith(idp):
            continue
        if args.limit and per.get(e, 0) >= args.limit:
            continue
        per[e] = per.get(e, 0) + 1
        rows.append(r)
done = set()
out = Path(args.out)
if out.exists():
    done = {(j["endpoint"], j["id"], j["repeat"]) for j in map(json.loads, out.open())}
print(f"{len(rows)} answers to score ({len(done)} already done) on {device}; scales {SCALES}", flush=True)
t0 = time.time()
with torch.no_grad(), out.open("a", encoding="utf-8") as sink:
    for n, r in enumerate(rows, 1):
        key = (r["endpoint"], r["id"], r["repeat"])
        if key in done:
            continue
        msgs = [{"role": "user", "content": r["prompt"]}]
        if r["endpoint"] in systems:
            msgs = [{"role": "system", "content": systems[r["endpoint"]]}] + msgs
        head = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, date_string="21 Nov 2026")
        h = tok(head, return_tensors="pt", add_special_tokens=False).input_ids
        a = tok(r["response"], return_tensors="pt", add_special_tokens=False).input_ids[:, :256]
        if a.shape[1] < 2:
            continue
        ids = torch.cat([h, a], dim=1).to(device)
        start = h.shape[1]
        ll = {}
        tgt = a[0].to(device).view(-1, 1)
        for s in SCALES:
            if s == 0.0:
                with model.disable_adapter():
                    logits = model(ids).logits[0, start - 1:-1].float()
            else:
                with rescale_adapter_scale(model, s):
                    logits = model(ids).logits[0, start - 1:-1].float()
            vals = []
            for T_ in TEMPS:                                        # stay on the device: only one number per temperature leaves it
                z = logits / T_
                vals.append(float((z.gather(-1, tgt).squeeze(-1) - torch.logsumexp(z, dim=-1)).sum()))
            ll[str(s)] = [round(v, 4) for v in vals]
        sink.write(json.dumps({"endpoint": r["endpoint"], "id": r["id"], "repeat": r["repeat"], "tokens": int(a.shape[1]),
                               "scales": SCALES, "temps": TEMPS, "ll": ll}) + "\n")
        sink.flush()
        if n % 50 == 0:
            print(f"{n}/{len(rows)} in {time.time() - t0:.0f} s", flush=True)
print("WB-DONE", flush=True)
