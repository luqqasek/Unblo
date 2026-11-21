"""Resumable raw-RGB synthetic pretraining with exact exposure accounting."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .common import PROJECT, append_jsonl, atomic_json, restore_rng, rng_state, seed_all, sha256
from .evaluate import evaluate_synthetic
from .generator import EncodedDataset
from .losses import segmentation, local_balanced, pair_loss
from .model import NativeLocalizer


def save_checkpoint(path, model, ema, optimizer, scheduler, step, rng, cfg, seen_views, best):
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "update": step, "rng": rng_state(), "data_rng": rng.bit_generator.state,
                "config": cfg, "seen_views": seen_views, "best_dev_miou": best}, temp)
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PROJECT/"config.json")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--family", choices=["pixelseal", "trustmark", "stegastamp"], required=True)
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stop-after", type=int, help="Stop this invocation at this absolute update; schedule unchanged")
    p.add_argument("--microbatch", type=int)
    p.add_argument("--dev-limit", type=int, default=256)
    p.add_argument("--overfit", action="store_true", help="Same fixed debug batch every update; never a generalization result")
    p.add_argument("--no-pair", action="store_true")
    p.add_argument("--amp-validated", action="store_true", help="Enable deeper AMP only after saving fidelity checks")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--real-fold", type=int, choices=[1,2,3])
    p.add_argument("--real-data", type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    cfg.update(family=a.family, paired=not a.no_pair, overfit=a.overfit, gpus=a.gpus)
    if a.microbatch:
        cfg["microbatch"] = a.microbatch
    if cfg["microbatch"] % 2 or cfg["effective_batch"] % cfg["microbatch"]:
        raise ValueError("Use even microbatches dividing the effective batch; keep complete pairs together")
    if a.amp_validated:
        cfg["amp"] = True
    seed_all(cfg["seed"])
    torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", 4))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = EncodedDataset(a.train, a.family)
    if cfg.get("cache_rgb", False):
        print(json.dumps({"cached_encoded_RGB_pairs":train.preload(workers=min(8,torch.get_num_threads()))}),flush=True)
    if any(r["partition"] != "train" for r in train.rows):
        raise ValueError("Synthetic pretraining accepts only train source identities")
    dev = EncodedDataset(a.dev, a.family) if a.dev else None
    if dev:
        if any(r["partition"] != "dev" for r in dev.rows):
            raise ValueError("Development data must come from dev source identities")
        if {r["source_id"] for r in dev.rows} & {r["source_id"] for r in train.rows}:
            raise ValueError("Source leakage across synthetic train and dev")
        if {tuple(r["message"]) for r in dev.rows} & {tuple(r["message"]) for r in train.rows}:
            raise ValueError("Synthetic validation requires unseen messages as well as unseen covers")
    provenance = json.loads((PROJECT/"pretrained.json").read_text())
    if sha256(a.backbone) != provenance["sha256"]:
        raise ValueError("Backbone SHA256 does not match verified ImageNet-22K checkpoint")
    model = NativeLocalizer(backbone_path=a.backbone, checkpointing=cfg["checkpointing"], amp=cfg["amp"]).to(a.device)
    if a.gpus > torch.cuda.device_count() and str(a.device).startswith("cuda"):
        raise ValueError("Insufficient GPUs in current single-node allocation")
    forward_model = torch.nn.DataParallel(model, device_ids=list(range(a.gpus))) if a.gpus > 1 else model
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameter_groups(cfg["backbone_lr"], cfg["new_lr"], cfg["weight_decay"]))
    def factor(update):
        if update < cfg["warmup"]:
            return (update+1)/cfg["warmup"]
        progress = min(1., (update-cfg["warmup"])/max(1, cfg["steps"]-cfg["warmup"]))
        return .5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    rng = np.random.default_rng(cfg["seed"])
    start_step, seen_views, best = 0, 0, -1.
    if a.resume:
        state = torch.load(a.resume, map_location=a.device, weights_only=False)
        if state["config"] != cfg:
            raise ValueError("Resume configuration differs; use a separate experiment/control directory")
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"]); rng.bit_generator.state = state["data_rng"]
        start_step, seen_views, best = state["update"], state["seen_views"], state["best_dev_miou"]
    a.out.mkdir(parents=True, exist_ok=True)
    atomic_json(a.out/"config.json", cfg)
    atomic_json(a.out/"provenance.json", dict(provenance,
        train_manifest_sha256=sha256(a.train), dev_manifest_sha256=sha256(a.dev) if a.dev else None,
        train_distinct_sources=len({r["source_id"] for r in train.rows}),
        train_encoded_realizations=len(train.rows),
        actual_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        parameter_note="classification head incl. head norm removed",
        precision="raw/normalization/main stem/local FP32; deeper BF16 only if amp is true; losses FP32",
        consistency="disabled for main global branch: local 27px neighborhood does not cover its full receptive field",
    ))
    stop = [False]
    def request_stop(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    fixed_batch = train.batch(np.random.default_rng(42), cfg["effective_batch"], paired=cfg["paired"]) if a.overfit else None
    finish = min(cfg["steps"], a.stop_after or cfg["steps"])
    times = []
    model.train()
    for update in range(start_step, finish):
        start = time.monotonic()
        batch = fixed_batch or train.batch(rng, cfg["effective_batch"], update, cfg["paired"])
        images, target, pairs, _, meta = batch
        optimizer.zero_grad(set_to_none=True)
        sums = dict(seg=0., local=0., pair=0.)
        aux = .1*min(1., update/cfg.get("aux_ramp",2000))
        micro = cfg["microbatch"]; b = len(images)
        train_pred = []
        for offset in range(0, b, micro):
            x = images[offset:offset+micro].to(a.device)
            y = target[offset:offset+micro].to(a.device)
            output = forward_model(x)
            micro_pairs = [(i-offset, j-offset) for i, j in pairs if offset <= i and j < offset+micro]
            seg = segmentation(output["logits"], y)
            loc = local_balanced(output["local_logits"], y)
            pair = pair_loss(output["logits"], y, micro_pairs)
            # Each image and each PAIR gets equal weight independently of accumulation.
            loss = (seg+aux*loc)*len(x)/b + (aux*pair*len(micro_pairs)/max(1, len(pairs)) if cfg["paired"] else 0)
            loss.backward()
            sums["seg"] += float(seg.detach())*len(x)/b
            sums["local"] += float(loc.detach())*len(x)/b
            sums["pair"] += float(pair.detach())*len(micro_pairs)/max(1, len(pairs))
            train_pred.append(output["logits"].detach().sigmoid().cpu())
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite gradients; aborting before corrupting model")
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            ema_parameters = dict(ema.named_parameters())
            for name, value in model.named_parameters():
                ema_parameters[name].lerp_(value, 1-cfg["ema"])
        if str(a.device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.monotonic()-start)
        seen_views += b
        step = update+1
        append_jsonl(a.out/"view_recipes.jsonl", {"update": step, "views": meta,
                      "curriculum": "simple" if update < 5000 else "components",
                      "pairs": pairs})
        if step % 10 == 0 or step == start_step+1:
            from wmloc.utils import score
            probs = torch.cat(train_pred).numpy()
            record = {"update": step, "seen_views": seen_views, **sums, "aux_weight": aux,
                      "grad_norm_before_clip": float(gradient_norm), "seconds_step": times[-1],
                      "mean_seconds_step_last_100": float(np.mean(times[-100:])),
                      "train_batch_miou": score(probs >= .5, target.numpy()),
                      "effective_paired_views": len(pairs)*2, "batch_views": b,
                      "primary_clean": sum(m["primary"] and m["primary_kind"] == "clean" for m in meta),
                      "primary_marked": sum(m["primary"] and m["primary_kind"] == "marked" for m in meta),
                      "effective_clean": sum(m["fg_fraction"] == 0 for m in meta),
                      "effective_marked": sum(m["fg_fraction"] == 1 for m in meta),
                      "effective_fg_le_1pct": sum(0 < m["fg_fraction"] <= .01 for m in meta),
                      "backbone_lr": optimizer.param_groups[0]["lr"],
                      "max_cuda_memory_bytes": torch.cuda.max_memory_allocated() if str(a.device).startswith("cuda") else None}
            for low, high in ((0, .0025), (.0025, .01), (.01, .05), (.05, .2), (.2, .5)):
                record[f"effective_minority_{low}_{high}"] = sum(
                    low < min(m["fg_fraction"], 1-m["fg_fraction"]) <= high for m in meta)
            append_jsonl(a.out/"train.jsonl", record); print(json.dumps(record), flush=True)
        if dev and not a.overfit and (step % cfg["eval_every"] == 0 or step == finish):
            evaluated_model = model if cfg.get("evaluation_weights") == "raw" else ema
            result = evaluate_synthetic(evaluated_model, dev, a.device, limit=a.dev_limit, out=a.out/f"dev/{step:07d}")
            miou = result["groups"]["all"]["miou"]["mean"]
            append_jsonl(a.out/"dev.jsonl", {"update": step, "miou": miou})
            print(json.dumps({"update": step, "dev_miou": miou}), flush=True)
            if miou > best:
                best = miou
                save_checkpoint(a.out/"best.pt", model, ema, optimizer, scheduler, step, rng, cfg, seen_views, best)
        if step % cfg["checkpoint_every"] == 0 or step == finish or stop[0]:
            save_checkpoint(a.out/"last.pt", model, ema, optimizer, scheduler, step, rng, cfg, seen_views, best)
        if stop[0]:
            break
    atomic_json(a.out/"invocation_summary.json", {
        "last_update": step if finish > start_step else start_step, "seen_views": seen_views,
        "updates_this_invocation": len(times), "measured_step_seconds_mean": float(np.mean(times)) if times else None,
        "training_seconds_excluding_eval_and_save": float(sum(times)), "best_dev_miou": best if best >= 0 else None,
        "is_overfit_diagnostic": a.overfit,
    })
    if a.real_fold:
        from .real_eval import evaluate_real
        evaluate_real(model, a.real_data, a.family, a.real_fold, a.out/"real_validation", a.device)


if __name__ == "__main__":
    main()
