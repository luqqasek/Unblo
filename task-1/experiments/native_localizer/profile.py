"""First allocated GPU run: tensor/gradient checks, throughput, memory, fidelity.

Random RGB is ONLY for throughput. Signal fidelity requires --encoded and uses
true clean/marked pairs. Never interpret random-input loss as task learning.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .common import atomic_json, rgb_tensor, seed_all, sha256
from .generator import EncodedDataset
from .losses import segmentation
from .model import NativeLocalizer


def delta_stats(a, b):
    d = a.float()-b.float()
    return {"rms": float(d.square().mean().sqrt()), "max_abs": float(d.abs().max()),
            "zero_fraction": float((d == 0).float().mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--encoded", type=Path)
    p.add_argument("--amp", action="store_true", help="Benchmark deeper AMP; first projections and local branch stay FP32")
    p.add_argument("--gpus", type=int, default=1)
    a = p.parse_args()
    seed_all(42); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = NativeLocalizer(backbone_path=a.backbone, checkpointing=True, amp=a.amp).to(a.device)
    if a.gpus > torch.cuda.device_count():
        raise ValueError("Requested more GPUs than visible in this single-node allocation")
    forward_model = torch.nn.DataParallel(model, device_ids=list(range(a.gpus))) if a.gpus > 1 else model
    optimizer = torch.optim.AdamW(model.parameter_groups())
    image = torch.rand(a.microbatch, 3, 512, 512, device=a.device)
    target = torch.randint(0,2,(a.microbatch,128,128),device=a.device).float()
    timings = []
    for i in range(a.steps):
        start = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        out = forward_model(image); loss = segmentation(out["logits"], target)
        loss.backward()
        if i == 0:
            gradients = {f"stage{j}": float(next(stage.parameters()).grad.norm())
                         for j, stage in enumerate(model.encoder.stages)}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if image.is_cuda:
            torch.cuda.synchronize()
        timings.append(time.monotonic()-start)
        if (i+1) % 10 == 0:
            print(json.dumps({"profile_microsteps": i+1, "seconds_mean": float(np.mean(timings[10:] or timings))}), flush=True)
    # Use an unmodified pretrained model for all precision comparisons.
    del optimizer, forward_model, model
    if image.is_cuda:
        torch.cuda.empty_cache()
    model = NativeLocalizer(backbone_path=a.backbone, checkpointing=False).to(a.device).eval()
    fidelity = {"status": "not_checked_without_true_pairs", "amp_enabled_for_training": False}
    if a.encoded:
        dataset = EncodedDataset(a.encoded)
        clean, marked, row = dataset.load(0)
        pair = torch.stack([rgb_tensor(clean), rgb_tensor(marked)]).to(a.device)
        with torch.inference_mode():
            fp32 = model(pair, return_features=True)
            normalized = (pair-model.mean)/model.std
            model.amp = True
            mixed = model(pair, return_features=True)
        fidelity = {
            "realization_id": row["realization_id"], "source_uint8_delta": row["residual_uint8"],
            "rgb_fp32_delta": delta_stats(pair[0], pair[1]),
            "normalization_fp32_delta": delta_stats(normalized[0], normalized[1]),
            "main_first_projection_delta": delta_stats(fp32["first"][0], fp32["first"][1]),
            "local_stem_delta": delta_stats(fp32["local"][0], fp32["local"][1]),
            "first_projection_amp_vs_fp32": delta_stats(fp32["first"], mixed["first"]),
            "local_amp_vs_fp32": delta_stats(fp32["local"], mixed["local"]),
            "logits_amp_vs_fp32": delta_stats(fp32["logits"], mixed["logits"]),
            "paired_logit_delta_fp32": delta_stats(fp32["logits"][0], fp32["logits"][1]),
            "paired_logit_delta_amp": delta_stats(mixed["logits"][0], mixed["logits"][1]),
            "amp_enabled_for_training": False,
            "note": "One diagnostic pair is insufficient for an AMP approval decision; inspect multiple regimes/families",
        }
    result = {
        "gpu_count_visible": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if image.is_cuda else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if image.is_cuda else None,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if image.is_cuda else None,
        "microbatch": a.microbatch, "microsteps": a.steps, "warmup_excluded": min(10,a.steps),
        "seconds_per_microstep": float(np.mean(timings[10:] or timings)),
        "estimated_effective_batch16_seconds": float(np.mean(timings[10:] or timings))*16/a.microbatch,
        "gradients_first_backward": gradients,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256(a.backbone), "fidelity": fidelity,
        "is_learning_result": False, "gpus_used": a.gpus,
        "precision": "deeper BF16, FP32 input projections/local branch" if a.amp else "full FP32; TF32 disabled",
    }
    atomic_json(a.out, result); print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
