"""Local preflight or a wall-limited H100 run. Never rents GPUs or submits results."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .core import (aligned, budget_steps, cosine, d4, digest, gate_decision, loss_fn, save_torch,
                   undo_d4, update_ema, verify_split, write_json, training_loss)
from .data import PairPool, RealPool


def preflight(config, verify_hashes=True, families_to_check=("trustmark", "pixelseal")):
    problems, families = [], {}
    for family in families_to_check:
        folder = Path(config["packed"]) / family
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            problems.append(f"Missing packed manifest: {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest["family"] != family or manifest["source_overlap"] != 0:
            problems.append(f"Invalid source provenance for {family}")
        for relative, expected in manifest["files"].items():
            path = folder / relative
            if not path.is_file() or (verify_hashes and digest(path) != expected):
                problems.append(f"Missing or changed packed asset: {path}")
        split = json.loads((folder / "split.json").read_text())
        verify_split(split["train"], split["validation"])
        if family == "pixelseal":
            source = Path(config["pixelseal_split"])
            if not source.is_file():
                problems.append(f"Missing PixelSeal checkpoint split: {source}")
            else:
                original = json.loads(source.read_text())
                if set(original["train"]) != set(split["train"]) or set(original["validation"]) != set(split["validation"]):
                    problems.append("PixelSeal checkpoint split does not match prepared training/validation")
            checkpoint = Path(config["pixelseal_checkpoint"])
            if checkpoint.is_file() and digest(checkpoint) != manifest.get("initial_checkpoint_sha256"):
                problems.append("PixelSeal checkpoint differs from the verified baseline checkpoint")
        families[family] = {"train_pairs": manifest["train_pairs"], "dev_pairs": manifest["dev_pairs"],
                            "baseline_miou": manifest["maps"]["baseline_miou"]}
    required_keys = ["backbone"] if "trustmark" in families_to_check else []
    if "pixelseal" in families_to_check:
        required_keys += ["pixelseal_checkpoint", "pixelseal_split"]
    for key in required_keys:
        if not Path(config[key]).is_file():
            problems.append(f"Missing {key}: {config[key]}")
    official = ("videoseal/videoseal/__init__.py", "pixelseal_card.yaml", "pixelseal/checkpoint.pth") if "pixelseal" in families_to_check else ()
    for relative in official:
        if not (Path("checkpoints") / relative).is_file():
            problems.append(f"Missing official PixelSeal asset: checkpoints/{relative}")
    batch = config.get("batch_size", 32)
    if batch < 16 or batch % 4 or config.get("real_batch", 8) != 8:
        problems.append("Use batch size >=16 divisible by 4, with exactly 8 real views")
    if not 0 <= config.get("trustmark_blend", .5) <= 1:
        problems.append("Invalid TrustMark blend weight")
    if not (Path(config["packed"]) / "frozen_stegastamp.json").is_file():
        problems.append("Missing measured frozen StegaStamp recipe")
    if config.get("eval_interval", 1000) < 1 or config.get("eval_batch", 4) < 1:
        problems.append("Evaluation interval and batch size must be positive")
    if not torch.cuda.is_available():
        problems.append("CUDA unavailable; use the CPU smoke command for local execution")
    return {"ready": not problems, "problems": problems, "families": families,
            "cuda_available": torch.cuda.is_available(), "torch": torch.__version__,
            "gpu_training_executed": False}


@torch.inference_mode()
def predict(model, real, batch=4, tta=True):
    was_training = model.training
    model.eval()
    outputs = []
    views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    for i in range(0, len(real.names), batch):
        x = real.images[i:i + batch].float() / 255
        bits = real.bits[i:i + batch]
        total = None
        for k, f in views:
            p = undo_d4(model(d4(x, k, f).contiguous(), bits).float().sigmoid(), k, f)
            total = p if total is None else total + p
        outputs.append((total / len(views)).cpu().numpy())
    model.train(was_training)
    return np.concatenate(outputs)


def fuse(family, probability, maps, weight=.5):
    from experiments.h100_hour.h2_decode import expected_miou_decode
    if family == "trustmark":
        p = (1 - weight) * maps["base_prob"] + weight * probability
        return expected_miou_decode(p) | maps["corrections"].astype(np.uint8)
    route_b = maps["base_mask"].mean((1, 2)) >= .5
    # Keep the existing base+DINO partial prediction exactly on route A.
    candidate = expected_miou_decode(probability)
    return np.where(route_b[:, None, None], candidate, maps["current_mask"]).astype(np.uint8)


@torch.inference_mode()
def evaluate(model, valid, dev, maps, family, config):
    from experiments.h100_hour.h2_decode import miou, report
    started = time.monotonic()
    probability = predict(model, valid, config.get("eval_batch", 4))
    target = valid.targets.cpu().numpy()
    masks = fuse(family, probability, maps, config.get("trustmark_blend", .5))
    result = report("fixed_sprint_blend", maps["current_mask"], masks, target)
    rng = torch.Generator(device=dev.device).manual_seed(93017)
    synth = []
    was_training = model.training
    model.eval()
    # Fixed 32 deterministic synthetic views, unchanged across evaluations.
    for _ in range(8):
        x, y, bits = dev.sample(4, rng, augment=False)
        p = model(x, bits).sigmoid().cpu().numpy()
        synth.extend(miou(p >= .5, y.cpu().numpy()).tolist())
    model.train(was_training)
    result.update(synthetic_miou=float(np.mean(synth)), evaluation_seconds=time.monotonic() - started,
                  standalone_miou=float(miou(probability >= .5, target).mean()),
                  protocol="Reused 57-image development fold; fixed blend and D4 TTA; not an independent test")
    return result, probability, masks


def save_checkpoint(path, model, ema, optimizer, step, config, rng, target, history):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": config, "target_steps": target, "history": history,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
                "rng_sampler": rng.get_state()}, temporary)
    os.replace(temporary, path)


def train_family(family, config, output, deadline):
    from .models import build_model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the sprint; use smoke.py for local CPU verification")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.get("threads", 6))
    torch.manual_seed(config.get("seed", 42))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda:0"
    rng = torch.Generator(device=device).manual_seed(config.get("seed", 42))
    folder = Path(config["packed"]) / family
    train = PairPool(folder / "synthetic_train", device)
    dev = PairPool(folder / "synthetic_dev", device)
    real = RealPool(folder / "real_train.npz", device)
    valid = RealPool(folder / "real_valid.npz", device)
    verify_split(real.names, valid.names)
    with np.load(folder / "validation_maps.npz") as archive:
        maps = {k: aligned(archive, valid.names, k) for k in archive.files if k != "names"}
    model = build_model(family, config, device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.05)
    restore = config.get(f"{family}_resume")
    if restore:
        state = torch.load(restore, map_location=device, weights_only=False)
        source_state = state[config.get("resume_state_key", "ema")]
        model.load_state_dict(source_state, strict=True)
        ema.load_state_dict(source_state, strict=True)
        del state
    compiled = torch.compile(model, dynamic=False) if config.get("compile", True) else model
    target = config.get(f"{family}_steps", 7000 if family == "trustmark" else 5000)
    requested_target = target
    batch = config.get("batch_size", 32)
    interval = config.get("eval_interval", 1000)
    gate_at = started + min(420, max(0, deadline - started - 60))
    stop, gate, history = "step_limit", None, []
    evaluation_seconds = 30.
    benchmark, last_eval, best = [], 0, -1.
    record = {"family": family, "gpu": torch.cuda.get_device_name(), "requested_steps": target,
              "compile": config.get("compile", True), "start": started, "deadline": deadline,
              "precision": "FP32 input/filter/first projections; BF16 supported deeper layers; FP32 loss"}
    write_json(output / "config.json", {**config, **record})
    step = 0
    while step < target:
        now = time.monotonic()
        if now + evaluation_seconds + 30 >= deadline:
            stop = "wall_budget"
            break
        stage2 = family == "pixelseal" or config.get("mixed_from_start", False) or step >= max(20, int(target * 4500 / 7000))
        if family == "pixelseal":
            lr = cosine(1e-4, 2e-6, step / max(target - 1, 1))
        elif stage2:
            boundary = max(20, int(target * 4500 / 7000))
            lr = cosine(5e-5, 1e-6, (step - boundary) / max(target - boundary - 1, 1))
        else:
            lr = cosine(3e-4, 5e-5, step / max(int(target * 4500 / 7000) - 1, 1))
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.cuda.synchronize()
        tick = time.monotonic()
        x, y, bits = train.sample(batch - (8 if stage2 else 0), rng)
        if stage2:
            rx, ry, rb = real.sample(8, rng)
            x, y, bits = torch.cat((x, rx)), torch.cat((y, ry)), torch.cat((bits, rb))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = compiled(x.contiguous(), bits)
        loss = training_loss(z, y, config.get("loss", "class_balanced_lovasz"))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, model, config.get("ema_decay", .995))
        torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        step += 1
        if 10 < step <= 20:
            benchmark.append(seconds)
        if step == 1 or step % 25 == 0:
            row = {"step": step, "loss": float(loss), "gradient_norm": float(norm), "seconds": seconds,
                   "elapsed": time.monotonic() - started, "stage2": stage2, "lr": lr,
                   "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            print(json.dumps(row), flush=True)
            with open(output / "train.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        due_gate = family == "trustmark" and gate is None and time.monotonic() >= gate_at - evaluation_seconds - 10
        due_eval = step == 20 or step % interval == 0 or step == target or due_gate
        if due_eval:
            result, probability, masks = evaluate(ema, valid, dev, maps, family, config)
            evaluation_seconds = result["evaluation_seconds"]
            result.update(step=step, elapsed=time.monotonic() - started)
            history.append(result)
            last_eval = step
            write_json(output / f"evaluation-{step}.json", result)
            if due_gate:
                gate = gate_decision(result["cand"], result["synthetic_miou"])
                write_json(output / "minute7-gate.json", {"decision": gate, **result})
            if result["cand"] > best:
                best = result["cand"]
                save_torch(output / "selected.pt", {"ema": ema.state_dict(), "step": step, "family": family, "config": config})
                np.savez(output / "selected-validation.npz", names=valid.names, probabilities=probability, masks=masks, step=step)
            save_checkpoint(output / "resume.pt", model, ema, optimizer, step, config, rng, target, history)
            print(json.dumps({"evaluation": result}), flush=True)
            if step == 20:
                per_step = float(np.median(benchmark))
                target = step + budget_steps(deadline - time.monotonic(), per_step, evaluation_seconds, interval, requested_target - step)
                write_json(output / "benchmark.json", {"seconds_per_update": per_step, "measured_updates": len(benchmark),
                           "images_per_second": batch / per_step, "adapted_target": target,
                           "evaluation_seconds": evaluation_seconds, "startup_and_first20_seconds": time.monotonic() - started,
                           "includes_loss_backward_optimizer_ema_data": True})
            if due_gate:
                if gate != "pass" and config.get("hard_quality_gate", True):
                    stop = gate
                    break
    # Reuse the latest complete evaluation if there is no safe time to run another.
    if step != last_eval and time.monotonic() + evaluation_seconds + 15 < deadline:
        result, probability, masks = evaluate(ema, valid, dev, maps, family, config)
        result.update(step=step, elapsed=time.monotonic() - started)
        history.append(result)
        write_json(output / f"evaluation-{step}.json", result)
        if result["cand"] > best:
            best = result["cand"]
            save_torch(output / "selected.pt", {"ema": ema.state_dict(), "step": step, "family": family, "config": config})
            np.savez(output / "selected-validation.npz", names=valid.names, probabilities=probability, masks=masks, step=step)
    if step:
        save_checkpoint(output / "resume.pt", model, ema, optimizer, step, config, rng, target, history)
    final = history[-1] if history else None
    if family == "trustmark" and gate is None and final:
        gate = gate_decision(final["cand"], final["synthetic_miou"])
    completed = {"family": family, "stop": stop, "steps": step, "gate": gate,
                 "best_development_miou": best if history else None, "final": final,
                 "elapsed_seconds": time.monotonic() - started, "target_steps": target,
                 "public_score": None, "test_inference_executed": False}
    write_json(output / "completed.json", completed)


class GateTimeout(RuntimeError):
    pass


def launch_worker(command, timeout, gate_path=None):
    """Parent watchdog covers compiler stalls, evaluation overruns and hung workers."""
    started = time.monotonic()
    deadline = started + timeout
    gate_checked = gate_path is None
    with subprocess.Popen(command) as process:
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
                if not gate_checked and now - started >= 420:
                    gate_checked = True
                    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
                    if gate.get("decision") != "pass":
                        raise GateTimeout("No passing TrustMark evaluation by minute 7")
                try:
                    code = process.wait(timeout=min(1., deadline - now))
                except subprocess.TimeoutExpired:
                    continue
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def orchestrate(config_path, config, started=None):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic() if started is None else started
    deadline = started + config.get("total_seconds", 1800)
    write_json(output / "frozen-config.json", config)
    frozen_config = (output / "frozen-config.json").resolve()
    for family in ("trustmark", "pixelseal"):
        if family == "pixelseal":
            prior = json.loads((output / "trustmark/completed.json").read_text())
            if prior["gate"] != "pass":
                write_json(output / "completed.json", {"status": "stopped_after_trustmark", "trustmark": prior})
                return
        remaining = min(config.get("block_seconds", 780), deadline - time.monotonic() - 240)
        if remaining < 60:
            break
        cmd = [sys.executable, "-m", "experiments.h100_sprint.run", "--config", str(frozen_config),
               "--family", family, "--deadline", str(time.monotonic() + remaining)]
        try:
            launch_worker(cmd, remaining, output / "trustmark/minute7-gate.json" if family == "trustmark" else None)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, GateTimeout) as error:
            write_json(output / "completed.json", {"status": "stopped", "family": family,
                       "reason": type(error).__name__, "checkpoints_preserved": True})
            return
    metrics = {}
    frozen_ss = Path(config["packed"]) / "frozen_stegastamp.json"
    if frozen_ss.exists():
        metrics["stegastamp"] = json.loads(frozen_ss.read_text())["miou"]
    for family in ("trustmark", "pixelseal"):
        completed = output / family / "completed.json"
        if completed.exists():
            metrics[family] = json.loads(completed.read_text())["best_development_miou"]
    mean = sum(metrics.values()) / 3 if len(metrics) == 3 and all(v is not None for v in metrics.values()) else None
    inference = config.get("inference")
    if inference and time.monotonic() + 15 < deadline:
        try:
            subprocess.run([sys.executable, "-m", "experiments.h100_sprint.infer", "--config", str(frozen_config)],
                           check=True, timeout=deadline - time.monotonic())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            write_json(output / "completed.json", {"status": "inference_stopped", "reason": type(error).__name__,
                       "elapsed": time.monotonic() - started})
            return
    write_json(output / "completed.json", {"status": "training_finished", "elapsed": time.monotonic() - started,
               "families": metrics, "development_mean": mean, "quality_target_met": mean is not None and mean >= .80,
               "inference_requested": bool(inference), "next": "Read inference.json; no leaderboard upload is performed"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--family", choices=("trustmark", "pixelseal"))
    p.add_argument("--deadline", type=float)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if a.family:
        if a.deadline is None:
            p.error("Worker requires its parent deadline")
        train_family(a.family, config, Path(config["output"]) / a.family, a.deadline)
        return
    started = time.monotonic()
    result = preflight(config)
    print(json.dumps(result, indent=2))
    if a.preflight:
        sys.exit(0 if result["ready"] else 2)
    if not result["ready"]:
        raise SystemExit("Preflight failed; no GPU work started")
    orchestrate(a.config.resolve(), config, started)


if __name__ == "__main__":
    main()
