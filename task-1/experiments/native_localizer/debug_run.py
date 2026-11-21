"""Sequential exact-embedding, signal-fidelity and tiny-overfit checks on ONE node."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .common import PROJECT, atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--family", default="stegastamp", choices=["pixelseal", "trustmark", "stegastamp"])
    p.add_argument("--gpus", type=int, default=1)
    a = p.parse_args()
    def run(module, *args):
        command = [sys.executable, "-m", f"experiments.native_localizer.{module}", *map(str,args)]
        print("STAGE", module, " ".join(map(str,args)), flush=True)
        subprocess.run(command, check=True)
    train = a.root/f"encoded/{a.family}/smoke-train"
    dev = a.root/f"encoded/{a.family}/smoke-dev"
    for partition, out, count in (("train", train, 64), ("dev", dev, 32)):
        run("screen_sources", "--sources", a.root/f"sources/coco2017/{partition}.jsonl",
            "--fingerprints", a.root/"sources/coco2017/competition_fingerprints.json")
        run("generator", "--sources", a.root/f"sources/coco2017/{partition}.jsonl",
            "--out", out, "--method", a.family, "--partition", partition, "--count", count,
            "--models", "checkpoints", "--decode-checks", 8)
    backbone = Path("checkpoints/convnext_base.fb_in22k/model.safetensors")
    run("profile", "--backbone", backbone, "--out", a.root/f"profile-{a.family}-paired-fp32.json",
        "--steps", 100, "--microbatch", 16, "--encoded", train/"encoded.jsonl", "--gpus", a.gpus)
    # AMP is profiled, not enabled for actual training, pending fidelity review.
    run("profile", "--backbone", backbone, "--out", a.root/f"profile-{a.family}-paired-amp.json",
        "--steps", 100, "--microbatch", 16, "--encoded", train/"encoded.jsonl", "--gpus", a.gpus, "--amp")
    cfg = json.loads((PROJECT/"config.json").read_text())
    cfg.update(warmup=10, steps=400, microbatch=16, checkpoint_every=100)
    # Faster warmup is ONLY for this fixed-batch capacity diagnostic.
    cfgpath = a.root/"overfit-config.json"; atomic_json(cfgpath, cfg)
    run("train", "--config", cfgpath, "--family", a.family, "--backbone", backbone,
        "--train", train/"encoded.jsonl", "--out", a.root/f"runs/{a.family}/overfit-fp32",
        "--overfit", "--stop-after", 200, "--gpus", a.gpus)
    atomic_json(a.root/f"debug-{a.family}-completed.json", {
        "exact_train_encoded_realizations": 64, "exact_dev_encoded_realizations": 32,
        "overfit_updates": 200, "generalization_demonstrated": False,
        "amp_training_enabled": False,
    })


if __name__ == "__main__":
    main()
