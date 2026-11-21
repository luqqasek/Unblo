"""First non-memorization learning experiment: exact 2.5k shard + fixed source-disjoint dev."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .common import atomic_json, jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--family", choices=["pixelseal", "trustmark", "stegastamp"], required=True)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--stop-after", type=int, default=1000)
    p.add_argument("--amp-validated", action="store_true")
    a = p.parse_args()
    def run(module, *args):
        print("STAGE", module, *args, flush=True)
        subprocess.run([sys.executable,"-m",f"experiments.native_localizer.{module}",*map(str,args)],check=True)
    for split, count in (("train",2500),("dev",256)):
        manifest = a.root/f"sources/coco2017/{split}.jsonl"
        run("screen_sources","--sources",manifest,"--fingerprints",a.root/"sources/coco2017/competition_fingerprints.json")
        rows = [r for r in jsonl(manifest) if not r.get("excluded")]
        if len(rows) < count:
            raise ValueError(f"Need {count} distinct {split} sources; only {len(rows)} available")
        run("generator","--sources",manifest,"--partition",split,"--method",a.family,
            "--count",count,"--models","checkpoints","--out",a.root/f"encoded/{a.family}/debug-{split}")
    out = a.root/f"runs/{a.family}/debug-seed42"
    args = ["--family",a.family,"--train",a.root/f"encoded/{a.family}/debug-train/encoded.jsonl",
            "--dev",a.root/f"encoded/{a.family}/debug-dev/encoded.jsonl",
            "--backbone","checkpoints/convnext_base.fb_in22k/model.safetensors",
            "--out",out,"--microbatch",16,"--gpus",a.gpus,"--stop-after",a.stop_after]
    if a.amp_validated:
        args.append("--amp-validated")
    if (out/"last.pt").exists():
        args.extend(["--resume",out/"last.pt"])
    run("train",*args)
    atomic_json(a.root/f"learning-{a.family}-last-invocation.json",{
        "stage":"same-content exact synthetic debug generalization",
        "train_sources":2500,"dev_sources":256,"target_update":a.stop_after,
        "is_full_generator_or_final_recipe":False,
    })


if __name__ == "__main__":
    main()
