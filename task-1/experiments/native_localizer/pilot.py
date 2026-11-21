"""Bounded pilot: throughput-aware batch, source-disjoint dev, real held-out mIoU.

This is a short screening protocol requested by the user, not the full 50k run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .common import PROJECT, atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--gpus",type=int,default=4)
    p.add_argument("--run-name",default="pilot-30min")
    a=p.parse_args(); started=time.monotonic()
    if not a.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in a.run_name) or a.run_name in (".",".."):
        raise ValueError("Use a simple experiment name without path separators")
    output=a.root/f"runs/stegastamp/{a.run_name}"
    if (output/"last.pt").exists() or (output/"train.jsonl").exists():
        raise FileExistsError("Preserving existing experiment: pass a fresh --run-name, or resume train.py explicitly")
    def run(module,*args,check=True):
        print("PILOT_STAGE",module,*args,flush=True)
        return subprocess.run([sys.executable,"-m",f"experiments.native_localizer.{module}",*map(str,args)],check=check)
    backbone=Path("checkpoints/convnext_base.fb_in22k/model.safetensors")
    fidelity=a.root/"fidelity-stegastamp-16pairs.json"
    if not fidelity.exists():
        run("fidelity","--manifest",a.root/"encoded/stegastamp/smoke-train/encoded.jsonl",
            "--backbone",backbone,"--out",fidelity,"--count",16)
    checks=json.loads(fidelity.read_text())
    gradient_cosines=[x for row in checks["rows"] for x in row["gradient_cosines"].values()]
    amp=bool(checks["first_projection_exact"] and checks["local_stem_exact"] and
             checks["all_gradients_finite"] and checks["paired_response_cosine_min"]>=.95 and min(gradient_cosines)>=.95)
    print(json.dumps({"precision_decision":"deeper_AMP" if amp else "FP32",
                      "response_cosine_min":checks["paired_response_cosine_min"],
                      "gradient_cosine_min":min(gradient_cosines)}),flush=True)
    profiles=[]
    # The earlier 100-step batch16 profiles are the reference; these are short
    # batch-capacity comparisons. Keep each run resumeless and separate.
    for batch in (64,128):
        path=a.root/f"pilot-profile-b{batch}.json"
        args=["--backbone",backbone,"--out",path,"--steps",20,"--microbatch",batch,"--gpus",a.gpus]
        if amp: args.append("--amp")
        result=run("profile",*args,check=False)
        if result.returncode==0:
            value=json.loads(path.read_text())
            if value["peak_memory_bytes"] < .9*value["vram_bytes"]:
                profiles.append((batch/value["seconds_per_microstep"],batch,value))
        else:
            print(json.dumps({"batch":batch,"profile_failed":True,"training_not_started_with_this_batch":True}),flush=True)
    if not profiles: raise RuntimeError("No validated larger batch fits the safety margin")
    throughput,batch,profile=max(profiles,key=lambda x:x[0])
    atomic_json(a.root/"pilot-batch-decision.json",{
        "batch":batch,"gpus":a.gpus,"images_per_second_compute_only":throughput,
        "precision":"deeper_AMP" if amp else "FP32", "profile":profile,
        "selection":"highest measured images/s under 90% VRAM; fixed before validation labels",
    })
    for split,count in (("train",2500),("dev",128)):
        manifest=a.root/f"sources/coco2017/{split}.jsonl"
        run("screen_sources","--sources",manifest,"--fingerprints",a.root/"sources/coco2017/competition_fingerprints.json")
        run("generator","--sources",manifest,"--partition",split,"--method","stegastamp",
            "--count",count,"--models","checkpoints","--out",a.root/f"encoded/stegastamp/pilot-{split}")
    cfg=json.loads((PROJECT/"config.json").read_text())
    cfg.update(effective_batch=batch,microbatch=batch,steps=200,warmup=20,
               aux_ramp=max(1,32000//batch),ema=.999**(batch/16),eval_every=50,
               checkpoint_every=100,cache_rgb=True,evaluation_weights="raw",amp=amp,
               pilot=True,protocol="predeclared 200 updates; no real labels train/select; raw last weights; real fold1 once @0.5")
    cfgpath=a.root/"pilot-config.json"; atomic_json(cfgpath,cfg)
    if time.monotonic()-started > 18*60:
        raise RuntimeError("Preparation exceeded pilot time budget; do not silently start a long experiment")
    run("train","--config",cfgpath,"--family","stegastamp",
        "--train",a.root/"encoded/stegastamp/pilot-train/encoded.jsonl",
        "--dev",a.root/"encoded/stegastamp/pilot-dev/encoded.jsonl",
        "--out",output,"--backbone",backbone,
        "--gpus",a.gpus,"--dev-limit",64,"--real-fold",1,"--real-data","data")
    atomic_json(a.root/"pilot-completed.json",{
        "seconds":time.monotonic()-started,"batch":batch,"updates":200,"views":200*batch,
        "real_validation_images":57,"family":"stegastamp","not_a_final_competition_pipeline":True,
    })


if __name__=="__main__":
    main()
