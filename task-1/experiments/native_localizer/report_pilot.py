"""Validate saved pilot metrics/maps and create the final bounded-pilot readout."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from wmloc.utils import miou_image
from .common import atomic_json, jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    run=a.artifacts/"runs/stegastamp/pilot-30min"
    real_dir=run/("real_validation_all" if (run/"real_validation_all/metrics.json").exists() else "real_validation")
    real=json.loads((real_dir/"metrics.json").read_text())
    count=real["n_unique_images"]
    fold_description="all three original fold partitions" if count==170 else "fold1"
    cfg=json.loads((run/"config.json").read_text())
    curves=jsonl(run/"dev.jsonl")
    last_step=curves[-1]["update"]
    synthetic=json.loads((run/f"dev/{last_step:07d}/metrics.json").read_text())
    maps=np.load(real_dir/"maps.npz")
    names=maps["names"].tolist(); prob=maps["probabilities"]; target=maps["targets"]
    with open(real_dir/"per_image.csv") as f:
        rows=list(csv.DictReader(f))
    if names != [r["image_name"] for r in rows] or len(names)!=count or len(set(names))!=count:
        raise ValueError("Real validation identities/counts disagree")
    if prob.dtype != np.float32:
        raise ValueError("Require FP32 maps for faithful reproduction of threshold and AP")
    scores=np.array([miou_image(p>=.5,y) for p,y in zip(prob,target)])
    if not np.allclose(scores,[float(r["miou"]) for r in rows],atol=1e-10,rtol=0):
        raise ValueError("Saved probability maps do not reproduce official per-image scores")
    groups=real["groups"]
    if abs(scores.mean()-groups["all"]["miou"]["mean"])>1e-10:
        raise ValueError("Summary score disagrees with official recomputation")
    rng=np.random.default_rng(42)
    bootstrap=scores[rng.integers(0,len(scores),(5000,len(scores)))].mean(1)
    interval=np.quantile(bootstrap,[.025,.975]).tolist()
    clean=np.array([float(r["constant_clean_miou"]) for r in rows])
    marked=np.array([float(r["constant_marked_miou"]) for r in rows])
    shuffled=np.array([miou_image(np.random.default_rng(42+i).permutation(p.ravel()).reshape(p.shape)>=.5,y)
                       for i,(p,y) in enumerate(zip(prob,target))])
    result={"real_validation_miou":float(scores.mean()),"image_bootstrap95":interval,
            "n_unique_real_images":count,"family":"stegastamp","folds":[1,2,3] if count==170 else [1],"threshold":.5,
            "weights":"raw last predeclared pilot update","updates":last_step,
            "effective_batch":cfg["effective_batch"],"training_views":last_step*cfg["effective_batch"],
            "fully_clean_constant_miou":float(clean.mean()),"fully_marked_constant_miou":float(marked.mean()),
            "oracle_best_constant_per_image_miou":float(np.maximum(clean,marked).mean()),
            "spatial_shuffle_miou":float(shuffled.mean()),
            "synthetic_dev_miou":synthetic["groups"]["all"]["miou"]["mean"],
            "synthetic_dev_partial_miou":synthetic["groups"]["partial"]["miou"]["mean"],
            "real_fg_iou":groups["all"]["fg_iou"]["mean"],"real_clean_iou":groups["all"]["clean_iou"]["mean"],
            "real_fg_ap":groups["all"]["fg_ap"]["mean"],"real_roc_auc":groups["all"]["roc_auc"]["mean"],
            "real_small_fg_iou":groups["fg_le_1pct"]["fg_iou"]["mean"],
            "real_small_n":groups["fg_le_1pct"]["n"],
            "real_1_to_5pct_fg_iou":groups["fg_1_to_5pct"]["fg_iou"]["mean"],
            "real_large_clean_iou":groups["fg_gt_50pct"]["clean_iou"]["mean"],
            "metrics_maps_verified":True,
            "limitations":["one family, reconstructed existing fold partitions", "repeatedly used real development data",
                           "no real labels used for model training/checkpoint/threshold selection",
                           "simple synthetic hard composites only; full edit mixture not implemented in this pilot",
                           f"bootstrap conditions on this fitted model and these {count} images; edited source siblings cannot be excluded",
                           "not a calibrated estimate of overall three-family hidden competition score"]}
    atomic_json(a.out/"summary.json",result)
    lines=["# Native RGB localizer: bounded quality pilot", "",
           f"**Real held-out StegaStamp mIoU: {scores.mean():.5f}** on {count} images, {fold_description}, threshold0.5.",
           f"Conditional image bootstrap95%: [{interval[0]:.5f}, {interval[1]:.5f}].", "",
           "The real validation images were not used to train or select the checkpoint or threshold.",
           "The final pilot update and raw weights were declared before reading real validation scores.", "",
           "| Evaluation | mIoU |", "|---|---:|",
           f"| Real held-out, {count} images | {scores.mean():.5f} |",
           f"| Synthetic dev, 64 unseen sources/messages | {result['synthetic_dev_miou']:.5f} |",
           f"| Synthetic dev, partial masks only | {result['synthetic_dev_partial_miou']:.5f} |",
           f"| Real: all-clean constant | {clean.mean():.5f} |",
           f"| Real: all-marked constant | {marked.mean():.5f} |",
           f"| Real: spatially shuffled model maps | {shuffled.mean():.5f} |",
           f"| Real: best constant per image, GT ORACLE diagnostic | {np.maximum(clean,marked).mean():.5f} |", "",
           f"FG IoU {result['real_fg_iou']:.5f}; clean IoU {result['real_clean_iou']:.5f}; macro FG AP {result['real_fg_ap']:.5f}.",
           f"Training: {last_step} updates x batch{cfg['effective_batch']} = {result['training_views']} views; 2,500 distinct COCO sources.",
           "ConvNeXt-v1 Base ImageNet-22K + learned local stem + progressive decoder; all stages trainable.",
           "93,777,762 parameters, native512 input, native128 logits. One node, four A10040GB, FP32.", "",
           "## Limits", "", *[f"- {x}." for x in result["limitations"]], "",
           "No test predictions, leaderboard submission, full nested CV or other-family training occurred in this pilot.",
           "The old models and outputs were preserved. This result must not be compared causally with historical",
           "multi-fold results as though budgets, splits and generators were matched.", "",
           "## Artifacts", "",
           "FP32 probability/GT maps and per-image CSV reproduce the official score exactly. Training/dev curves,",
           "source/message manifests, numerical-fidelity checks, resource profiles and failure galleries accompany this report."]
    (a.out/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
