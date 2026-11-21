"""Predeclared synthetic-only transfer evaluation; real labels never train/select."""
import csv
from pathlib import Path
import json

import numpy as np
import torch

from wmloc.data import LocalisationDataset
from .common import PROJECT, atomic_json, rgb_tensor
from .evaluate import image_metrics, summarize, gallery


@torch.inference_mode()
def evaluate_real(model, data, family, fold, out, device="cuda"):
    ds=LocalisationDataset(data,"train",methods=[family])
    splits=json.loads((PROJECT/"reconstructed_splits.json").read_text())["families"][family]
    names=splits[fold-1]["validation"]
    lookup={r["image_name"]:i for i,r in enumerate(ds.meta)}
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    was_training=model.training; model.eval()
    rows=[]; maps=[]; targets=[]; examples=[]
    for name in names:
        item=ds[lookup[name]]
        probability=model(rgb_tensor(item["image"])[None].to(device))["logits"].sigmoid()[0].cpu().numpy()
        row=image_metrics(probability,item["mask128"],.5)
        row.update(image_name=name,family=family,fold=fold)
        rows.append(row); maps.append(probability.astype(np.float32)); targets.append(item["mask128"])
        examples.append((row["miou"],name,item["image"],item["mask128"],probability))
    summary=summarize(rows)
    summary.update(family=family,fold=fold,n_unique_images=len(names),threshold=.5,
                   protocol="synthetic-only; final predeclared pilot update; raw weights; no real labels used for training/selection",
                   limitation="reconstructed original fold; repeatedly used development data; one family, not overall hidden-test score")
    atomic_json(out/"metrics.json",summary)
    with open(out/"per_image.csv","w") as f:
        writer=csv.DictWriter(f,fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/"maps.npz",names=names,probabilities=np.stack(maps),targets=np.stack(targets))
    (out/"gallery").mkdir(exist_ok=True)
    ordered=sorted(examples,key=lambda x:x[0])
    for value,name,image,gt,prob in ordered[:6]+ordered[-3:]:
        gallery(out/"gallery"/f"{name}-{value:.3f}.png",image,gt,prob)
    print(json.dumps({"REAL_VALIDATION":True,"family":family,"fold":fold,"n":len(names),
                      "miou":summary["groups"]["all"]["miou"]["mean"],
                      "fg_iou":summary["groups"]["all"]["fg_iou"]["mean"],
                      "clean_iou":summary["groups"]["all"]["clean_iou"]["mean"]}),flush=True)
    model.train(was_training)
    return summary


def main():
    import argparse
    from .model import NativeLocalizer
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--folds",nargs="+",type=int,default=[2,3])
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    family = state["config"]["family"]
    if not state["config"].get("pilot") or family not in ("stegastamp", "pixelseal", "trustmark"):
        raise ValueError("Additional real folds are valid here only for the label-independent synthetic-only pilot")
    model=NativeLocalizer(checkpointing=False).cuda().eval()
    model.load_state_dict(state["model"])
    for fold in a.folds:
        target=a.run/("real_validation" if fold==1 else f"real_validation_fold{fold}")
        if not (target/"metrics.json").exists():
            evaluate_real(model,a.data,family,fold,target)
    rows=[]; names=[]; probabilities=[]; targets=[]
    for fold in (1,2,3):
        directory=a.run/("real_validation" if fold==1 else f"real_validation_fold{fold}")
        with open(directory/"per_image.csv") as f:
            for r in csv.DictReader(f):
                rows.append({k:(v if k in ("image_name","family") else None if v=="" else float(v)) for k,v in r.items()})
        z=np.load(directory/"maps.npz")
        names.extend(z["names"].tolist()); probabilities.extend(z["probabilities"]); targets.extend(z["targets"])
    if len(names)!=170 or len(set(names))!=170:
        raise ValueError("Require each real family image exactly once")
    out=a.run/"real_validation_all"; out.mkdir(exist_ok=True)
    summary=summarize(rows)
    summary.update(family=family,n_unique_images=170,folds=[1,2,3],threshold=.5,
                   protocol="same frozen synthetic-only checkpoint; no real training or label-based selection; original outer-fold partitions pooled per image")
    atomic_json(out/"metrics.json",summary)
    with open(out/"per_image.csv","w") as f:
        writer=csv.DictWriter(f,fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/"maps.npz",names=names,probabilities=np.stack(probabilities),targets=np.stack(targets))
    print(json.dumps({"ALL_REAL_VALIDATION":True,"n":170,"miou":summary["groups"]["all"]["miou"]["mean"]}),flush=True)


if __name__=="__main__":
    main()
