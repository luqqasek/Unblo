"""Stratified three-fold inner selection; unchanged outer folds and training seeds.

Run from repository root with task-1/.venv/Scripts/python.exe <this file>.
Resumes completed fold/seed runs. Outer labels never select epoch or threshold.
"""
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'frozen_extractor'))
import cross_validate as cv

OUT=HERE/'results'
SEEDS=(42,7,123)


def initialize(features,seed):
    torch.manual_seed(seed)
    head=cv.model.Head([f.shape[1] for f in features])
    return head,torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001),np.random.default_rng(seed)


def inner_folds(fit,targets,flags):
    """Distribute each joint area/flag stratum round-robin; works for rare strata."""
    strata=flags.astype(int)*4+np.digitize(targets.mean((1,2)),[.25,.5,.75])
    rng=np.random.default_rng(42)
    parts=[[],[],[]]
    cursor=0
    for label in sorted(set(strata[fit])):
        group=rng.permutation([i for i in fit if strata[i]==label])
        for i in group:
            parts[cursor%3].append(int(i)); cursor+=1
    assert sorted(sum(parts,[]))==sorted(fit.tolist())
    return [np.array(sorted(p),dtype=int) for p in parts]


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    cache=torch.load(cv.model.OUT/'stegastamp_features.pt',weights_only=True)
    source=np.load(cv.OUT/'stegastamp_oof.npz')
    assert cache['names']==source['names'].tolist()
    names=cache['names']; features=cache['features']; gt=source['target']
    targets=torch.from_numpy(gt).float()
    meta=json.loads((cv.model.ROOT/'data/train/metadata.json').read_text())
    flags_by_name={r['image_name']:r['full_watermark'] for r in meta}
    flags=np.array([flags_by_name[n] for n in names])
    splits=json.loads((cv.OUT/'folds.json').read_text())['stegastamp']
    old=json.loads((HERE.parent/'stegastamp_diagnosis/results/metrics.json').read_text())
    baseline={(r['fold'],r['seed']):r for r in old['runs'] if r['mode'] in ('original','nested')}
    config=dict(seeds=list(SEEDS),outer_manifest_sha256=hashlib.sha256((cv.OUT/'folds.json').read_bytes()).hexdigest(),
        selection='3 stratified inner folds; pooled per-image mIoU at 0.5 selects epoch; pooled inner predictions select threshold; fresh outer refit',
        strata='full_watermark x mask area bins [0,.25,.5,.75,1]',split_seed=42,**cv.model.CONFIG)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    runs=[]; manifest=[]
    for fold,split in enumerate(splits,1):
        fit=np.array([names.index(n) for n in split['train']])
        val=np.array([names.index(n) for n in split['validation']])
        parts=inner_folds(fit,gt,flags)
        assert not set(fit)&set(val)
        manifest.append(dict(fold=fold,train=split['train'],validation=split['validation'],
            inner_validation=[[names[i] for i in p] for p in parts]))
        for seed in SEEDS:
            stem=f'stegastamp_fold{fold}_seed{seed}'
            result_file=OUT/f'{stem}.json'
            if result_file.exists():
                runs.append(json.loads(result_file.read_text())); continue
            models=[initialize(features,seed) for _ in parts]
            inner_fit=[np.setdiff1d(fit,p) for p in parts]
            best,stale,history=-1,0,[]
            for epoch in range(1,cv.model.CONFIG['max_epochs']+1):
                probs=[]
                for (head,opt,rng),train_ids,valid_ids in zip(models,inner_fit,parts):
                    assert not set(train_ids)&set(valid_ids)
                    cv.train_epoch(head,opt,rng,features,targets,train_ids)
                    probs.append(cv.predict(head,features,valid_ids))
                pooled=np.concatenate(probs)
                inner_gt=gt[np.concatenate(parts)]
                metric=cv.model.metrics(pooled,inner_gt,.5)['miou']
                history.append(metric)
                if metric>best:
                    best,stale,selected_epoch=metric,0,epoch
                    best_prob=pooled.copy()
                else: stale+=1
                if epoch%10==0:print(f'fold={fold} seed={seed} inner epoch={epoch} mean={metric:.4f}',flush=True)
                if stale>=cv.model.CONFIG['patience']:break
            threshold=max(cv.model.CONFIG['threshold_candidates'],key=lambda t:cv.model.metrics(best_prob,inner_gt,t)['miou'])
            head,opt,rng=initialize(features,seed)
            for _ in range(selected_epoch):cv.train_epoch(head,opt,rng,features,targets,fit)
            prob=cv.predict(head,features,val)
            measured=cv.model.metrics(prob,gt[val],threshold)
            base=baseline[fold,seed]
            result=dict(fold=fold,seed=seed,epoch=selected_epoch,threshold=threshold,miou=measured['miou'],
                baseline=base['miou'],delta=measured['miou']-base['miou'],per_image=measured['per_image'],
                iou_clean=measured['iou_clean'],iou_watermark=measured['iou_watermark'],inner_history=history,
                fixed_threshold_miou=cv.model.metrics(prob,gt[val],.5)['miou'],names=[names[i] for i in val])
            torch.save(dict(state_dict=head.state_dict(),channels=[f.shape[1] for f in features],epoch=selected_epoch,threshold=threshold,seed=seed),OUT/f'{stem}.pt')
            np.savez_compressed(OUT/f'{stem}.npz',names=np.array(names)[val],probability=prob,threshold=threshold)
            result_file.write_text(json.dumps(result,indent=2))
            runs.append(result)
            print(f'COMPLETE fold={fold} seed={seed}: {measured["miou"]:.6f} vs {base["miou"]:.6f}, delta={result["delta"]:+.6f}, epoch={selected_epoch}, threshold={threshold}',flush=True)
    (OUT/'splits.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'metrics.json').write_text(json.dumps(dict(config=config,runs=runs),indent=2))
    spec=importlib.util.spec_from_file_location('stable_selection_report',HERE/'report.py')
    report=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    report.main()


if __name__=='__main__':main()
