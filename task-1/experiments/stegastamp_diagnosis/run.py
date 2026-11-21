"""Diagnose saved OOF errors and repeat StegaStamp training on identical folds."""
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'frozen_extractor'))
import cross_validate as cv

OUT = HERE / 'results'
SEEDS = (7, 123)


def initialize(features, seed):
    torch.manual_seed(seed)
    head = cv.model.Head([f.shape[1] for f in features])
    return head, torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001), np.random.default_rng(seed)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    splits = json.loads((cv.OUT / 'folds.json').read_text())
    original = json.loads((cv.OUT / 'metrics.json').read_text())
    ds = cv.model.LocalisationDataset(cv.model.ROOT / 'data', 'train')
    metadata = {m['image_name']: m for m in ds.meta}
    rows, summaries, runs = [], [], []
    for method in cv.METHODS:
        saved = np.load(cv.OUT / f'{method}_oof.npz')
        names = saved['names'].tolist()
        cache = torch.load(cv.model.OUT / f'{method}_features.pt', weights_only=True)
        assert names == cache['names']
        target = saved['target']
        values = cv.model.metrics(saved['probability'], target, saved['threshold'][:, None, None])['per_image']
        assert abs(np.mean(values) - original['methods'][method]['oof_miou']) < 1e-12
        for fold, split in enumerate(splits[method], 1):
            ids = [names.index(n) for n in split['validation']]
            train = [names.index(n) for n in split['train']]
            assert not set(ids) & set(train)
            prior = target[train].mean(0) >= .5
            for i in ids:
                mask = target[i].astype(bool)
                pred = saved['probability'][i] >= saved['threshold'][i]
                holes = ndimage.binary_fill_holes(mask) & ~mask
                boundary = (np.count_nonzero(mask[1:] != mask[:-1]) + np.count_nonzero(mask[:, 1:] != mask[:, :-1])) / mask.size
                rows.append(dict(method=method, fold=fold, name=names[i], miou=values[i],
                    area=float(mask.mean()), predicted_area=float(pred.mean()),
                    holes=float(holes.mean()), boundary=float(boundary),
                    components=int(ndimage.label(mask)[1]), full_watermark=metadata[names[i]]['full_watermark'],
                    bit_accuracy=cache['bit_accuracy'][i],
                    fp=float((pred & ~mask).mean()), fn=float((~pred & mask).mean()),
                    prior=cv.model.miou_image(prior, mask)))
            fr = [r for r in rows if r['method'] == method and r['fold'] == fold]
            summary = dict(method=method, fold=fold, n=len(fr))
            for key in ['miou','area','predicted_area','holes','boundary','components','full_watermark','bit_accuracy','fp','fn','prior']:
                summary[key] = float(np.mean([r[key] for r in fr]))
            summary['small_region_n'] = sum(r['area'] < .25 for r in fr)
            summary['by_full_watermark'] = {str(flag): dict(n=sum(r['full_watermark']==flag for r in fr),
                miou=float(np.mean([r['miou'] for r in fr if r['full_watermark']==flag]))) for flag in [False,True]}
            summaries.append(summary)
        if method != 'stegastamp':
            continue
        features = cache['features']
        targets = torch.from_numpy(target).float()
        for fold, split in enumerate(splits[method], 1):
            ids = {key: [names.index(n) for n in split[key]] for key in split}
            orig = original['methods'][method]['folds'][fold-1]
            runs.append(dict(fold=fold, seed=42, mode='original', epoch=orig['epoch'], threshold=orig['threshold'], miou=orig['selected']['miou']))
            for seed in SEEDS:
                # Fixed hyperparameters isolate seed variation from selection variation.
                head, opt, rng = initialize(features, seed)
                for epoch in range(orig['epoch']):
                    cv.train_epoch(head,opt,rng,features,targets,ids['train'])
                prob = cv.predict(head,features,ids['validation'])
                fixed = cv.model.metrics(prob,target[ids['validation']],orig['threshold'])
                runs.append(dict(fold=fold,seed=seed,mode='fixed',epoch=orig['epoch'],threshold=orig['threshold'],miou=fixed['miou']))
                # Repeat the original selection procedure using identical inner splits.
                head,opt,rng = initialize(features,seed)
                best,stale,history = -1,0,[]
                for epoch in range(cv.model.CONFIG['max_epochs']):
                    cv.train_epoch(head,opt,rng,features,targets,ids['inner_train'])
                    prob = cv.predict(head,features,ids['inner_validation'])
                    metric = cv.model.metrics(prob,target[ids['inner_validation']],.5)['miou']
                    history.append(metric)
                    if metric > best:
                        best,stale,selected_epoch = metric,0,epoch+1
                        state = copy.deepcopy(head.state_dict())
                    else:
                        stale += 1
                    if stale >= cv.model.CONFIG['patience']:
                        break
                head.load_state_dict(state)
                prob = cv.predict(head,features,ids['inner_validation'])
                threshold = max(cv.model.CONFIG['threshold_candidates'],key=lambda t: cv.model.metrics(prob,target[ids['inner_validation']],t)['miou'])
                head,opt,rng = initialize(features,seed)
                for epoch in range(selected_epoch):
                    cv.train_epoch(head,opt,rng,features,targets,ids['train'])
                prob = cv.predict(head,features,ids['validation'])
                measured = cv.model.metrics(prob,target[ids['validation']],threshold)
                runs.append(dict(fold=fold,seed=seed,mode='nested',epoch=selected_epoch,threshold=threshold,
                    miou=measured['miou'],fixed_threshold_miou=cv.model.metrics(prob,target[ids['validation']],.5)['miou'],inner_history=history))
                np.savez_compressed(OUT/f'stegastamp_fold{fold}_seed{seed}.npz',names=np.array(names)[ids['validation']],probability=prob,threshold=threshold)
                torch.save(dict(state_dict=head.state_dict(),seed=seed,epoch=selected_epoch,threshold=threshold),OUT/f'stegastamp_fold{fold}_seed{seed}.pt')
                print(f'Fold {fold}, seed {seed}: fixed {fixed["miou"]:.4f}, nested {measured["miou"]:.4f}, epoch {selected_epoch}, threshold {threshold}',flush=True)
        # Inspect original errors: 4 worst and 2 best images per fold.
        for fold,split in enumerate(splits[method],1):
            ids = sorted([names.index(n) for n in split['validation']],key=lambda i: values[i])
            chosen = ids[:4] + ids[-2:]
            fig,axes = plt.subplots(6,4,figsize=(12,17))
            for ax,i in zip(axes,chosen):
                im = Image.open(ds.dir/'images'/f'{names[i]}.png')
                pred = saved['probability'][i]>=saved['threshold'][i]
                error = np.zeros((128,128,3)); error[...,0]=pred & ~target[i].astype(bool); error[...,2]=~pred & target[i].astype(bool)
                for a,img,title in zip(ax,[im,target[i],pred,error],[f'{names[i]}\nmIoU {values[i]:.3f}','Ground truth','Prediction','FP red / FN blue']):
                    a.imshow(img,vmin=0,vmax=1); a.set_title(title,fontsize=9); a.axis('off')
            fig.tight_layout(); fig.savefig(OUT/f'stegastamp_fold{fold}_examples.png',dpi=110); plt.close(fig)
    with (OUT/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result=dict(protocol='Diagnostic repeats on existing 3-fold CV; original inner/outer manifests unchanged. No test data.',
        seeds=[42,*SEEDS],summaries=summaries,runs=runs,metadata_keys=sorted(ds.meta[0]),
        config=cv.model.CONFIG)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for fold in [1,2,3]:
        r=[r for r in rows if r['method']=='stegastamp' and r['fold']==fold]
        axes[0].scatter([x['area'] for x in r],[x['miou'] for x in r],alpha=.6,label=f'Fold {fold}')
        rs=[r for r in runs if r['fold']==fold and r['mode']!='fixed']
        axes[1].plot([r['seed'] for r in rs],[r['miou'] for r in rs],'o',label=f'Fold {fold}')
    axes[0].set(xlabel='Watermarked fraction',ylabel='Original OOF mIoU'); axes[1].set(xlabel='Training seed',ylabel='Nested selection mIoU')
    for a in axes:a.legend()
    fig.tight_layout(); fig.savefig(OUT/'stegastamp_diagnosis_examples.png',dpi=130); plt.close(fig)
    print('Saved diagnosis artifacts:',OUT,flush=True)


if __name__=='__main__': main()
