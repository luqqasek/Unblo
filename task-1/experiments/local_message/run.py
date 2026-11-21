"""WAM-inspired auxiliary local bit decoding on frozen watermark features."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'frozen_extractor'))
import cross_validate as cv
OUT=HERE/'results'
VARIANTS={'control':0., 'message01':.1, 'message1':1.}


class JointHead(cv.model.Head):
    def __init__(self,channels,bits):
        super().__init__(channels)
        self.message_head=nn.Conv2d(32,bits,1)

    def joint(self,features):
        x=torch.cat([F.interpolate(p(f),size=(32,32),mode='bilinear',align_corners=False)
                     for p,f in zip(self.projections,features)],1)
        shared=self.fuse[:-1](x)
        segment=F.interpolate(self.fuse[-1](shared),size=(128,128),mode='bilinear',align_corners=False)[:,0]
        return segment,self.message_head(shared)

    def forward(self,features):
        return self.joint(features)[0]


def message_loss(logits,messages,coverage):
    target=messages[:,:,None,None].expand_as(logits)
    per_pixel=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    mass=coverage.sum((1,2))
    numerator=(per_pixel*coverage[:,None]).sum((1,2,3))
    valid=mass>0
    return (numerator/(mass.clamp_min(1e-8)*messages.shape[1]))[valid].mean() if valid.any() else logits.sum()*0


def train_epoch(head,opt,rng,features,y,messages,coverage,ids,weight):
    head.train()
    order=rng.permutation(ids)
    for start in range(0,len(order),8):
        batch=order[start:start+8]
        opt.zero_grad()
        logits,bits=head.joint([f[batch] for f in features])
        p=logits.sigmoid(); target=y[batch]
        dice=1-((2*(p*target).sum((1,2))+1)/(p.sum((1,2))+target.sum((1,2))+1)).mean()
        objective=.5*F.binary_cross_entropy_with_logits(logits,target)+.5*dice
        if weight:
            objective=objective+weight*message_loss(bits,messages[batch],coverage[batch])
        objective.backward(); opt.step()


def diagnostics(head,features,ids,messages,coverage,fit):
    head.eval()
    with torch.inference_mode():
        bits=torch.cat([head.joint([f[ids[j:j+8]] for f in features])[1] for j in range(0,len(ids),8)])
    target=messages[ids]; weights=coverage[ids]
    def accuracy(expected,w):
        correct=((bits>0)==expected[:,:,None,None]).float().mean(1)
        return float(((correct*w).sum((1,2))/w.sum((1,2)).clamp_min(1e-8)).mean())
    majority=(messages[fit].mean(0)>=.5).float()[None].expand_as(target)
    return dict(bit_accuracy_watermark=accuracy(target,weights),
                bit_accuracy_shuffled_message=accuracy(target.roll(1,0),weights),
                bit_accuracy_clean=accuracy(target,1-weights),
                training_majority_accuracy=float((majority==target).float().mean()),
                masked_bit_bce=float(message_loss(bits,target,weights)))


def fit(variant,method,fold,ds,splits):
    path=OUT/variant/f'{method}_fold{fold+1}.json'
    if path.exists(): return json.loads(path.read_text())
    path.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    cache=cv.model.extract(method,ds); features=cache['features']; names=cache['names']
    lookup={m['image_name']:m for m in ds.meta}
    y=torch.from_numpy(np.stack([ds.m128[n] for n in names])).float()
    messages=torch.tensor([[int(c) for c in lookup[n]['message']] for n in names],dtype=torch.float32)
    coverage=F.avg_pool2d(y[:,None],4)[:,0]
    split=splits[method][fold]
    ids={k:np.array([names.index(n) for n in v]) for k,v in split.items()}
    assert not set(ids['train']) & set(ids['validation'])
    assert not set(ids['inner_train']) & set(ids['inner_validation'])
    assert set(ids['inner_train']) | set(ids['inner_validation']) == set(ids['train'])
    channels=[f.shape[1] for f in features]
    def initialize():
        torch.manual_seed(42)
        h=JointHead(channels,messages.shape[1])
        return h,torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001),np.random.default_rng(42)
    head,opt,rng=initialize(); best=-1; stale=0; history=[]
    for epoch in range(50):
        train_epoch(head,opt,rng,features,y,messages,coverage,ids['inner_train'],VARIANTS[variant])
        prob=cv.predict(head,features,ids['inner_validation'])
        measured=cv.model.metrics(prob,y[ids['inner_validation']].numpy(),.5)['miou']
        history.append(measured)
        if measured>best:
            best=measured; stale=0; chosen=epoch+1; state=copy.deepcopy(head.state_dict())
        else: stale+=1
        print(variant,method,fold+1,'inner',epoch+1,round(measured,5),flush=True)
        if stale>=8: break
    head.load_state_dict(state)
    prob=cv.predict(head,features,ids['inner_validation'])
    threshold=max(cv.model.CONFIG['threshold_candidates'],key=lambda t:cv.model.metrics(prob,y[ids['inner_validation']].numpy(),t)['miou'])
    head,opt,rng=initialize()
    for _ in range(chosen):
        train_epoch(head,opt,rng,features,y,messages,coverage,ids['train'],VARIANTS[variant])
    prob=cv.predict(head,features,ids['validation'])
    row=dict(variant=variant,method=method,fold=fold+1,weight=VARIANTS[variant],epoch=chosen,threshold=threshold,
             selected=cv.model.metrics(prob,y[ids['validation']].numpy(),threshold),
             fixed=cv.model.metrics(prob,y[ids['validation']].numpy(),.5),history=history,
             diagnostics=diagnostics(head,features,ids['validation'],messages,coverage,ids['train']),
             train_names=split['train'],val_names=split['validation'],seconds=time.perf_counter()-start)
    torch.save(dict(state_dict=head.state_dict(),channels=channels,bits=messages.shape[1],weight=VARIANTS[variant],epoch=chosen,threshold=threshold),path.with_suffix('.pt'))
    restored=JointHead(channels,messages.shape[1]); restored.load_state_dict(torch.load(path.with_suffix('.pt'),weights_only=True)['state_dict'])
    np.testing.assert_allclose(cv.predict(restored,features,ids['validation']),prob,atol=1e-7)
    np.savez_compressed(path.with_suffix('.npz'),probability=prob,target=y[ids['validation']].numpy(),names=split['validation'])
    if variant=='control':
        baseline=json.loads((cv.OUT/'metrics.json').read_text())['methods'][method]['folds'][fold]
        np.testing.assert_allclose(row['selected']['per_image'],baseline['selected']['per_image'],atol=1e-7)
        assert chosen==baseline['epoch'] and threshold==baseline['threshold']
    path.write_text(json.dumps(row,indent=2),encoding='utf-8')
    print('RESULT',variant,method,fold+1,row['selected']['miou'],row['diagnostics'],flush=True)
    return row


def publish(summary):
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
    subprocess.run([sys.executable,str(cv.model.ROOT/'dashboard/build.py')],check=True)


def main():
    torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    ds=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    splits=json.loads((cv.OUT/'folds.json').read_text())
    baseline=json.loads((cv.OUT/'metrics.json').read_text())
    summary=dict(status='running',protocol='WAM-inspired masked auxiliary bit BCE at 32x32; segmentation unchanged',
                 gate='Overall delta>=0.005 and at least two methods improve; only best passing weight proceeds to CV',variants={})
    for variant in VARIANTS:
        rows=[fit(variant,m,0,ds,splits) for m in cv.METHODS]
        delta=[r['selected']['miou']-baseline['methods'][r['method']]['folds'][0]['selected']['miou'] for r in rows]
        passed=bool(variant!='control' and np.mean(delta)>=.005 and sum(d>0 for d in delta)>=2)
        summary['variants'][variant]=dict(screen=rows,screen_delta=float(np.mean(delta)),passed=passed,cv=[])
        print('GATE',variant,passed,np.mean(delta),flush=True); publish(summary)
    candidates=[(v['screen_delta'],k) for k,v in summary['variants'].items() if v['passed']]
    summary['cv_variant']=max(candidates)[1] if candidates else None
    if candidates:
        winner=summary['cv_variant']; rows=summary['variants'][winner]['screen'].copy()
        for fold in (1,2): rows.extend(fit(winner,m,fold,ds,splits) for m in cv.METHODS)
        summary['variants'][winner]['cv']=rows
    summary['status']='complete'; publish(summary)


if __name__=='__main__': main()
