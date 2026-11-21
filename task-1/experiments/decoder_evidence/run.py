"""Learn localization from frozen pretrained decoder/message agreement."""
import copy
import argparse
import json
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from prepare import HERE, OUT, cv, prepare, prepare_bits


class EvidenceHead(nn.Module):
    def __init__(self,channels,variant,evidence_channels=4):
        super().__init__()
        self.variant=variant
        self.base=cv.model.Head(channels) if variant not in ('map_only','bit_map') else None
        self.correction=nn.Sequential(nn.Conv2d(evidence_channels,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))
        if self.base is not None:
            nn.init.zeros_(self.correction[-1].weight); nn.init.zeros_(self.correction[-1].bias)

    def forward(self,features):
        evidence=F.interpolate(features[-1],size=(32,32),mode='bilinear',align_corners=False)
        correction=F.interpolate(self.correction(evidence),size=(128,128),mode='bilinear',align_corners=False)[:,0]
        return correction if self.base is None else self.base(features[:-1])+correction


def fit(variant,method,fold,ds,splits,evidence):
    if variant=='blend': return fit_blend(method,fold,ds,splits,evidence)
    path=OUT/variant/f'{method}_fold{fold+1}.json'
    if path.exists(): return json.loads(path.read_text())
    path.parent.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    cache=cv.model.extract(method,ds); names=cache['names']
    assert names==evidence['names']
    is_wrong=variant.endswith('_wrong')
    real_key='perbit' if variant.startswith('bit_') else 'real'
    wrong_key='wrong_perbit' if variant.startswith('bit_') else 'wrong'
    features=cache['features']+[evidence[wrong_key if is_wrong else real_key]]
    target=torch.from_numpy(np.stack([ds.m128[n] for n in names])).float()
    split=splits[method][fold]
    ids={k:np.array([names.index(n) for n in v]) for k,v in split.items()}
    assert not set(ids['train']) & set(ids['validation'])
    assert not set(ids['inner_train']) & set(ids['inner_validation'])
    assert set(ids['inner_train']) | set(ids['inner_validation']) == set(ids['train'])
    channels=[f.shape[1] for f in cache['features']]
    def initialize():
        torch.manual_seed(42); h=EvidenceHead(channels,variant,features[-1].shape[1])
        return h,torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001),np.random.default_rng(42)
    head,opt,rng=initialize(); best=-1; stale=0; history=[]
    for epoch in range(50):
        cv.train_epoch(head,opt,rng,features,target,ids['inner_train'])
        p=cv.predict(head,features,ids['inner_validation'])
        score=cv.model.metrics(p,target[ids['inner_validation']].numpy(),.5)['miou']
        history.append(score)
        if score>best:
            best=score; stale=0; chosen=epoch+1; state=copy.deepcopy(head.state_dict())
        else: stale+=1
        print(variant,method,fold+1,'inner',epoch+1,round(score,5),flush=True)
        if stale>=8: break
    head.load_state_dict(state)
    p=cv.predict(head,features,ids['inner_validation'])
    threshold=max(cv.model.CONFIG['threshold_candidates'],key=lambda t:cv.model.metrics(p,target[ids['inner_validation']].numpy(),t)['miou'])
    head,opt,rng=initialize()
    for _ in range(chosen): cv.train_epoch(head,opt,rng,features,target,ids['train'])
    p=cv.predict(head,features,ids['validation'])
    counter=cv.predict(head,cache['features']+[evidence[real_key if is_wrong else wrong_key]],ids['validation'])
    selected=cv.model.metrics(p,target[ids['validation']].numpy(),threshold)
    counter_metrics=cv.model.metrics(counter,target[ids['validation']].numpy(),threshold)
    row=dict(variant=variant,method=method,fold=fold+1,selected=selected,
             counterfactual=counter_metrics,epoch=chosen,threshold=threshold,history=history,
             train_names=split['train'],val_names=split['validation'],seconds=time.perf_counter()-start)
    torch.save(dict(state_dict=head.state_dict(),channels=channels,evidence_channels=features[-1].shape[1],variant=variant,epoch=chosen,threshold=threshold),path.with_suffix('.pt'))
    restored=EvidenceHead(channels,variant,features[-1].shape[1]); restored.load_state_dict(torch.load(path.with_suffix('.pt'),weights_only=True)['state_dict'])
    np.testing.assert_allclose(cv.predict(restored,features,ids['validation']),p,atol=1e-7)
    np.savez_compressed(path.with_suffix('.npz'),probability=p,counterfactual=counter,target=target[ids['validation']].numpy(),names=split['validation'])
    path.write_text(json.dumps(row,indent=2),encoding='utf-8')
    print('RESULT',variant,method,fold+1,selected['miou'],'counterfactual',counter_metrics['miou'],flush=True)
    return row


def fit_blend(method,fold,ds,splits,evidence):
    path=OUT/'blend'/f'{method}_fold{fold+1}.json'
    if path.exists(): return json.loads(path.read_text())
    path.parent.mkdir(parents=True,exist_ok=True)
    map_row=fit('map_only',method,fold,ds,splits,evidence)
    baseline=json.loads((cv.OUT/'metrics.json').read_text())['methods'][method]['folds'][fold]
    cache=cv.model.extract(method,ds); names=cache['names']; features=cache['features']
    y=torch.from_numpy(np.stack([ds.m128[n] for n in names])).float()
    split=splits[method][fold]; ids={k:np.array([names.index(n) for n in v]) for k,v in split.items()}
    probs=[]
    for kind,epochs,expected in [('baseline',baseline['epoch'],baseline['inner_history'][baseline['epoch']-1]),
                                 ('map_only',map_row['epoch'],map_row['history'][map_row['epoch']-1])]:
        torch.manual_seed(42)
        head=cv.model.Head([f.shape[1] for f in features]) if kind=='baseline' else EvidenceHead([f.shape[1] for f in features],kind)
        inputs=features if kind=='baseline' else features+[evidence['real']]
        opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001); rng=np.random.default_rng(42)
        for _ in range(epochs): cv.train_epoch(head,opt,rng,inputs,y,ids['inner_train'])
        p=cv.predict(head,inputs,ids['inner_validation'])
        measured=cv.model.metrics(p,y[ids['inner_validation']].numpy(),.5)['miou']
        assert abs(measured-expected)<1e-7,(kind,measured,expected)
        probs.append(p)
    candidates=[]
    for weight in (0.,.25,.5,.75,1.):
        p=(1-weight)*probs[0]+weight*probs[1]
        for threshold in cv.model.CONFIG['threshold_candidates']:
            candidates.append((cv.model.metrics(p,y[ids['inner_validation']].numpy(),threshold)['miou'],weight,threshold))
    # Stable iteration order prefers lower evidence weight on exact ties.
    selected=max(candidates,key=lambda x:x[0]); _,weight,threshold=selected
    with np.load(cv.OUT/f'{method}_oof.npz') as z:
        lookup={n:i for i,n in enumerate(z['names'].tolist())}
        base=z['probability'][[lookup[n] for n in split['validation']]]
    with np.load(OUT/'map_only'/f'{method}_fold{fold+1}.npz') as z:
        assert z['names'].tolist()==split['validation']
        prob=(1-weight)*base+weight*z['probability']
        counter=(1-weight)*base+weight*z['counterfactual']
    row=dict(variant='blend',method=method,fold=fold+1,blend_weight=weight,inner_selection=candidates,
             epoch=map_row['epoch'],baseline_epoch=baseline['epoch'],threshold=threshold,
             selected=cv.model.metrics(prob,y[ids['validation']].numpy(),threshold),
             counterfactual=cv.model.metrics(counter,y[ids['validation']].numpy(),threshold),
             train_names=split['train'],val_names=split['validation'])
    np.savez_compressed(path.with_suffix('.npz'),probability=prob,counterfactual=counter,target=y[ids['validation']].numpy(),names=split['validation'])
    path.write_text(json.dumps(row,indent=2),encoding='utf-8')
    print('RESULT blend',method,fold+1,row['selected']['miou'],'weight',weight,'counter',row['counterfactual']['miou'],flush=True)
    return row


def publish(summary):
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
    subprocess.run([sys.executable,str(cv.model.ROOT/'dashboard/build.py')],check=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--methods',nargs='+',choices=cv.METHODS,default=list(cv.METHODS)); args=parser.parse_args()
    torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    ds=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    splits=json.loads((cv.OUT/'folds.json').read_text())
    baseline=json.loads((cv.OUT/'metrics.json').read_text())
    summary=dict(status='running',methods={},baseline=baseline,
                 gate='Per-method delta>=0.005; matched vs wrong-at-inference margin>=0.005; residual must also beat trained wrong control; best passing variant gets CV')
    for method in args.methods:
        evidence=prepare_bits(method,prepare(method))
        summary['methods'][method]=dict(kind=evidence['kind'],global_bit_accuracy=float(np.mean(evidence['global_bit_accuracy'])),
            projection_max_error=evidence['projection_max_error'],runs=[],cv=[],winner=None)
        data=summary['methods'][method]
        for variant in ('map_only','residual','residual_wrong','blend','bit_map','bit_residual','bit_residual_wrong'):
            data['runs'].append(fit(variant,method,0,ds,splits,evidence)); publish(summary)
        wrong=data['runs'][2]['selected']['miou']
        reference=baseline['methods'][method]['folds'][0]['selected']['miou']
        candidates=[]
        bit_wrong=next(r['selected']['miou'] for r in data['runs'] if r['variant']=='bit_residual_wrong')
        for row in [r for r in data['runs'] if not r['variant'].endswith('_wrong')]:
            score=row['selected']['miou']
            control_ok=(row['variant']!='residual' or score>wrong) and (row['variant']!='bit_residual' or score>bit_wrong)
            if score-reference>=.005 and score-row['counterfactual']['miou']>=.005 and control_ok:
                candidates.append((score,row['variant']))
        if candidates:
            data['winner']=max(candidates)[1]
            data['cv']=[next(r for r in data['runs'] if r['variant']==data['winner'])]
            for fold in (1,2): data['cv'].append(fit(data['winner'],method,fold,ds,splits,evidence))
        data['finished']=True; publish(summary)
        print('METHOD COMPLETE',method,'winner',data['winner'],flush=True)
    summary['status']='complete' if set(args.methods)==set(cv.METHODS) else 'partial'; publish(summary)


if __name__=='__main__': main()
