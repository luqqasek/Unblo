"""Matched residual fusion pilot; true/wrong message, strict inner selection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from probe import ROOT,OUT,save,dense,map_statistics,resize_maps,load_watermark

sys.path.insert(0,str(ROOT/'experiments/synthetic_scaling'))
import train as scaling


class Correction(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(7,8,1),nn.ReLU(),nn.Conv2d(8,1,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,maps,base):
        return base+self.net(torch.cat([maps,base[:,None]],1))[:,0]


def evidence(method,names):
    path=OUT/'fusion'/f'{method}_maps.pt'
    if path.exists():
        c=torch.load(path,weights_only=True);assert c['names']==names;return c
    wm=load_watermark(method,ROOT/'checkpoints','cuda:0')
    meta={m['image_name']:m for m in json.loads((ROOT/'data/train/metadata.json').read_text())}
    real=[];wrong=[]
    with torch.no_grad():
        for name in names:
            bits=np.array([int(c) for c in meta[name]['message']])
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little'))
            signs=torch.tensor(2*np.stack([bits,rng.integers(0,2,len(bits))])-1,device='cuda:0',dtype=torch.float32)
            rgb=np.array(Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB'))
            x=torch.from_numpy(rgb).permute(2,0,1)[None].float().to('cuda:0')/255
            channels=[]
            for scale in (wm.resolution,512):
                stats=map_statistics(dense(wm,x,scale),signs)
                channels.extend([resize_maps(v[:,None])[:,0] for v in
                                 (stats['cos'],stats['acc'],stats['norm'].expand(2,-1,-1))])
            maps=torch.stack(channels,1).cpu()
            real.append(maps[0]);wrong.append(maps[1])
    result=dict(names=names,real=torch.stack(real),wrong=torch.stack(wrong))
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(result,path)
    del wm;torch.cuda.empty_cache()
    return result


def normalize(maps,ids):
    mean=maps[ids].mean((0,2,3),keepdim=True)
    std=maps[ids].std((0,2,3),keepdim=True).clamp_min(.05)
    return (maps-mean)/std,mean,std


def initialize():
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    head=Correction().cuda()
    return head,torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001),np.random.default_rng(42)


def epoch(head,opt,rng,maps,base,target,ids):
    head.train()
    order=rng.permutation(ids)
    for start in range(0,len(order),8):
        b=order[start:start+8];opt.zero_grad(set_to_none=True)
        logits=head(maps[b],base[b]);p=logits.sigmoid();y=target[b]
        dice=1-((2*(p*y).sum((1,2))+1)/(p.sum((1,2))+y.sum((1,2))+1)).mean()
        loss=.5*F.binary_cross_entropy_with_logits(logits,y)+.5*dice
        loss.backward();opt.step()


@torch.no_grad()
def predict(head,maps,base,ids):
    head.eval()
    return torch.cat([head(maps[ids[j:j+8]],base[ids[j:j+8]]).sigmoid() for j in range(0,len(ids),8)])


def baseline_logits(data,state):
    head,_,_=scaling.initialize(data,42,state)
    p=scaling.predict(head,data,np.arange(len(data.names))).clamp(1e-6,1-1e-6)
    return torch.logit(p)


def fit_fold(method,data,cache,fold):
    path=OUT/'fusion'/f'{method}_fold{fold}.json'
    if path.exists():return json.loads(path.read_text())
    variant={'pixelseal':'n4k_s2500','trustmark':'n4k_s5000'}[method]
    source=ROOT/'experiments/synthetic_scaling/results/runs'/method
    split=scaling.splits(method)[fold-1]
    idx={n:i for i,n in enumerate(data.names)}
    ids={k:np.array([idx[n] for n in split[k]]) for k in ('train','validation','inner_train','inner_validation')}
    assert not set(ids['train'])&set(ids['validation'])
    assert not set(ids['inner_train'])&set(ids['inner_validation'])
    original=json.loads((source/f'{variant}_fold{fold}_seed42.json').read_text())
    pre=torch.load(source/f'{variant}_seed42_pretrain.pt',weights_only=True)['state_dict']
    head,opt,rng=scaling.initialize(data,42,pre)
    for _ in range(original['epoch']):scaling.epoch(head,opt,rng,data,ids['inner_train'])
    inner_prob=scaling.predict(head,data,np.arange(len(data.names))).clamp(1e-6,1-1e-6)
    inner_base=torch.logit(inner_prob)
    reproduced=scaling.gpu_score(inner_prob[ids['inner_validation']],data.target[ids['inner_validation']],.5)
    expected=original['inner_history'][original['epoch']-1]['miou']
    assert abs(reproduced-expected)<1e-6,(reproduced,expected)
    del head,opt
    outer_state=torch.load(source/f'{variant}_fold{fold}_seed42.pt',weights_only=True)['state_dict']
    outer_base=baseline_logits(data,outer_state)
    old=np.load(source/f'{variant}_fold{fold}_seed42.npz')
    np.testing.assert_allclose(outer_base[ids['validation']].sigmoid().cpu().numpy(),old['probability'],rtol=1e-5,atol=1e-6)
    result=dict(method=method,fold=fold,baseline_miou=original['miou'],baseline_threshold=original['threshold'],
                inner_reproduction_error=abs(reproduced-expected),variants={})
    for label in ('real','wrong'):
        raw=cache[label].cuda()
        normalized,_,_=normalize(raw,ids['inner_train'])
        head,opt,rng=initialize()
        best=scaling.gpu_score(inner_prob[ids['inner_validation']],data.target[ids['inner_validation']],.5)
        best_p=inner_prob[ids['inner_validation']].clone();selected=0;stale=0;history=[]
        for e in range(1,31):
            epoch(head,opt,rng,normalized,inner_base,data.target,ids['inner_train'])
            p=predict(head,normalized,inner_base,ids['inner_validation'])
            score=scaling.gpu_score(p,data.target[ids['inner_validation']],.5)
            history.append(dict(epoch=e,miou=score))
            if score>best:best,best_p,selected,stale=score,p.clone(),e,0
            else:stale+=1
            if stale>=8:break
        threshold=max((.3,.4,.5,.6,.7),key=lambda t:scaling.gpu_score(best_p,data.target[ids['inner_validation']],t))
        normalized,mean,std=normalize(raw,ids['train'])
        head,opt,rng=initialize()
        for _ in range(selected):epoch(head,opt,rng,normalized,outer_base,data.target,ids['train'])
        prob=predict(head,normalized,outer_base,ids['validation'])
        counter=(cache['wrong' if label=='real' else 'real'].cuda()-mean)/std
        counter_prob=predict(head,counter,outer_base,ids['validation'])
        miou=scaling.gpu_score(prob,data.target[ids['validation']],threshold)
        counter_miou=scaling.gpu_score(counter_prob,data.target[ids['validation']],threshold)
        result['variants'][label]=dict(miou=miou,counterfactual_miou=counter_miou,epoch=selected,threshold=threshold,inner_history=history)
        dest=path.with_name(f'{method}_fold{fold}_{label}')
        torch.save(dict(state_dict=scaling.state_cpu(head),mean=mean.cpu(),std=std.cpu(),epoch=selected,threshold=threshold,
                        train_names=split['train'],validation_names=split['validation']),dest.with_suffix('.pt'))
        np.savez_compressed(dest.with_suffix('.npz'),probability=prob.cpu().numpy(),counterfactual=counter_prob.cpu().numpy(),
                            target=data.target[ids['validation']].cpu().numpy(),names=split['validation'],threshold=threshold)
        print(method,'fold',fold,label,'baseline',original['miou'],'fusion',miou,'counter',counter_miou,'epoch',selected,flush=True)
    save(path,result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+',default=['pixelseal','trustmark'])
    args=parser.parse_args();scaling.setup('cuda:0')
    for method in args.methods:
        data=scaling.load_real(method,'cuda:0');cache=evidence(method,data.names)
        first=fit_fold(method,data,cache,1)
        real=first['variants']['real']['miou'];wrong=first['variants']['wrong']['miou']
        passed=real-first['baseline_miou']>=.005 and real-wrong>=.005
        records=[first]
        if passed:
            records += [fit_fold(method,data,cache,fold) for fold in (2,3)]
        save(OUT/'fusion'/f'{method}_summary.json',dict(method=method,screen_passed=passed,folds=records))
        del data,cache;torch.cuda.empty_cache()


if __name__=='__main__':main()
