"""Generate strictly nested stacking inputs from existing frozen features, real images only.

For each outer fold: calibration head excludes meta-validation; training maps are
3-way cross-fitted within meta-training, excluding meta-validation entirely.
Final fusion fitting maps are separately 3-way cross-fitted within outer-training.
Each head selects its own epoch inside its training subset, then refits. All
outputs/checkpoints stay in this experiment. No feature extractor is retrained.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from common import *

class Head(nn.Module):
    # Exact architecture/state keys of frozen_extractor.run.Head, copied to avoid
    # imports with historical experiment output-directory side effects.
    def __init__(self,channels):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.Conv2d(c,32,1),nn.GroupNorm(8,32),nn.ReLU()) for c in channels])
        self.fuse=nn.Sequential(nn.Conv2d(32*len(channels),32,3,padding=1),nn.GroupNorm(8,32),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,1,1))
    def forward(self,features):
        x=torch.cat([F.interpolate(p(f),size=(32,32),mode='bilinear',align_corners=False) for p,f in zip(self.projections,features)],1)
        return F.interpolate(self.fuse(x),size=(128,128),mode='bilinear',align_corners=False)[:,0]

def scores(p,g):
    pred=p>=.5;g=g.astype(bool)
    tp=(pred&g).sum((1,2));fp=(pred&~g).sum((1,2));fn=(~pred&g).sum((1,2));tn=(~pred&~g).sum((1,2))
    return np.mean((np.divide(tp,tp+fp+fn,out=np.ones(len(tp)),where=tp+fp+fn!=0)+np.divide(tn,tn+fp+fn,out=np.ones(len(tp)),where=tn+fp+fn!=0))/2)

def fit_predict(stem,train,val,names,features,target,initial,forbidden):
    directory=OUT/'fusion/crossfit';directory.mkdir(parents=True,exist_ok=True);path=directory/f'{stem}.npz'
    train=np.asarray(train);val=np.asarray(val)
    assert not set(train)&set(val) and not set(train)&set(forbidden)
    expected=dict(train_names=[names[i] for i in train],validation_names=[names[i] for i in val],forbidden_names=[names[i] for i in forbidden])
    if path.exists():
        audit=json.loads(path.with_suffix('.json').read_text());assert all(audit[k]==v for k,v in expected.items())
        z=np.load(path);assert z['names'].tolist()==expected['validation_names'];return z['probability']
    def init():
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        head=Head([x.shape[1] for x in features]).cuda(0)
        if initial is not None:head.load_state_dict(initial,strict=True)
        return head,torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001),np.random.default_rng(42)
    def epoch(head,opt,rng,ids):
        head.train()
        order=rng.permutation(ids)
        for j in range(0,len(order),8):
            ix=order[j:j+8];opt.zero_grad(set_to_none=True)
            logits=head([x[ix] for x in features]);y=target[ix];p=logits.sigmoid()
            dice=1-((2*(p*y).sum((1,2))+1)/(p.sum((1,2))+y.sum((1,2))+1)).mean()
            loss=.5*F.binary_cross_entropy_with_logits(logits,y)+.5*dice
            loss.backward();opt.step()
    def predict(head,ids):
        head.eval()
        with torch.inference_mode():return torch.cat([head([x[ids[j:j+8]] for x in features]).sigmoid().cpu() for j in range(0,len(ids),8)]).numpy()
    start=time.perf_counter();perm=np.random.default_rng(314159).permutation(train);ncal=int(np.ceil(len(train)*.2));iv=perm[:ncal];it=perm[ncal:]
    head,opt,rng=init();best=-1;stale=0;history=[];truth=target[iv].cpu().numpy()
    for e in range(1,51):
        epoch(head,opt,rng,it);m=float(scores(predict(head,iv),truth));history.append(m)
        if m>best:best=m;best_epoch=e;stale=0
        else:stale+=1
        if stale>=8:break
    del head,opt
    head,opt,rng=init()
    for _ in range(best_epoch):epoch(head,opt,rng,train)
    p=predict(head,val)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},epoch=best_epoch,**expected),path.with_suffix('.pt'))
    np.savez_compressed(path,names=expected['validation_names'],probability=p)
    save(path.with_suffix('.json'),dict(**expected,epoch=best_epoch,epoch_selection_train=[names[i] for i in it],epoch_selection_validation=[names[i] for i in iv],history=history,seconds=time.perf_counter()-start))
    print(f'CROSSFIT {stem}: train={len(train)} val={len(val)} epoch={best_epoch} seconds={time.perf_counter()-start:.1f}',flush=True)
    return p

def main():
    gate=json.loads((OUT/'eval_detectivesam/gate.json').read_text());assert gate['pass']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    records,_,gt,splits,_=cohort(); lookup={r['image_id']:i for i,r in enumerate(records)}
    for family in METHODS:
        feature_path=HERE.parent/('stegastamp_stn_alignment/results/aligned_features.pt' if family=='stegastamp' else f'frozen_extractor/results/{family}_features.pt')
        z=torch.load(feature_path,map_location='cpu',weights_only=True);names=z['names'];features=[x.cuda(0) for x in z['features']]
        target=torch.from_numpy(np.stack([gt[lookup[n]] for n in names])).float().cuda(0)
        initial=torch.load(HERE.parent/'synthetic_pretrain_2k/results/pretrain_seed42.pt',map_location='cpu',weights_only=True)['state_dict'] if family=='stegastamp' else None
        for fold,s in enumerate(splits[family],1):
            output=OUT/'fusion'/f'{family}_fold{fold}_inputs.npz'
            if output.exists():continue
            train=np.array([names.index(n) for n in s['train']]);outer=np.array([names.index(n) for n in s['validation']])
            cal=np.array([names.index(n) for n in s['inner_validation']]);fit=np.array([names.index(n) for n in s['inner_train']])
            assert set(fit)|set(cal)==set(train) and not set(fit)&set(cal)
            cp=fit_predict(f'{family}_fold{fold}_cal',fit,cal,names,features,target,initial,outer)
            saved={}
            for label,ids,forbidden in [('fit',fit,np.r_[outer,cal]),('refit',train,outer)]:
                probs=np.zeros((len(ids),128,128),np.float32);seen=np.zeros(len(ids),int)
                for j,(tr,va) in enumerate(KFold(3,shuffle=True,random_state=20261121).split(ids)):
                    probs[va]=fit_predict(f'{family}_fold{fold}_{label}{j}',ids[tr],ids[va],names,features,target,initial,forbidden)
                    seen[va]+=1
                assert (seen==1).all();saved[label+'_names']=[names[i] for i in ids];saved[label+'_probability']=probs
            np.savez_compressed(output,**saved,cal_names=[names[i] for i in cal],cal_probability=cp)
            save(output.with_suffix('.json'),dict(family=family,fold=fold,feature_sha256=sha(feature_path),outer_validation=s['validation'],seed=42,
                 note='Crossfit head epochs selected only inside its own train, then fresh refit. Calibration-training maps also exclude meta-validation. STN and pretrained seed42 preserved; no new synthetic training.'))
        del features,target
    print('NESTED CROSSFITTED BASELINE INPUTS COMPLETE',flush=True)

if __name__=='__main__':main()
