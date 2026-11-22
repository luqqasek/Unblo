"""Time-bounded decoder fine-tuning after analytical weight surgery; training-only selection."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.native_localizer.model import block
from experiments.h100_sprint.core import aligned,digest,loss_fn,training_loss,write_json
from experiments.h100_hour.h2_decode import expected_miou_decode,miou,report

class SafeChannelNorm(nn.Module):
    def __init__(self, original):
        super().__init__();self.norm=original.norm
    def forward(self,x):
        mean=x.mean(1,keepdim=True);var=(x-mean).square().mean(1,keepdim=True)
        return (x-mean)*torch.rsqrt(var+self.norm.eps)*self.norm.weight[None,:,None,None]+self.norm.bias[None,:,None,None]

@torch.inference_mode()
def evaluate(head,x,y,indices):
    gains=(1,2,4,8);scores=np.zeros(4)
    for i in indices:
        z=head(x[i:i+1])[:,0];p=(z*torch.tensor(gains,device='mps')[:,None,None]).sigmoid().cpu().numpy()
        scores+=miou(expected_miou_decode(p),np.broadcast_to(y[i],p.shape))/len(indices)
    return float(scores.max()),gains[int(scores.argmax())]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',type=float,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--feature-mix',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(192)
    cache=Path('outputs/manual-weight-features/pixelseal-prehead');source=Path('experiments/h100_sprint/results/tuned/pixelseal-best.pt');saved=torch.load(source,map_location='cpu',weights_only=False);original={k.removeprefix('tri.base.head.'):v.clone() for k,v in saved['ema'].items() if k.startswith('tri.base.head.')};source_hash=digest(source)
    train=np.load(cache/'train-features.npy',mmap_mode='r');labels=np.load(cache/'train-labels.npz');gt=labels['targets'];names=labels['names'];order=sorted(range(113),key=lambda i:hashlib.sha256(str(names[i]).encode()).hexdigest());fit=order[:85];inner=order[85:]
    x=torch.from_numpy(train.copy()).to('mps');y=torch.tensor(gt.astype(np.float32),device='mps');head=nn.Sequential(block(166,128),nn.Conv2d(128,1,1)).to('mps');head[0][1]=SafeChannelNorm(head[0][1]);head.load_state_dict(original)
    score,gain=evaluate(head,x,gt,inner);best=dict(score=score,gain=gain,recipe='control',step=0,state=copy.deepcopy(original));rows=[{k:v for k,v in best.items() if k!='state'}];rng=np.random.default_rng(192)
    write_json(a.out/'protocol.json',dict(method='Decoder-only gradient fine-tuning on cached166-channel features; source backbone frozen',recipes=['mixstyle','cutmix','combined'] if a.feature_mix else ['bce_dice','balanced_lovasz','hard_boundary'],max_updates_per_recipe=300,fit_images=85,inner_images=28,development_images=57,source_sha256=source_hash,deadline_unix=a.deadline,development_labels_used_for_fitting=False,cloud_cost_usd=0,submitted=False))
    updates=0
    recipes=('mixstyle','cutmix','combined') if a.feature_mix else ('bce_dice','balanced_lovasz','hard_boundary')
    for recipe in recipes:
        if time.time()>a.deadline-90:break
        head.load_state_dict(original);head.train();opt=torch.optim.AdamW(head.parameters(),lr=2e-4,weight_decay=.05)
        for step in range(1,301):
            if time.time()>a.deadline-65:break
            ids=rng.choice(fit,4,replace=True);xs=[];ys=[]
            for i in ids:
                if rng.random()<.5:
                    minority=0 if gt[i].mean()>.5 else 1;pos=np.argwhere(gt[i]==minority);cy,cx=pos[rng.integers(len(pos))];top=int(np.clip(cy-32,0,64));left=int(np.clip(cx-32,0,64))
                else:top,left=rng.integers(0,65,2)
                xs.append(x[i,:,top:top+64,left:left+64]);ys.append(y[i,top:top+64,left:left+64])
            xb=torch.stack(xs);yb=torch.stack(ys);rot=int(rng.integers(4));xb=torch.rot90(xb,rot,(-2,-1));yb=torch.rot90(yb,rot,(-2,-1))
            if rng.random()<.5:xb=xb.flip(-1);yb=yb.flip(-1)
            if recipe in ('mixstyle','combined'):
                xb=xb.clone();context=xb[:,128:160];mean=context.mean((2,3),keepdim=True);std=context.var((2,3),keepdim=True,unbiased=False).add(1e-6).sqrt();perm=torch.tensor(rng.permutation(4),device='mps');lam=float(rng.beta(.3,.3))
                xb[:,128:160]=(context-mean)/std*(lam*std+(1-lam)*std[perm])+lam*mean+(1-lam)*mean[perm]
            if recipe in ('cutmix','combined') and rng.random()<.75:
                xb=xb.clone();yb=yb.clone();perm=torch.tensor(rng.permutation(4),device='mps');side=int(rng.integers(16,49));top,left=rng.integers(0,65-side,2)
                xb[:,:,top:top+side,left:left+side]=xb[perm,:,top:top+side,left:left+side].clone();yb[:,top:top+side,left:left+side]=yb[perm,top:top+side,left:left+side].clone()
            xb=xb.contiguous();yb=yb.contiguous()
            z=head(F.dropout2d(xb,p=.05,training=True))[:,0].contiguous()
            if recipe=='bce_dice':loss=training_loss(z,yb,'bce_dice')
            else:
                loss=loss_fn(z,yb)
                if recipe=='hard_boundary':
                    dilation=F.max_pool2d(yb[:,None],3,1,1)[:,0];erosion=-F.max_pool2d(-yb[:,None],3,1,1)[:,0];edge=dilation-erosion
                    pixels=F.binary_cross_entropy_with_logits(z,yb,reduction='none');hard=pixels.flatten(1).topk(1024,dim=1).values.mean();boundary=(pixels*edge).sum()/edge.sum().clamp_min(1);loss=loss+.2*hard+.2*boundary
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();updates+=1
            if step in (50,100,200,300):
                head.eval();sc,ga=evaluate(head,x,gt,inner);row=dict(recipe=recipe,step=step,score=sc,gain=ga);rows.append(row);write_json(a.out/'training-selection.json',rows);print(json.dumps(row),flush=True)
                if sc>best['score']:best=dict(**row,state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()})
                head.train()
    selected={k:v for k,v in best.items() if k!='state'};write_json(a.out/'selection.json',dict(winner=selected,frozen_before_development=True,updates=updates));head.load_state_dict(best['state']);head.eval();head[-1].weight.data.mul_(best['gain']);head[-1].bias.data.mul_(best['gain'])
    valid=np.load(cache/'valid-features.npy',mmap_mode='r');vl=np.load(cache/'valid-labels.npz');prob=[]
    with torch.inference_mode():
        for i in range(57):
            if time.time()>a.deadline-3:raise TimeoutError('decoder development deadline')
            prob.append(head(torch.from_numpy(valid[i:i+1].copy()).to('mps')).sigmoid()[0,0].cpu().numpy())
    prob=np.stack(prob)
    with np.load('/tmp/watermark-sprint-real-preflight/pixelseal/validation_maps.npz') as z:current=aligned(z,vl['names'].tolist(),'current_prob')
    old=torch.from_numpy(np.load(cache/'valid-original-logits.npy')*4).sigmoid().numpy();base=expected_miou_decode(.7*current+.3*old);masks=expected_miou_decode(.7*current+.3*prob)
    r=report('decoder_gradient_refine',base,masks,vl['targets']);r.update(family='pixelseal',selected=selected,candidates_completed=len(rows)*4,updates=updates,protocol='Training-selected decoder fine-tune; reused57 development; identity view; fixed70/30 blend',finished_unix=time.time(),cloud_cost_usd=0,submitted=False)
    assert digest(source)==source_hash;changed=0
    for k,v in head.state_dict().items():changed+=int((saved['ema']['tri.base.head.'+k]!=v.cpu()).sum());saved['ema']['tri.base.head.'+k]=v.cpu()
    r['changed_parameters']=changed;saved['manual_edit']='decoder_gradient_refine';saved['selection']=selected;torch.save(saved,a.out/'candidate.pt');r['candidate_sha256']=digest(a.out/'candidate.pt');write_json(a.out/'result.json',r);np.savez_compressed(a.out/'development.npz',names=vl['names'],probabilities=prob,masks=masks,targets=vl['targets'],baseline_masks=base);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
