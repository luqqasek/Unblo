"""Bounded analytical nonlinear/spatial head surgery; no development-label fitting."""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.h100_sprint.core import aligned, digest, write_json
from experiments.h100_hour.h2_decode import expected_miou_decode, miou, report

GROUPS = {'normalized': [0,2], 'spatial': [0,1,2], 'nonlinear': [0,3,4], 'retrieval': [0,2,5]}

class SurgicalHead(nn.Module):
    def __init__(self, mean, scale, projection, prototypes, weight=None, original=None, old_network=None):
        super().__init__()
        for key,value in [('mean',mean),('scale',scale),('projection',projection),('prototypes',prototypes)]:
            self.register_buffer(key, torch.as_tensor(value,dtype=torch.float32))
        self.register_buffer('original',torch.as_tensor(original,dtype=torch.float32))
        self.old_network=old_network
        self.channels=int(self.mean.numel());self.dimension=4*self.channels+258
        self.weight=nn.Parameter(torch.zeros(self.dimension) if weight is None else torch.as_tensor(weight,dtype=torch.float32))
    def basis(self,h):
        z=((h-self.mean[None,:,None,None])/self.scale[None,:,None,None]).clamp(-8,8)
        c=((h-h.mean((2,3),keepdim=True))/self.scale[None,:,None,None]).clamp(-8,8)
        local=z-F.avg_pool2d(z,9,1,4,count_include_pad=False)
        square=z.square().clamp_max(16)/4
        elm=F.relu(F.conv2d(c,self.projection[:,:,None,None]))/2
        norm=F.normalize(c,dim=1)
        retrieval=(4*(F.conv2d(norm,self.prototypes[:,:,None,None])-1)).exp()
        old=self.old_network(h) if self.old_network is not None else F.conv2d(h,self.original[:-1,None,None,None].permute(1,0,2,3),self.original[-1:])
        return torch.cat([z,local,c,square,elm,retrieval,old,torch.ones_like(old)],1)
    def forward(self,h):
        return F.conv2d(self.basis(h),self.weight[None,:,None,None])


def positions(target,index,n=256):
    rng=np.random.default_rng(400+index)
    return [rng.choice(np.flatnonzero(target.reshape(-1)==v),min(n,int((target==v).sum())),replace=False) for v in (0,1)]

@torch.inference_mode()
def moments(head,features,targets,indices,deadline):
    grams=[np.zeros((head.dimension,head.dimension),np.float64) for _ in range(2)]
    cross=[np.zeros(head.dimension,np.float64) for _ in range(2)]
    for count,i in enumerate(indices):
        if time.time()>deadline: raise TimeoutError('moment deadline')
        h=torch.from_numpy(features[i:i+1].copy()).to('mps')
        basis=head.basis(h)[0].flatten(1)
        for value,ix in enumerate(positions(targets[i],i)):
            x=basis[:,torch.as_tensor(ix,device='mps')].T.cpu().numpy().astype(np.float64)
            gram=x.T@x/len(x); cr=(2*value-1)*x.mean(0)
            for rho in (0,1):
                mass=float((targets[i]==value).mean()) if rho==0 else .5
                grams[rho]+=mass*gram/len(indices); cross[rho]+=mass*cr/len(indices)
        if (count+1)%20==0: print(json.dumps({'phase':'moments','images':count+1,'total':len(indices)}),flush=True)
    return grams,cross

@torch.inference_mode()
def score(head,features,targets,indices,candidates,deadline):
    weight=torch.as_tensor(np.stack([c['weight'] for c in candidates]),dtype=torch.float32,device='mps')[:,:,None,None]
    scores=np.zeros(len(candidates),np.float64)
    for i in indices:
        if time.time()>deadline: raise TimeoutError('selection deadline')
        h=torch.from_numpy(features[i:i+1].copy()).to('mps')
        prob=F.conv2d(head.basis(h),weight).sigmoid()[0].cpu().numpy()
        scores+=miou(expected_miou_decode(prob),np.broadcast_to(targets[i],prob.shape))/len(indices)
    return scores


def solve(gram,cross,kind,ridge,channels=128):
    offsets=[0,channels,2*channels,3*channels,4*channels,4*channels+128,4*channels+256]
    ix=np.r_[np.concatenate([np.arange(offsets[g],offsets[g+1]) for g in GROUPS[kind]]),len(cross)-2,len(cross)-1]
    g=gram[np.ix_(ix,ix)]; scale=np.maximum(np.diag(g),1e-3)
    penalty=scale*ridge; penalty[-1]*=.01
    w=np.zeros(len(cross),np.float64)
    w[ix]=np.linalg.solve(g+np.diag(penalty),cross[ix])
    return w


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',type=float,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--family',default='pixelseal',choices=['pixelseal','trustmark']);ap.add_argument('--prehead',action='store_true');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    cache=Path('outputs/manual-weight-features')/(a.family+('-prehead' if a.prehead else '-mps'));meta=json.loads((cache/'manifest.json').read_text())
    features=np.load(cache/'train-features.npy',mmap_mode='r');channels=features.shape[1];labels=np.load(cache/'train-labels.npz');targets=labels['targets'];names=labels['names']
    order=sorted(range(113),key=lambda i:hashlib.sha256(str(names[i]).encode()).hexdigest());fit=order[:85];inner=order[85:]
    byarea=sorted(inner,key=lambda i:targets[i].mean());screen=[byarea[i] for i in np.linspace(0,27,12).round().astype(int)]
    samples=[];centroids=[]
    for i in fit:
        h=features[i].reshape(channels,-1);cs=[]
        for ix in positions(targets[i],i): samples.append(h[:,ix].T.copy());cs.append(h[:,ix].mean(1))
        centroids.append(cs)
    samples=np.concatenate(samples);mean=samples.mean(0);scale=np.maximum(samples.std(0),.1)
    proto=np.asarray(centroids)[np.linspace(0,84,64).round().astype(int)].reshape(128,channels)
    proto=(proto-np.asarray(centroids).mean((0,1)))/scale;proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-8)
    rng=np.random.default_rng(139);projection=rng.normal(size=(128,channels)).astype(np.float32)/math.sqrt(channels)
    old=np.load(cache/'original_head.npz');original=np.r_[old['weight'].reshape(-1),old['bias']]
    old_network=None
    if a.prehead:
        from experiments.native_localizer.model import block
        old_network=nn.Sequential(block(channels,128),nn.Conv2d(128,1,1))
        state=torch.load(meta['source_checkpoint'],map_location='cpu',weights_only=False)['ema']
        old_network.load_state_dict({k.removeprefix('tri.base.head.'):v for k,v in state.items() if k.startswith('tri.base.head.')},strict=True)
    head=SurgicalHead(mean,scale,projection,proto,original=original,old_network=old_network).to('mps').eval().requires_grad_(False)
    old_index=head.dimension-2
    write_json(a.out/'protocol.json',dict(deadline_unix=a.deadline,fit_images=85,inner_images=28,screen_images=12,
        method='Analytical replacement head with spatial, normalized, quadratic/random and prototype-retrieval features',
        selection='98 configurations screened on training12; top6 reranked on training28, then refit selected coefficients on training113',
        development='Reused57; identity view; fixed70/30 blend; no development labels in weights or model selection',
        source_sha256=meta['source_sha256'],cloud_cost_usd=0,submitted=False,
        references=['https://arxiv.org/abs/2207.09519','https://arxiv.org/abs/1801.06975']))
    grams,cross=moments(head,features,targets,fit,a.deadline-210)
    candidates=[]
    for gain in (1,4):
        w=np.zeros(head.dimension);w[old_index]=gain;candidates.append(dict(name=f'control_gain{gain}',kind='control',gain=gain,weight=w))
    for kind in GROUPS:
      for rho in (0,1):
       for ridge in (.01,.1,1):
        w=solve(grams[rho],cross[rho],kind,ridge,channels)
        for blend in (.5,1):
         for gain in (2,8):
          v=w*blend;v[old_index]+=1-blend;v*=gain
          candidates.append(dict(name=f'{kind}_rho{rho}_ridge{ridge}_blend{blend}_gain{gain}',kind=kind,rho=rho,ridge=ridge,blend=blend,gain=gain,weight=v))
    scores=score(head,features,targets,screen,candidates,a.deadline-150)
    rows=[dict(**{k:v for k,v in c.items() if k!='weight'},screen_miou=float(s)) for c,s in zip(candidates,scores)]
    write_json(a.out/'screen.json',rows);ix=np.argsort(-scores)[:6];finalists=[candidates[i] for i in ix]
    scores2=score(head,features,targets,inner,finalists,a.deadline-120)
    winner=finalists[int(scores2.argmax())]
    selection=dict(**{k:v for k,v in winner.items() if k!='weight'},inner_miou=float(scores2.max()))
    write_json(a.out/'selection.json',dict(winner=selection,finalists=[dict(name=c['name'],inner_miou=float(s)) for c,s in zip(finalists,scores2)],frozen_before_development=True))
    print(json.dumps({'selected':selection}),flush=True)
    if winner['kind']!='control':
        grams,cross=moments(head,features,targets,list(range(113)),a.deadline-55)
        w=solve(grams[winner['rho']],cross[winner['rho']],winner['kind'],winner['ridge'],channels)*winner['blend']
        w[old_index]+=1-winner['blend'];w*=winner['gain']
    else:w=winner['weight']
    head.weight.data.copy_(torch.tensor(w,dtype=torch.float32,device='mps'))
    valid=np.load(cache/'valid-features.npy',mmap_mode='r');vl=np.load(cache/'valid-labels.npz');assert not set(vl['names'])&set(names)
    probabilities=[]
    with torch.inference_mode():
      for i in range(57):
        if time.time()>a.deadline-5:raise TimeoutError('development deadline')
        probabilities.append(head(torch.from_numpy(valid[i:i+1].copy()).to('mps')).sigmoid()[0,0].cpu().numpy())
    probability=np.stack(probabilities)
    with np.load('/tmp/watermark-sprint-real-preflight/'+a.family+'/validation_maps.npz') as z:
        if 'current_prob' in z.files:
            current=aligned(z,vl['names'].tolist(),'current_prob')
        else:
            current=aligned(z,vl['names'].tolist(),'base_prob')
        corrections=aligned(z,vl['names'].tolist(),'corrections').astype(np.uint8)
        current_mask=aligned(z,vl['names'].tolist(),'current_mask')
    oldprob=torch.from_numpy(np.load(cache/'valid-original-logits.npy')*4).sigmoid().numpy()
    baseline=expected_miou_decode(.7*current+.3*oldprob) if a.family=='pixelseal' else expected_miou_decode(.8*current+.2*oldprob)|corrections
    masks=expected_miou_decode(.7*current+.3*probability) if a.family=='pixelseal' else expected_miou_decode(.8*current+.2*probability)|corrections
    result=report(selection['name'],baseline,masks,vl['targets']);result.update(family=a.family,selected=selection,candidates_completed=len(candidates),protocol='Analytical nonlinear/spatial replacement head; reused57 development; identity view; fixed blend '+('70/30' if a.family=='pixelseal' else '80/20'),cloud_cost_usd=0,submitted=False,finished_unix=time.time(),standalone_miou=float(miou(expected_miou_decode(probability),vl['targets']).mean()))
    result['retained_recipe_miou']=float(miou(current_mask,vl['targets']).mean())
    source=Path(meta['source_checkpoint']);assert digest(source)==meta['source_sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False)
    saved['replacement_head']={k:v.detach().cpu() for k,v in head.state_dict().items()};saved['manual_edit']=selection['name'];saved['replacement_module']='experiments.manual_weights.nonlinear_surgery.SurgicalHead';saved['replace_path']=meta['head_path']
    torch.save(saved,a.out/'candidate.pt');result['candidate_sha256']=digest(a.out/'candidate.pt')
    np.savez_compressed(a.out/'development.npz',names=vl['names'],probabilities=probability,masks=masks,targets=vl['targets'],baseline_masks=baseline)
    write_json(a.out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except TimeoutError as e:print(json.dumps({'stopped_at_deadline':str(e)}),flush=True)
