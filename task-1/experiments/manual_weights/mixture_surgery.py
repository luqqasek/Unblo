"""Analytical two-expert head: partial marks versus clean holes, learned image router."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.manual_weights.nonlinear_surgery import SurgicalHead,moments,solve
from experiments.h100_sprint.core import aligned,digest,write_json
from experiments.h100_hour.h2_decode import expected_miou_decode,miou,report

class MixtureHead(nn.Module):
    def __init__(self,core,experts,router_mean,router_scale,router,temperature=1,blend=1,hard=False):
        super().__init__();self.core=core
        for k,v in [('experts',experts),('router_mean',router_mean),('router_scale',router_scale),('router',router)]:self.register_buffer(k,torch.as_tensor(v,dtype=torch.float32))
        self.temperature=temperature;self.blend=blend;self.hard=hard
    @staticmethod
    def image_features(basis,channels):
        z=basis[:,:channels];p=basis[:,-2:-1].sigmoid()
        return torch.cat([z.mean((2,3)),z.std((2,3)),p.mean((2,3)),p.std((2,3)),p.amin((2,3)),p.amax((2,3))],1)
    def forward(self,h):
        basis=self.core.basis(h);f=self.image_features(basis,self.core.channels)
        gate=((f-self.router_mean)/self.router_scale)@self.router[:-1]+self.router[-1]
        gate=(gate>0).float() if self.hard else (gate*self.temperature).sigmoid()
        z=F.conv2d(basis,self.experts[:,:,None,None]);g=gate[:,None,None,None]
        return self.blend*((1-g)*z[:,0:1]+g*z[:,1:2])+(1-self.blend)*basis[:,-2:-1]*4

def fit_router(x,y,fit):
    mean=x[fit].mean(0);scale=np.maximum(x[fit].std(0),.1);z=np.c_[(x[fit]-mean)/scale,np.ones(len(fit))]
    penalty=np.full(z.shape[1],len(fit));penalty[-1]*=.01
    w=np.linalg.solve(z.T@z+np.diag(penalty),z.T@(2*y[fit]-1))
    return mean,scale,w

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    cache=Path('outputs/manual-weight-features/pixelseal-mps');meta=json.loads((cache/'manifest.json').read_text());data=np.load(cache/'train-features.npy',mmap_mode='r');labels=np.load(cache/'train-labels.npz');gt=labels['targets'];names=labels['names'];mode=(gt.mean((1,2))>.5).astype(float)
    source=Path('experiments/manual_weights/results/ten-minute-nonlinear/candidate.pt');saved=torch.load(source,map_location='cpu',weights_only=False);s=saved['replacement_head'];core=SurgicalHead(s['mean'],s['scale'],s['projection'],s['prototypes'],original=s['original']).to('mps').eval()
    order=sorted(range(113),key=lambda i:hashlib.sha256(str(names[i]).encode()).hexdigest());fit=order[:85];inner=order[85:];byarea=sorted(inner,key=lambda i:gt[i].mean());screen=[byarea[i] for i in np.linspace(0,27,12).round().astype(int)]
    write_json(a.out/'protocol.json',dict(method='Two analytical mask-regime experts with a ridge-fitted image router; soft/hard routing selected within training',fit_images=85,inner_images=28,deadline_unix=a.deadline,source_sha256=meta['source_sha256'],development_labels_used_for_fitting=False,submitted=False))
    xf=[]
    for i in range(113):
        if time.time()>a.deadline-160:raise TimeoutError('router features')
        basis=core.basis(torch.from_numpy(data[i:i+1].copy()).to('mps'));xf.append(MixtureHead.image_features(basis,128).cpu().numpy()[0])
    xf=np.asarray(xf);rm,rs,rw=fit_router(xf,mode,fit);routing=np.c_[(xf-rm)/rs,np.ones(113)]@rw
    all_m=[]
    for m in (0,1):all_m.append(moments(core,data,gt,[i for i in fit if mode[i]==m],a.deadline-150))
    pairs=[]
    for kind in ('spatial','nonlinear','retrieval'):
      for rho in (0,1):
       for ridge in (.01,.1,1):
        for gain in (2,8):
         weights=np.stack([solve(g[rho],c[rho],kind,ridge)*gain for g,c in all_m]);pairs.append(dict(kind=kind,rho=rho,ridge=ridge,gain=gain,weights=weights))
    candidates=[dict(pair=k,temperature=t,hard=hard,blend=b) for k in range(len(pairs)) for t,hard in ((1,False),(4,False),(1,True)) for b in (.5,1)]
    weights=torch.tensor(np.stack([x['weights'] for x in pairs]).reshape(-1,770),dtype=torch.float32,device='mps')[:,:,None,None]
    def evaluate(indices,ids,deadline):
        scores=np.zeros(len(ids));
        for i in indices:
            if time.time()>deadline:raise TimeoutError('mixture selection')
            basis=core.basis(torch.from_numpy(data[i:i+1].copy()).to('mps'));pred=F.conv2d(basis,weights)[0].cpu().numpy().reshape(len(pairs),2,128,128);old=basis[0,-2].cpu().numpy()*4;prob=[]
            for k in ids:
                c=candidates[k];gate=float(routing[i]>0) if c['hard'] else 1/(1+np.exp(-np.clip(routing[i]*c['temperature'],-50,50)))
                z=c['blend']*((1-gate)*pred[c['pair'],0]+gate*pred[c['pair'],1])+(1-c['blend'])*old
                prob.append(1/(1+np.exp(-np.clip(z,-50,50))))
            prob=np.stack(prob);scores+=miou(expected_miou_decode(prob),np.broadcast_to(gt[i],prob.shape))/len(indices)
        return scores
    score=evaluate(screen,list(range(len(candidates))),a.deadline-115);top=np.argsort(-score)[:6];rerank=evaluate(inner,top,a.deadline-95);winner=candidates[int(top[rerank.argmax()])];pair=pairs[winner['pair']]
    selection={**winner,**{k:v for k,v in pair.items() if k!='weights'},'inner_miou':float(rerank.max())};write_json(a.out/'selection.json',dict(winner=selection,router_inner_accuracy=float(((routing[inner]>0)==mode[inner]).mean()),frozen_before_development=True));print(json.dumps({'mixture_selected':selection}),flush=True)
    final=[]
    for m in (0,1):
        g,c=moments(core,data,gt,[i for i in range(113) if mode[i]==m],a.deadline-55);final.append(solve(g[pair['rho']],c[pair['rho']],pair['kind'],pair['ridge'])*pair['gain'])
    rm,rs,rw=fit_router(xf,mode,list(range(113)))
    model=MixtureHead(core,np.stack(final),rm,rs,rw,winner['temperature'],winner['blend'],winner['hard']).to('mps').eval();valid=np.load(cache/'valid-features.npy',mmap_mode='r');vl=np.load(cache/'valid-labels.npz');probs=[]
    for i in range(57):
        if time.time()>a.deadline-5:raise TimeoutError('mixture development')
        probs.append(model(torch.from_numpy(valid[i:i+1].copy()).to('mps')).sigmoid()[0,0].cpu().numpy())
    probs=np.stack(probs)
    with np.load('/tmp/watermark-sprint-real-preflight/pixelseal/validation_maps.npz') as z:current=aligned(z,vl['names'].tolist(),'current_prob')
    old=torch.from_numpy(np.load(cache/'valid-original-logits.npy')*4).sigmoid().numpy();base=expected_miou_decode(.7*current+.3*old);masks=expected_miou_decode(.7*current+.3*probs)
    r=report('regime_mixture',base,masks,vl['targets']);r.update(family='pixelseal',selected=selection,candidates_completed=len(candidates),protocol='Training-selected two-expert architecture; reused57 development; identity view; fixed70/30 blend',finished_unix=time.time(),submitted=False,cloud_cost_usd=0)
    assert digest(Path(meta['source_checkpoint']))==meta['source_sha256'];saved=torch.load(meta['source_checkpoint'],map_location='cpu',weights_only=False);saved.update(replacement_head={k:v.cpu() for k,v in model.state_dict().items()},replacement_module='experiments.manual_weights.mixture_surgery.MixtureHead',replace_path='tri.base.head.1',selection=selection);torch.save(saved,a.out/'candidate.pt');r['candidate_sha256']=digest(a.out/'candidate.pt');write_json(a.out/'result.json',r);np.savez_compressed(a.out/'development.npz',names=vl['names'],probabilities=probs,masks=masks,targets=vl['targets'],baseline_masks=base);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
