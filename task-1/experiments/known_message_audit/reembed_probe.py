"""Counterfactual re-embedding NCC with true and two random messages."""
import argparse
import hashlib
import json
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from probe import ROOT, OUT, save, metrics, load_watermark


def local_mean(x, window):
    left=(window-1)//2
    right=window-1-left
    return F.avg_pool2d(F.pad(x,(left,right,left,right),mode='reflect'),window,stride=1)


def ncc(a,b,window):
    ma,mb=local_mean(a,window),local_mean(b,window)
    covariance=(local_mean(a*b,window)-ma*mb).sum(1,keepdim=True)
    va=(local_mean(a*a,window)-ma*ma).clamp_min(0).sum(1,keepdim=True)
    vb=(local_mean(b*b,window)-mb*mb).clamp_min(0).sum(1,keepdim=True)
    return F.avg_pool2d(covariance/(va*vb).sqrt().clamp_min(1e-8),4)[:,0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--methods',nargs='+',default=['pixelseal','trustmark','stegastamp'])
    parser.add_argument('--limit',type=int,default=170)
    args=parser.parse_args()
    torch.set_num_threads(2)
    metadata=json.loads((ROOT/'data/train/metadata.json').read_text())
    z=np.load(ROOT/'data/train/masks_128.npz')
    targets=dict(zip(z['names'],z['masks']))
    for method in args.methods:
        started=time.monotonic()
        # SS stays on official ONNX CPU path; arithmetic and PyTorch models use GPU.
        wm=load_watermark(method,ROOT/'checkpoints','cpu' if method=='stegastamp' else 'cuda:0')
        rows=sorted((m for m in metadata if m['method']==method),key=lambda m:m['image_name'])[:args.limit]
        for i,meta in enumerate(rows):
            name=meta['image_name']; dest=OUT/'reembed'/method/f'{name}.json'
            if dest.exists():continue
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little'))
            bits=np.array([int(c) for c in meta['message']])
            messages=[bits,rng.integers(0,2,len(bits)),rng.integers(0,2,len(bits))]
            rgb=np.array(Image.open(ROOT/'data/train/images'/f'{name}.png').convert('RGB'))
            residuals=[]
            for message in messages:
                positive=wm.embed(rgb,message).astype(np.float32)/255
                negative=wm.embed(rgb,1-message).astype(np.float32)/255
                residuals.append(positive-negative)
            residual=torch.tensor(np.stack(residuals),device='cuda:0').permute(0,3,1,2)
            x=torch.tensor(rgb,device='cuda:0').permute(2,0,1)[None].float()/255
            hp=x-gaussian_blur(x,[19,19],[3.,3.])
            if method=='pixelseal':
                weights=torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]
                residual=(residual*weights).sum(1,keepdim=True)
                hp=(hp*weights).sum(1,keepdim=True)
            maps={}
            for window in (16,32):
                scores=ncc(hp,residual,window).cpu().numpy()
                maps[f'ncc{window}']=scores[0]
                maps[f'ncc{window}_null']=scores[1:].mean(0)
                maps[f'ncc{window}_calibrated']=scores[0]-scores[1:].mean(0)
            assert all(np.isfinite(v).all() for v in maps.values())
            target=targets[name]
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest.with_suffix('.npz'),**maps)
            save(dest,dict(name=name,method=method,fg_fraction=float(target.mean()),
                           full_watermark=bool(meta['full_watermark']),metrics=metrics(maps,target)))
            if (i+1)%10==0 or i==0:
                print(method,'reembed',i+1,'/',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
        del wm
        torch.cuda.empty_cache()


if __name__=='__main__':main()
