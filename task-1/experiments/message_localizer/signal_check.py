"""Exploratory real-data signal audit, correct versus shuffled messages; no training."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from experiments.native_localizer.common import atomic_json,rgb_tensor
from .extractor import Extractor,match_maps,resize

def blur(x,k=9):return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='reflect'),k,1)
def correlation(a,b):
    num=blur((a*b).sum(1,keepdim=True),17)
    den=(blur(a.square().sum(1,keepdim=True),17)*blur(b.square().sum(1,keepdim=True),17)).sqrt().clamp_min(1e-9)
    return resize(num/den)[0,0]

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=170);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ds=LocalisationDataset('data','train',methods=[a.family]); ex=Extractor(a.family).eval().requires_grad_(False)
 wm=load_watermark(a.family,models_dir='checkpoints',device='cuda')
 a.out.mkdir(parents=True,exist_ok=True); allrows=[]; started=time.monotonic()
 for i in range(min(a.limit,len(ds))):
  item=ds[i]; path=a.out/f'{item["image_name"]}.npz'
  if path.exists():continue
  x=rgb_tensor(item['image'])[None].cuda(); b=torch.tensor(item['message'],device='cuda')[None].float()
  wrong=torch.tensor(ds.meta[(i+1)%len(ds)]['message'] and [int(c) for c in ds.meta[(i+1)%len(ds)]['message']],device='cuda')[None].float()
  maps={}; labels=item['mask128'].ravel(); hp=x-blur(x)
  with torch.no_grad():
   for size in (ex.native,384,512):
    _,z=ex(x,size)
    for query,msg in [('correct',b),('wrong',wrong)]:
     values=resize(match_maps(z,msg))[0]
     for k,name in enumerate(('dot','cos','acc')):maps[f'dense_{size}_{name}_{query}']=values[k]
   # Verify dense pre-pooling logits preserve the original wrapper exactly.
   if i==0:
    original=wm.logits(x);_,z=ex(x,ex.native)
    error=float((z.mean((-2,-1))-original).abs().max())
    if not torch.allclose(z.mean((-2,-1)),original,atol=1e-4,rtol=1e-4):raise RuntimeError(f'Logit mismatch {error}')
    atomic_json(a.out/'forward_check.json',{'max_abs_error':error,'native_size':ex.native,'nbits':ex.nbits})
   for side,stride in [(128,64),(256,128)]:
    coords=[(y,x0) for y in range(0,513-side,stride) for x0 in range(0,513-side,stride)]
    sums=torch.zeros(2,512,512,device='cuda');counts=torch.zeros(512,512,device='cuda')
    for st in range(0,len(coords),8):
     selected=coords[st:st+8];patch=torch.cat([x[:,:,y:y+side,xx:xx+side] for y,xx in selected]);_,z=ex(patch,ex.native);logits=z.mean((-2,-1))
     vals=torch.stack([((logits>0)==(msg>0.5)).float().mean(1) for msg in (b,wrong)])
     for j,(y,xx) in enumerate(selected):sums[:,y:y+side,xx:xx+side]+=vals[:,j,None,None];counts[y:y+side,xx:xx+side]+=1
    for q,name in enumerate(('correct','wrong')):maps[f'window_{side}_{name}']=resize((sums[q]/counts)[None,None])[0,0]
   # Occlusion evidence is descriptive, not assumed causally clean.
   _,z=ex(x,ex.native);base=match_maps(z,b)[:,0].mean()
   occlusion=torch.zeros(1,1,512,512,device='cuda')
   coords=[(y,xx) for y in range(0,512,128) for xx in range(0,512,128)]
   for st in range(0,16,8):
    batch=x.repeat(8,1,1,1)
    for j,(y,xx) in enumerate(coords[st:st+8]):batch[j,:,y:y+128,xx:xx+128]=x[:,:,y:y+128,xx:xx+128].mean((-2,-1))[0,:,None,None]
    _,z=ex(batch,ex.native);values=base-match_maps(z,b)[:,0].mean((-2,-1))
    for j,(y,xx) in enumerate(coords[st:st+8]):occlusion[:,:,y:y+128,xx:xx+128]=values[j]
   maps['occlusion_correct']=resize(occlusion)[0,0]
  for query,msg in [('correct',b),('wrong',wrong)]:
   xx=x.detach().requires_grad_(True);_,z=ex(xx,ex.native)
   score=(z.mean((-2,-1))*(2*msg-1)).mean();g=torch.autograd.grad(score,xx)[0]
   # RGB directional derivative; equal RGB channel displacement corresponds to Y at fixed chroma.
   prod=(hp*g).sum(1,keepdim=True) if a.family!='pixelseal' else (hp*torch.tensor([.299,.587,.114],device='cuda')[None,:,None,None]).sum(1,keepdim=True)*g.sum(1,keepdim=True)
   maps[f'gradient_{query}']=F.avg_pool2d(prod,4)[0,0].detach()
  for query,msg in [('correct',b),('wrong',wrong)]:
   bits=msg[0].cpu().int().tolist();r1=wm.embed(item['image'],bits);r0=wm.embed(item['image'],[1-v for v in bits])
   residual=rgb_tensor(r1.astype(np.float32)-r0.astype(np.float32))[None].cuda()
   maps[f'reembed_{query}']=correlation(hp,residual)
  arrays={k:v.detach().cpu().numpy().astype(np.float32) for k,v in maps.items()}
  np.savez_compressed(path,**arrays,target=item['mask128'],image_name=item['image_name'])
  if (i+1)%10==0:print(json.dumps({'family':a.family,'images':i+1,'seconds':time.monotonic()-started}),flush=True)
 # Summaries do not claim diagnostic threshold calibration as independent validation.
 for path in sorted(a.out.glob('*.npz')):
  z=np.load(path);y=z['target'].ravel()
  for name in z.files:
   if name in ('target','image_name'):continue
   score=z[name].ravel();allrows.append({'image_name':str(z['image_name']),'map':name,'auc':float(roc_auc_score(y,score)),'ap':float(average_precision_score(y,score))})
 summary={name:{'n':sum(r['map']==name for r in allrows),'macro_auc':float(np.mean([r['auc'] for r in allrows if r['map']==name])),'macro_ap':float(np.mean([r['ap'] for r in allrows if r['map']==name]))} for name in sorted({r['map'] for r in allrows})}
 atomic_json(a.out/'summary.json',{'family':a.family,'seconds':time.monotonic()-started,'maps':summary,'protocol':'exploratory fixed maps on real development data; no model training or threshold selection'})
 atomic_json(a.out/'per_image.json',allrows)
 print('SIGNAL_RESULT',json.dumps(summary),flush=True)
if __name__=='__main__':main()
