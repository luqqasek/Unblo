"""GT-free scan restricted by baseline EMPTY prediction, partitioned over two GPUs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .tiled import neighbors
from .train import summary
from experiments.small_zoom.proposals import boxmask
from experiments.native_localizer.common import atomic_json,rgb_tensor
from experiments.native_localizer.evaluate import image_metrics
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

def scan_boxes():
    boxes=[]
    for w in range(24,81,8):
        for h in range(24,81,8):
            if max(w,h)>2*min(w,h):continue
            for x in sorted(set(range(0,513-w,16))|{512-w}):
                for y in sorted(set(range(0,513-h,16))|{512-h}):boxes.append((x,y,x+w,y+h))
    return boxes

def crop_batch(x,boxes,size):
    """Bilinear upsampling with ROI-edge clamps; verify exact finalists separately."""
    b=torch.tensor(boxes,device=x.device,dtype=torch.float32);t=(torch.arange(size,device=x.device,dtype=torch.float32)+.5)/size
    xx=b[:,0,None]+t*(b[:,2]-b[:,0])[:,None]-.5;yy=b[:,1,None]+t*(b[:,3]-b[:,1])[:,None]-.5
    xx=torch.maximum(torch.minimum(xx,b[:,2,None]-1),b[:,0,None]);yy=torch.maximum(torch.minimum(yy,b[:,3,None]-1),b[:,1,None])
    grid=torch.stack([((xx+.5)/512*2-1)[:,None,:].expand(-1,size,-1),((yy+.5)/512*2-1)[:,:,None].expand(-1,-1,size)],-1)
    return F.grid_sample(x[None].expand(len(boxes),-1,-1,-1),grid,mode='bilinear',padding_mode='border',align_corners=False)

@torch.inference_mode()
def infer(image,bits,name,wm,boxes):
    x=rgb_tensor(image).cuda();b=torch.tensor(bits,device='cuda').float();wrong=torch.tensor(np.random.default_rng(int(name[:8],16)+4321).integers(0,2,len(bits)),device='cuda').float();best=[];wrongmax=0.;max_error=0.
    for start in range(0,len(boxes),256):
        bs=boxes[start:start+256];crops=crop_batch(x,bs,wm.resolution)
        if start==0:
            exact=torch.cat([F.interpolate(x[None,:,y0:y1,x0:x1],(wm.resolution,wm.resolution),mode='bilinear',align_corners=False,antialias=True) for x0,y0,x1,y1 in bs[:4]])
            max_error=float((exact-crops[:4]).abs().max());assert max_error<1e-4,max_error
        z=wm.logits(crops).float();soft=(torch.tanh(z/2)*(b*2-1)).mean(1);acc=((z>0)==b.bool()).float().mean(1);wa=((z>0)==wrong.bool()).float().mean(1);wrongmax=max(wrongmax,float(wa.max()))
        best.extend({'box':bs[j],'soft_match':float(soft[j]),'bit_accuracy':float(acc[j])} for j in soft.topk(min(32,len(bs))).indices.tolist());best=sorted(best,key=lambda r:r['soft_match'],reverse=True)[:32]
    cache={};seeds=[r['box'] for r in best]
    for step in [4,2,1]:
        groups=[neighbors(b,step) for b in seeds];new=sorted(set(b for g in groups for b in g)-set(cache))
        for start in range(0,len(new),64):
            bs=new[start:start+64];crops=torch.cat([F.interpolate(x[None,:,y0:y1,x0:x1],(wm.resolution,wm.resolution),mode='bilinear',align_corners=False,antialias=True) for x0,y0,x1,y1 in bs]);z=wm.logits(crops).float();soft=(torch.tanh(z/2)*(b*2-1)).mean(1);acc=((z>0)==b.bool()).float().mean(1)
            wa=((z>0)==wrong.bool()).float().mean(1);wrongmax=max(wrongmax,float(wa.max()))
            for box,s,aa in zip(bs,soft.tolist(),acc.tolist()):cache[box]={'box':box,'soft_match':s,'bit_accuracy':aa}
        seeds=[max(g,key=lambda box:cache[box]['soft_match']) for g in groups]
    exact_best=sorted(cache.values(),key=lambda r:(r['bit_accuracy'],r['soft_match']),reverse=True)
    return {'image_name':name,'correct_top':exact_best[:10],'scan_count':len(boxes),'refine_count':len(cache),'wrong_max_accuracy':wrongmax,'screening_resize_max_error':max_error,'final_verification':'exact original interpolate native245'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--part',type=int,choices=[0,1],required=True);a=p.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/f'local-presence-runs/tm-scan-{a.part}';out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists();started=time.monotonic()
    wm=load_watermark('trustmark',models_dir='checkpoints',device='cuda');ds=LocalisationDataset('data','train',methods=['trustmark']);old={r['image_name']:r for r in json.loads((a.root/'small-zoom-finalscale/trustmark/proposals.json').read_text())};base={}
    for fold in [1,2,3]:
        with np.load(a.root/f'region-runs/trustmark/objects-adapt/fold{fold}/maps.npz') as z:base.update({str(n):p.squeeze() for n,p in zip(z['names'],z['probabilities'])})
    eligible=[]
    for i in range(len(ds)):
        n=ds.meta[i]['image_name'];p=base[n].copy()
        if old[n]['correct_top'] and old[n]['correct_top'][0]['bit_accuracy']>=.9:p[boxmask(old[n]['correct_top'][0]['box'])]=1
        if not (p>=.5).any():eligible.append(i)
    selected=eligible[a.part::2];boxes=scan_boxes();atomic_json(out/'config.json',{'eligible_rule':'final #2409 baseline predicts EMPTY; no GT','all_eligible':[ds.meta[i]['image_name'] for i in eligible],'selected':[ds.meta[i]['image_name'] for i in selected],'part':a.part,'scan_count_per_image':len(boxes),'sizes':'24:8:80, aspect<=2','stride':16,'batch':256,'refine_steps':[4,2,1],'acceptance':.9})
    print('SCAN_CONFIG',a.part,len(selected),len(boxes),flush=True);rows=[]
    for j,i in enumerate(selected):
        it=ds[i];result=infer(it['image'],it['message'],it['image_name'],wm,boxes);rows.append(result);atomic_json(out/'proposals.json',rows)
        print('SCAN_PROGRESS',a.part,j+1,len(selected),'seconds',time.monotonic()-started,'name',it['image_name'],'best',result['correct_top'][:1],flush=True)
    atomic_json(out/'completed.json',{'seconds':time.monotonic()-started,'n':len(rows),'peak_bytes':torch.cuda.max_memory_allocated(),'accepted':sum(bool(r['correct_top'] and r['correct_top'][0]['bit_accuracy']>=.9) for r in rows),'wrong_accepts':sum(r['wrong_max_accuracy']>=.9 for r in rows)})
if __name__=='__main__':main()
