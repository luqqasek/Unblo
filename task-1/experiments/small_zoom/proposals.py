"""Frozen public NRGMark proposals, scored by unchanged competition decoders.

No masks, edit metadata or GT sizes enter proposal generation or selection.
"""
import argparse, hashlib, importlib, json, sys, time, types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
from experiments.native_localizer.common import atomic_json

def boxmask(box):
    mask=np.zeros((512,512),np.uint8);x0,y0,x1,y1=map(round,box)
    mask[max(0,y0):min(512,y1),max(0,x0):min(512,x1)]=1
    return downsample_mask(mask).astype(bool)

def detector(root):
    code=root/'checkpoints/nrgmark/source';package=types.ModuleType('nrg_external')
    package.__path__=[str(code)];sys.modules['nrg_external']=package
    cls=importlib.import_module('nrg_external.yolo_wo_decoder_multibox').fasterRCNN
    model=cls()
    path=root/'checkpoints/nrgmark/trustmark_bbox_Q.ckpt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='7e1da5667a4da070573d71bbf5a03d586f8836397bbcdf2c2044865ec37df0a5'
    state=torch.load(path,map_location='cpu',weights_only=False)
    state=state.get('state_dict',state)
    weights={k[5:]:v for k,v in state.items() if k.startswith('yolo.')}
    model.load_state_dict(weights,strict=True)
    return model.cuda().eval().requires_grad_(False)

def boxes_to_search(boxes):
    candidates=set()
    for box in boxes:
        x0,y0,x1,y1=map(float,box);w=x1-x0;h=y1-y0
        # Focus this bounded correction on small regions, identically for every image.
        if w*h>512*512*.15 or min(w,h)<8:continue
        for mx in [-8,-4,0,4,8]:
            for my in [-8,-4,0,4,8]:
                for dx,dy in [(0,0),(-8,0),(8,0),(0,-8),(0,8)]:
                    b=(max(0,round(x0-mx+dx)),max(0,round(y0-my+dy)),min(512,round(x1+mx+dx)),min(512,round(y1+my+dy)))
                    if min(b[2]-b[0],b[3]-b[1])>=8 and (b[2]-b[0])*(b[3]-b[1])<=512*512*.10:candidates.add(b)
    return sorted(candidates)

@torch.inference_mode()
def infer(image,bits,name,det,wm,batch=64,scales=(512,)):
    x=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).cuda().float()/255
    boxes=[];rawrows=[]
    for scale in scales:
        scaled=x if scale==512 else F.interpolate(x[None],(scale,scale),mode='bilinear',align_corners=False,antialias=True)[0]
        raw,_=det([scaled*2-1]);scaled_boxes=(raw['boxes'][0]*512/scale).cpu().tolist();boxes+=scaled_boxes
        # Raw top-4 per scale; crop decoding always uses original source pixels.
        rawrows.extend({'box':b,'score':float(s),'label':int(l),'proposal_scale':scale} for b,s,l in zip(scaled_boxes,raw['scores'][0].cpu(),raw['labels'][0].cpu()))
    candidates=boxes_to_search(boxes)
    wrong=np.random.default_rng(int(name[:8],16)+23456).integers(0,2,len(bits))
    bits=torch.tensor(bits,device='cuda').bool();wrong=torch.tensor(wrong,device='cuda').bool()
    records=[];wrongrecords=[]
    for start in range(0,len(candidates),batch):
        bs=candidates[start:start+batch]
        crops=torch.cat([F.interpolate(x[None,:,y0:y1,x0:x1],(wm.resolution,wm.resolution),mode='bilinear',align_corners=False,antialias=True) for x0,y0,x1,y1 in bs])
        logits=wm.logits(crops).float()
        for query,rows in [(bits,records),(wrong,wrongrecords)]:
            acc=((logits>0)==query).float().mean(1).cpu().tolist()
            soft=(logits*(query.float()*2-1)).sigmoid().mean(1).cpu().tolist()
            rows.extend({'box':list(b),'bit_accuracy':a,'soft_match':s} for b,a,s in zip(bs,acc,soft))
    records.sort(key=lambda r:(r['bit_accuracy'],r['soft_match']),reverse=True)
    wrongrecords.sort(key=lambda r:(r['bit_accuracy'],r['soft_match']),reverse=True)
    return {'image_name':name,'raw_proposals':rawrows,'search_count':len(candidates),'correct_top':records[:10],'wrong_top':wrongrecords[:10]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--limit',type=int);p.add_argument('--multiscale',action='store_true');p.add_argument('--scales',nargs='+',type=int);p.add_argument('--run-name');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scales=tuple(a.scales) if a.scales else ((512,1536) if a.multiscale else (512,))
    out=a.root/(a.run_name or ('small-zoom-multiscale' if a.multiscale else 'small-zoom-runs'))/a.family;out.mkdir(parents=True,exist_ok=True)
    det=detector(a.root);wm=load_watermark(a.family,models_dir='checkpoints',device='cuda')
    ds=LocalisationDataset('data','train',methods=[a.family]);rows=[];start=time.monotonic()
    for i in range(min(len(ds),a.limit or len(ds))):
        item=ds[i]
        row=infer(item['image'],item['message'],item['image_name'],det,wm,scales=scales)
        rows.append(row)
        atomic_json(out/'proposals.json',rows)
        if i%10==0 or i+1==len(ds):print(json.dumps({'family':a.family,'images':i+1,'seconds':time.monotonic()-start,'last_candidates':row['search_count'],'last_best':row['correct_top'][:1]}),flush=True)
    atomic_json(out/'inference.json',{'seconds':time.monotonic()-start,'n':len(rows),'scales':scales,'peak_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'batch':64,'threshold_fixed':.9,'source':'Adobe trustmark commit 4c60b9fdb4125fca3001f0803e6890a6b55371ac','mask_inputs':False})
if __name__=='__main__':main()
