"""GT-free proposal union and exact raw100-bit crop verification.

Image resizing is confined to the existing detector/decoder's input contract;
every verified crop is extracted again from original observed uint8 pixels.
Ground-truth diagnostics are deliberately a separate function called last.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import label,find_objects
from experiments.small_zoom.proposals import detector,boxmask
from experiments.local_presence.verification import verified
from experiments.native_localizer.common import rgb_tensor,atomic_json
from experiments.native_localizer.evaluate import image_metrics
from .evaluate import summarize
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

def starts(size,stride):return sorted(set(range(0,513-size,stride))|{512-size})
def valid(b):return 0<=b[0]<b[2]<=512 and 0<=b[1]<b[3]<=512 and min(b[2]-b[0],b[3]-b[1])>=8 and (b[2]-b[0])*(b[3]-b[1])<=.15*512**2
def iou(a,b):
    i=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return i/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
def soft_nms(rows,k=16):
    pending=[dict(r,box=[int(round(v)) for v in r['box']]) for r in rows if valid(r['box'])];kept=[]
    while pending and len(kept)<k:
        pending.sort(key=lambda r:r['score'],reverse=True);r=pending.pop(0)
        if r['score']<=0:break
        kept.append(r)
        for other in pending:other['score']*=np.exp(-iou(r['box'],other['box'])**2/.5)
        pending=[o for o in pending if iou(r['box'],o['box'])<.98]
    return kept
def local_proposals(prob):
    labels,n=label(prob>=.5);out=[]
    for idx,s in enumerate(find_objects(labels),1):
        if s is None:continue
        ys,xs=s;mask=labels[s]==idx;score=float(prob[s][mask].max());box=[xs.start*4,ys.start*4,xs.stop*4,ys.stop*4]
        if valid(box):out.append({'box':box,'score':score,'origin':'local_connected_component'})
    return sorted(out,key=lambda r:r['score'],reverse=True)[:8]
def neighbors(box,step):
    x,y,x1,y1=box;w=x1-x;h=y1-y;result={tuple(box)}
    for axis in range(4):
        for sign in (-1,1):
            p=[x,y,w,h];p[axis]+=step*sign;b=[p[0],p[1],p[0]+p[2],p[1]+p[3]]
            if valid(b):result.add(tuple(b))
    return sorted(result)

@torch.no_grad()
def score_crops(image,message,boxes,wm):
    x=rgb_tensor(image).cuda();bits=torch.as_tensor(message,device='cuda').bool();result=[]
    for start in range(0,len(boxes),64):
        bs=boxes[start:start+64]
        crops=torch.cat([F.interpolate(x[None,:,y:y1,xx:x1],(wm.resolution,wm.resolution),mode='bilinear',align_corners=False,antialias=True) for xx,y,x1,y1 in bs])
        logits=wm.logits(crops).float();count=((logits>0)==bits).sum(1);soft=((2*bits.float()-1)*torch.tanh(logits/2)).mean(1)
        for b,c,s in zip(bs,count.tolist(),soft.tolist()):result.append({'box':list(b),'correct_bits':c,'bit_accuracy':c/100,'soft_match':s})
    return result

@torch.no_grad()
def propose(image,local,old,det):
    x=rgb_tensor(image).cuda();rows=[dict(r,origin='existing_full_image') for r in old.get('raw_proposals',[])];tiles=[]
    for size,stride in ((256,128),(160,80)):
        tiles.extend((xx,y,size) for y in starts(size,stride) for xx in starts(size,stride))
    assert len(tiles)==45
    for begin in range(0,len(tiles),4):
        ts=tiles[begin:begin+4];inputs=[F.interpolate(x[None,:,y:y+s,xx:xx+s],(1024,1024),mode='bilinear',align_corners=False,antialias=True)[0]*2-1 for xx,y,s in ts]
        raw,_=det(inputs)
        for j,(xx,y,size) in enumerate(ts):
            for b,score in zip(raw['boxes'][j].cpu().tolist(),raw['scores'][j].cpu().tolist()):
                box=[round(b[0]*size/1024+xx),round(b[1]*size/1024+y),round(b[2]*size/1024+xx),round(b[3]*size/1024+y)]
                if valid(box):rows.append({'box':box,'score':score,'origin':'tiled','tile':[xx,y,size]})
    rows.extend(local_proposals(local));return rows

def refine(image,bits,rows,old,wm):
    seeds=[tuple(r['box']) for r in soft_nms(rows)];cache={}
    for scale in (8,4,2):
        groups=[neighbors(b,scale) for b in seeds];new=sorted({b for group in groups for b in group}-set(cache))
        for r in score_crops(image,bits,new,wm):cache[tuple(r['box'])]=r
        seeds=[max(group,key=lambda b:cache[b]['soft_match']) for group in groups]
    # Never discard a previously verified crop because a new proposal ranked above it.
    previous=old.get('correct_top',[])[:1]
    accepted=[r for r in previous if verified(r) and valid(r['box'])]
    accepted.extend(r for r in cache.values() if verified(r));accepted.sort(key=lambda r:(r['bit_accuracy'],r['soft_match']),reverse=True)
    # Conservative single best additional rectangle, plus the original accepted one.
    new=[r for r in accepted if r not in previous][:1]
    return [r for r in previous if verified(r) and valid(r['box'])]+new,list(cache.values())

def offline_audit(image,bits,gt,proposals,searched,accepted,wm):
    yy,xx=np.where(gt);box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    boxes=[]
    for dx in (-4,-2,0,2,4):
        for dy in (-4,-2,0,2,4):
            b=[max(0,box[0]+dx),max(0,box[1]+dy),min(512,box[2]+dx),min(512,box[3]+dy)]
            if b[2]>b[0] and b[3]>b[1]:boxes.append(b)
    exact=score_crops(image,bits,boxes,wm);best=max(r['correct_bits'] for r in exact);rawiou=max([iou(box,r['box']) for r in proposals],default=0)
    matching=[r for r in accepted if iou(box,r['box'])>=.25];fill=float(gt.mean()*512**2/((box[2]-box[0])*(box[3]-box[1])))
    failure='non_decodable_at_GT' if best<90 else 'missing_proposal' if rawiou<.25 else 'alignment_failure' if not matching else 'shape_mismatch' if fill<.98 else 'recovered_rectangle'
    return {'category':failure,'best_gt_jitter_correct_bits':best,'raw_proposal_best_bbox_iou':rawiou,'bbox_fill':fill,'offline_only':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--step',required=True);p.add_argument('--part',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;started=time.monotonic()
    det=detector(a.root);wm=load_watermark('trustmark',models_dir='checkpoints',device='cuda');ds=LocalisationDataset('data','train',methods=['trustmark']);lookup={r['image_name']:i for i,r in enumerate(ds.meta)}
    old={r['image_name']:r for r in json.loads((a.root/'small-zoom-finalscale/trustmark/proposals.json').read_text())}
    maps=np.load(a.run/f'maps-{a.step}.npz');out=a.run/'refined';out.mkdir(exist_ok=True);records=[];predictions=[];names=[];targets=[]
    for i in range(a.part,len(maps['names']),4):
        name=str(maps['names'][i]);item=ds[lookup[name]];raw=propose(item['image'],maps['local_probabilities'][i],old.get(name,{}),det)
        accepted,searched=refine(item['image'],item['message'],raw,old.get(name,{}),wm)
        prob=maps['raw_probabilities'][i].copy();veto=maps['delta_clean'][i]>2
        for r in accepted:prob[boxmask(r['box'])&~veto]=1
        y=item['mask128'];metric=image_metrics(prob,y);metric['image_name']=name
        audit=offline_audit(item['image'],item['message'],item['mask'].astype(bool),raw,searched,accepted,wm) if 0<y.mean()<=.05 else None
        row={'image_name':name,'raw_proposals':raw,'accepted':accepted,'search_count':len(searched),'metrics':metric,'offline_audit':audit};records.append(row);predictions.append(prob);names.append(name);targets.append(y)
        atomic_json(out/f'part{a.part}.json',{'records':records,'seconds':time.monotonic()-started});print('TM_REFINEMENT',name,metric['miou'],audit,flush=True)
    np.savez_compressed(out/f'part{a.part}.npz',names=names,probabilities=predictions,targets=targets)
if __name__=='__main__':main()
