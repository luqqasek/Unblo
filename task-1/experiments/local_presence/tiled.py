"""Additional GT-free tiles and bounded soft-score coordinate search for TM."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.small_zoom.proposals import detector,boxmask
from experiments.native_localizer.common import atomic_json,rgb_tensor
from experiments.native_localizer.evaluate import image_metrics
from .train import summary
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset

def starts(size,stride):return sorted(set(range(0,513-size,stride))|{512-size})
def valid(box):
    x0,y0,x1,y1=box;return 0<=x0<x1<=512 and 0<=y0<y1<=512 and min(x1-x0,y1-y0)>=8 and (x1-x0)*(y1-y0)<=.10*512**2
def neighbors(box,step):
    x0,y0,x1,y1=box;out={tuple(box)}
    for axis in range(4):
        for sign in [-1,1]:
            b=list(box)
            if axis==0:b[0]+=sign*step;b[2]+=sign*step
            elif axis==1:b[1]+=sign*step;b[3]+=sign*step
            elif axis==2:b[0]-=sign*step;b[2]+=sign*step
            else:b[1]-=sign*step;b[3]+=sign*step
            b=[max(0,min(512,int(round(v)))) for v in b]
            if valid(b):out.add(tuple(b))
    return sorted(out)

@torch.inference_mode()
def infer(image,message,name,det,wm):
    x=rgb_tensor(image).cuda();tiles=[];origins=[]
    for size,stride in [(256,128),(160,80)]:
        for y in starts(size,stride):
            for xx in starts(size,stride):
                tiles.append((xx,y,size));origins.append((xx,y,size))
    raw=[]
    for i in range(0,len(tiles),4):
        batch=[F.interpolate(x[None,:,y:y+s,xx:xx+s],(1024,1024),mode='bilinear',align_corners=False,antialias=True)[0]*2-1 for xx,y,s in tiles[i:i+4]]
        result,_=det(batch)
        for j,(xx,y,s) in enumerate(origins[i:i+4]):
            for box,score,label in zip(result['boxes'][j].cpu().tolist(),result['scores'][j].cpu().tolist(),result['labels'][j].cpu().tolist()):
                if score<=0:continue
                b=[round(box[0]*s/1024+xx),round(box[1]*s/1024+y),round(box[2]*s/1024+xx),round(box[3]*s/1024+y)]
                if valid(b):raw.append({'box':b,'score':score,'label':label,'tile':[xx,y,s]})
    raw.sort(key=lambda r:r['score'],reverse=True);seeds=[]
    def biou(a,b):
        inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    for row in raw:
        if all(biou(row['box'],b)<.8 for b in seeds):seeds.append(tuple(row['box']))
        if len(seeds)>=16:break
    message=torch.tensor(message,device='cuda').float();wrong=torch.tensor(np.random.default_rng(int(name[:8],16)+7238).integers(0,2,len(message)),device='cuda').float();cache={}
    def score(boxes):
        new=sorted(set(boxes)-set(cache))
        for start in range(0,len(new),64):
            bs=new[start:start+64];crops=torch.cat([F.interpolate(x[None,:,y0:y1,x0:x1],(wm.resolution,wm.resolution),mode='bilinear',align_corners=False,antialias=True) for x0,y0,x1,y1 in bs])
            logits=wm.logits(crops).float();acc=((logits>0)==message.bool()).float().mean(1);soft=(torch.tanh(logits/2)*(message*2-1)).mean(1)
            wa=((logits>0)==wrong.bool()).float().mean(1);ws=(torch.tanh(logits/2)*(wrong*2-1)).mean(1)
            for b,a,s,v,w in zip(bs,acc.tolist(),soft.tolist(),wa.tolist(),ws.tolist()):cache[b]={'box':list(b),'bit_accuracy':a,'soft_match':s,'wrong_accuracy':v,'wrong_soft':w}
    score(seeds)
    for step in [8,4,2]:
        groups=[neighbors(b,step) for b in seeds];score([b for g in groups for b in g])
        seeds=[max(g,key=lambda b:cache[b]['soft_match']) for g in groups]
    records=sorted(cache.values(),key=lambda r:(r['bit_accuracy'],r['soft_match']),reverse=True)
    wrongrows=sorted(cache.values(),key=lambda r:(r['wrong_accuracy'],r['wrong_soft']),reverse=True)
    return {'image_name':name,'raw_proposals':raw,'searched_boxes':[list(b) for b in cache],'correct_top':records[:10],'wrong_top':wrongrows[:10],'search_count':len(cache)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'local-presence-runs/tm-tiled';out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    det=detector(a.root);wm=load_watermark('trustmark',models_dir='checkpoints',device='cuda');ds=LocalisationDataset('data','train',methods=['trustmark'])
    old={r['image_name']:r for r in json.loads((a.root/'small-zoom-finalscale/trustmark/proposals.json').read_text())};base={}
    for fold in [1,2,3]:
        with np.load(a.root/f'region-runs/trustmark/objects-adapt/fold{fold}/maps.npz') as z:base.update({str(n):p.squeeze() for n,p in zip(z['names'],z['probabilities'])})
    rows=[];baserows=[];proposals=[];diagnostics=[]
    for i in range(len(ds)):
        item=ds[i];name=item['image_name'];result=infer(item['image'],item['message'],name,det,wm);proposals.append(result)
        prob=base[name].copy()
        if old[name]['correct_top'] and old[name]['correct_top'][0]['bit_accuracy']>=.9:prob[boxmask(old[name]['correct_top'][0]['box'])]=1
        y=item['mask128'].astype(bool);br=image_metrics(prob,y);br['image_name']=name;baserows.append(br)
        accepted=bool(result['correct_top'] and result['correct_top'][0]['bit_accuracy']>=.9)
        if accepted:prob[boxmask(result['correct_top'][0]['box'])]=1
        rr=image_metrics(prob,y);rr.update(image_name=name,accepted=accepted);rows.append(rr)
        masks=[boxmask(b) for b in result['searched_boxes']]
        oracle=max([float((m&y).sum()/max(1,(m|y).sum())) for m in masks],default=0)
        diagnostics.append({'image_name':name,'true_area':float(y.mean()),'candidate_oracle_iou':oracle,'search_count':result['search_count'],'accepted':accepted,'wrong_accepted':bool(result['wrong_top'] and result['wrong_top'][0]['wrong_accuracy']>=.9)})
        if (i+1)%10==0 or i+1==len(ds):
            atomic_json(out/'proposals.json',proposals);atomic_json(out/'per_image.json',rows);atomic_json(out/'diagnostics.json',diagnostics)
            print('TILED_PROGRESS',i+1,'seconds',time.monotonic()-start,'new_accepts',sum(r['accepted'] for r in rows),flush=True)
    delta=np.array([r['miou']-b['miou'] for r,b in zip(rows,baserows)])
    result={'baseline':summary(baserows),'final':summary(rows),'delta_miou':float(delta.mean()),'improved':int((delta>1e-8).sum()),'worse':int((delta<-1e-8).sum()),'wrong_accepts':sum(r['wrong_accepted'] for r in diagnostics),'seconds':time.monotonic()-start,'peak_bytes':torch.cuda.max_memory_allocated(),'protocol':'Existing OOF + frozen additive proposals; development, not fresh independent test'}
    atomic_json(out/'completed.json',result);print('TILED_DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
