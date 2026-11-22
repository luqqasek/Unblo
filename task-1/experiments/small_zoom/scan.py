"""GT-free bounded coarse/fine crop search seeded by OOF probabilities and a grid."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from experiments.native_localizer.common import atomic_json
from .proposals import boxes_to_search

def coarse_boxes(prob):
    score=gaussian_filter(prob.astype(float),2);centers=[]
    for _ in range(16):
        y,x=np.unravel_index(score.argmax(),score.shape);centers.append((4*x+2,4*y+2));score[max(0,y-10):y+11,max(0,x-10):x+11]=-1
    centers += [(x,y) for x in [64,192,320,448] for y in [64,192,320,448]]
    boxes=set()
    for cx,cy in centers:
        for w in [32,48,64,80,96,112,128,144]:
            for h in [32,48,64,80,96,112,128,144]:
                if max(w,h)>2*min(w,h):continue
                x=max(0,min(512-w,cx-w//2));y=max(0,min(512-h,cy-h//2));boxes.add((x,y,x+w,y+h))
    return sorted(boxes)

@torch.inference_mode()
def decode_candidates(x,boxes,wm,queries,batch=96):
    scores=[]
    for start in range(0,len(boxes),batch):
        crops=torch.cat([F.interpolate(x[None,:,y0:y1,x0:x1],(wm.resolution,wm.resolution),mode='bilinear',align_corners=False,antialias=True) for x0,y0,x1,y1 in boxes[start:start+batch]])
        z=wm.logits(crops).float();acc=((z[:,None]>0)==queries[None]).float().mean(2);soft=(z[:,None]*(queries.float()[None]*2-1)).sigmoid().mean(2)
        scores.append(torch.stack([acc,soft],-1).cpu().numpy())
    return np.concatenate(scores)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=root/'small-scan-runs/trustmark';out.mkdir(parents=True,exist_ok=True);maps={}
    for fold in [1,2,3]:
        with np.load(root/f'region-runs/trustmark/objects-adapt/fold{fold}/maps.npz') as z:
            maps.update({str(n):p.squeeze() for n,p in zip(z['names'],z['probabilities'])})
    wm=load_watermark('trustmark',models_dir='checkpoints',device='cuda');ds=LocalisationDataset('data','train',methods=['trustmark'])
    rows=json.loads((out/'proposals.json').read_text()) if (out/'proposals.json').exists() else [];done={r['image_name'] for r in rows};start=time.monotonic()
    for i in range(len(ds)):
        item=ds[i];name=item['image_name']
        if name in done:continue
        x=torch.tensor(np.array(item['image'],copy=True)).permute(2,0,1).cuda().float()/255
        wrong=np.random.default_rng(int(name[:8],16)+23456).integers(0,2,100);queries=torch.tensor(np.stack([item['message'],wrong]),device='cuda').bool()
        boxes=coarse_boxes(maps[name]);scores=decode_candidates(x,boxes,wm,queries)
        # Use both correct and random-query maxima to make the null search comparable.
        chosen=set()
        for q in range(2):chosen.update(np.lexsort((scores[:,q,1],scores[:,q,0]))[-8:].tolist())
        refined=sorted(set(boxes_to_search([boxes[j] for j in chosen]))-set(boxes))
        if refined:scores=np.concatenate([scores,decode_candidates(x,refined,wm,queries)]);boxes+=refined
        row={'image_name':name,'search_count':len(boxes),'raw_proposals':[],'coarse_centers':'16 nonmaximum-suppressed probability maxima +4x4 grid'}
        for q,key in [(0,'correct_top'),(1,'wrong_top')]:
            order=np.lexsort((scores[:,q,1],scores[:,q,0]))[-10:][::-1];row[key]=[{'box':list(boxes[j]),'bit_accuracy':float(scores[j,q,0]),'soft_match':float(scores[j,q,1])} for j in order]
        rows.append(row);atomic_json(out/'proposals.json',rows)
        if (i+1)%5==0 or i==0:print(json.dumps({'images':i+1,'seconds':time.monotonic()-start,'candidates':len(boxes),'best':row['correct_top'][0]}),flush=True)
    atomic_json(out/'inference.json',{'seconds_this_invocation':time.monotonic()-start,'n':len(rows),'peak_bytes':torch.cuda.max_memory_allocated(),'batch':96,'threshold_fixed':.9,'GT_inputs':False})
if __name__=='__main__':main()
