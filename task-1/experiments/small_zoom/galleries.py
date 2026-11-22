import json
from pathlib import Path
import numpy as np
from wmloc.data import LocalisationDataset
from experiments.native_localizer.evaluate import gallery
from .evaluate_proposals import boxmask

def main():
    here=Path(__file__).resolve().parent;run=here/'results/finalscale/trustmark';base=here.parent/'region_data/results/runs/trustmark/objects-adapt'
    p={}
    for f in [1,2,3]:
        with np.load(base/f'fold{f}/maps.npz') as z:p.update({str(n):v.squeeze() for n,v in zip(z['names'],z['probabilities'])})
    proposals={r['image_name']:r for r in json.loads((run/'proposals.json').read_text())};rows=json.loads((run/'fixed_crop90_per_image.json').read_text());baseline={r['image_name']:r for r in json.loads((run/'baseline_per_image.json').read_text())}
    improved=sorted(rows,key=lambda r:r['miou']-baseline[r['image_name']]['miou'],reverse=True)[:4]
    recovered_tiny=[r for r in rows if 0<r['true_area']<=.01 and r['fg_iou']>0]
    failed=[r for r in rows if 0<r['true_area']<=.01 and r['fg_iou']==0][:3]+[r for r in rows if .01<r['true_area']<=.05 and r['fg_iou']==0][:3]
    ds=LocalisationDataset(here.parents[1]/'data','train',methods=['trustmark']);idx={r['image_name']:i for i,r in enumerate(ds.meta)};out=here/'results/galleries';out.mkdir(parents=True,exist_ok=True);manifest=[]
    for j,r in enumerate(improved+recovered_tiny+failed):
        n=r['image_name'];item=ds[idx[n]];prob=p[n].copy();cand=proposals[n]['correct_top']
        if cand and cand[0]['bit_accuracy']>=.9:prob[boxmask(cand[0]['box'])]=1
        gallery(out/f'example_{j}_{n}.png',item['image'],item['mask128'],prob,p[n]);(here/f'results/trustmark_example_{j:02d}.png').write_bytes((out/f'example_{j}_{n}.png').read_bytes());manifest.append({'image_name':n,'baseline':baseline[n],'corrected':r,'candidate':cand[:1]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('galleries',len(manifest))
if __name__=='__main__':main()
