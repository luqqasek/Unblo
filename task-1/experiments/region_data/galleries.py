"""Representative real failures and paired comparisons, no test images."""
import json
from pathlib import Path
import numpy as np
from wmloc.data import LocalisationDataset
from experiments.native_localizer.evaluate import gallery,image_metrics
from experiments.native_localizer.common import atomic_json
P=Path(__file__).resolve().parent;R=P/'results'

def load_maps(folder):
 out={}
 for p in sorted(folder.glob('fold*/maps.npz')):
  with np.load(p) as z:
   out.update({str(n):v.copy() for n,v in zip(z['names'],z['probabilities'])})
 return out

def main():
 manifest=[]
 for family in ['pixelseal','trustmark']:
  base=load_maps(R/'runs'/family/'control-adapt');new=load_maps(R/'runs'/family/'objects-adapt')
  if len(base)!=170 or len(new)!=170:continue
  ds=LocalisationDataset(P.parents[1]/'data','train',methods=[family]);lookup={m['image_name']:i for i,m in enumerate(ds.meta)}
  rows=json.loads((R/'runs'/family/'objects-adapt/per_image.json').read_text())
  choices=[]
  for label,predicate,sortkey in [('fg_le1',lambda r:0<r['true_area']<=.01,lambda r:r['true_area']),('fg_1to5',lambda r:.01<r['true_area']<=.05,lambda r:r['fg_iou']),('clean_le1',lambda r:.99<=r['true_area']<1,lambda r:r['clean_iou']),('clean_1to5',lambda r:.95<=r['true_area']<.99,lambda r:r['clean_iou'])]:
   selected=sorted([r for r in rows if predicate(r)],key=sortkey)[:2]
   choices.extend((label,r) for r in selected)
  for i,(label,r) in enumerate(choices):
   name=r['image_name'];item=ds[lookup[name]];path=R/'galleries'/f'example_{family}_{i}_{label}.png';path.parent.mkdir(exist_ok=True)
   gallery(path,item['image'],item['mask128'],new[name],baseline=base[name]);manifest.append(dict(family=family,image_name=name,stratum=label,path=str(path.relative_to(R)),new_metrics=image_metrics(new[name],item['mask128']),control_metrics=image_metrics(base[name],item['mask128'])))
 atomic_json(R/'gallery_manifest.json',manifest);print('Gallery panels',len(manifest))
if __name__=='__main__':main()
