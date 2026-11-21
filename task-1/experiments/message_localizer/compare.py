"""Paired image-level comparison of fixed recipes against deployed incumbent OOF."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.native_localizer.common import atomic_json
from experiments.native_localizer.evaluate import image_metrics,summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args();out=Path(__file__).resolve().parent/'results';results={}
 incumbent=json.loads((out/'incumbent_oof.json').read_text())
 for family in ('pixelseal','trustmark'):
  z=np.load(out/f'incumbent_{family}.npz');lookup={str(n):i for i,n in enumerate(z['names'])};base={r['image_name']:r['miou'] for r in incumbent[family]['rows']}
  results[family]={'incumbent_miou':incumbent[family]['miou'],'variants':{}}
  for variant,paths in [('synthetic_only',[a.runs/family/'matched-v1/real/maps.npz']),('adapted',[a.runs/family/f'matched-v1-adapt/fold{f}/maps.npz' for f in (1,2,3)])]:
   if not all(p.exists() for p in paths):continue
   rows=[];fused=[]
   for path in paths:
    q=np.load(path)
    for name,pred,target in zip(q['names'],q['probabilities'],q['targets']):
     name=str(name);idx=lookup[name];assert np.array_equal(target,z['targets'][idx])
     r=image_metrics(pred,target,.5);r['image_name']=name;rows.append(r)
     f=image_metrics((pred+z['probabilities'][idx])/2,target,float((.5+z['thresholds'][idx])/2));f['image_name']=name;fused.append(f)
   assert len(rows)==len({r['image_name'] for r in rows})==170
   for label,rr in [(variant,rows),(variant+'_fixed_equal_ensemble',fused)]:
    summary=summarize(rr);delta=np.array([r['miou']-base[r['image_name']] for r in rr]);rng=np.random.default_rng(42)
    boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
    results[family]['variants'][label]={'miou':summary['groups']['all']['miou']['mean'],'delta_vs_incumbent':float(delta.mean()),'paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'improved_images':int((delta>0).sum()),'groups':summary['groups']}
    atomic_json(out/f'{family}_{label}_per_image.json',rr)
 atomic_json(out/'comparison.json',results)
 for family,d in results.items():
  print(family,'incumbent',d['incumbent_miou'])
  for v,r in d['variants'].items():print(v,{k:r[k] for k in ('miou','delta_vs_incumbent','paired_bootstrap95','improved_images')})
if __name__=='__main__':main()
