"""Read-only reconstruction of the submitted legacy ensemble's saved OOF scores."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from wmloc.utils import miou_image
out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
result={}
for family,variant in [('pixelseal','n4k_s2500'),('trustmark','n4k_s5000')]:
 directory=ROOT/'experiments/synthetic_scaling/results/runs'/family;rows=[];maps=[];targets=[];names=[];thresholds=[]
 for fold in (1,2,3):
  zs=[np.load(directory/f'{variant}_fold{fold}_seed{s}.npz') for s in (42,7,123)]
  assert all(np.array_equal(z['names'],zs[0]['names']) and np.array_equal(z['target'],zs[0]['target']) for z in zs)
  probability=np.mean([z['probability'] for z in zs],axis=0);threshold=float(np.mean([z['threshold'] for z in zs]))
  maps.extend(probability);targets.extend(zs[0]['target']);names.extend(zs[0]['names']);thresholds.extend([threshold]*len(probability))
  for name,p,y in zip(zs[0]['names'],probability,zs[0]['target']):rows.append({'image_name':str(name),'miou':miou_image(p>=threshold,y),'fold':fold,'fg_fraction':float(y.mean())})
 np.savez_compressed(out/f'incumbent_{family}.npz',names=names,probabilities=np.stack(maps),targets=np.stack(targets),thresholds=thresholds)
 result[family]={'variant':variant,'miou':float(np.mean([r['miou'] for r in rows])),'rows':rows}
(out/'incumbent_oof.json').write_text(json.dumps(result,indent=2))
print(json.dumps({f:r['miou'] for f,r in result.items()}))
