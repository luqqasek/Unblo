"""Image bootstrap and stratification of measured GT-selected proposal oracle masks."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'results'
base=np.load(ROOT/'experiments/local_presence/results/ps-data-small/maps-150.npz');d=json.loads((OUT/'proposal_screen.json').read_text());lookup={str(n):i for i,n in enumerate(base['names'])};rows=[]
def score(pred,gt):
 result=[]
 for cls in [False,True]:
  a=pred==cls;b=gt==cls;union=(a|b).sum();result.append(float((a&b).sum()/union) if union else 1.)
 return result
for item in d['per_image']:
 name=item['image_name'];i=lookup[name];gt=base['targets'][i]>.5;pred=base['probabilities'][i]>.5;oracle=pred.copy();masks=np.unpackbits(np.load(OUT/'proposals'/f'{name}.npz')['packed'],axis=1).reshape(-1,128,128).astype(bool)
 actions=item['best_pair_actions'] if item['oracle_two']>item['oracle_one']+1e-12 else ([item['best_single_action']] if item['oracle_one']>item['base']+1e-12 else [])
 for action in actions:oracle[masks[action//2]]=action%2==0
 bs=score(pred,gt);os=score(oracle,gt);assert abs(np.mean(bs)-item['base'])<1e-10;assert abs(np.mean(os)-item['oracle_two'])<1e-10
 area=float(gt.mean());minority=gt if area<=.5 else ~gt;yy,xx=np.where(minority);fill=float(minority.sum()/((yy.max()-yy.min()+1)*(xx.max()-xx.min()+1))) if len(yy) else 0
 a=min(area,1-area);bucket='le1' if a<=.01 else '1to5' if a<=.05 else '5to20' if a<=.2 else 'gt20'
 rows.append({'image_name':name,'regime':'foreground_islands' if area<=.5 else 'clean_holes','minority_area_bucket':bucket,'geometry_proxy':'rectangular' if fill>=.98 else 'irregular','base_iou0':bs[0],'base_iou1':bs[1],'oracle_iou0':os[0],'oracle_iou1':os[1],'base_miou':float(np.mean(bs)),'oracle_miou':float(np.mean(os))})
def summarize(part):
 if not part:return {'n':0}
 result={'n':len(part)}
 for key in ['base_iou0','base_iou1','oracle_iou0','oracle_iou1','base_miou','oracle_miou']:result[key]=float(np.mean([r[key] for r in part]))
 for arm in ['base','oracle']:
  result[arm+'_foreground_complete_misses']=sum(r[arm+'_iou1']==0 for r in part if r['regime']=='foreground_islands')
  result[arm+'_clean_complete_misses']=sum(r[arm+'_iou0']==0 for r in part if r['regime']=='clean_holes')
 delta=np.array([r['oracle_miou']-r['base_miou'] for r in part]);rng=np.random.default_rng(42);boot=delta[rng.integers(0,len(part),(10000,len(part)))].mean(1)
 result['oracle_delta']=float(delta.mean());result['paired_image_bootstrap95']=np.quantile(boot,[.025,.975]).tolist();return result
out={'protocol':'GT-selected <=2-action oracle, fixed57 PixelSeal. CI conditional on frozen model and fixed proposals; NOT a deployable PCMV gain.','all':summarize(rows),'by_regime_and_size':{reg+'_'+bucket:summarize([r for r in rows if r['regime']==reg and r['minority_area_bucket']==bucket]) for reg in ['foreground_islands','clean_holes'] for bucket in ['le1','1to5','5to20','gt20']},'by_geometry':{g:summarize([r for r in rows if r['geometry_proxy']==g]) for g in ['rectangular','irregular']},'per_image':rows}
(OUT/'mask_metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['all'],indent=2))
