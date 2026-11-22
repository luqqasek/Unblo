"""Evaluate saved GT-independent proposals, fixed thresholds, all images."""
import argparse,json
from pathlib import Path
import numpy as np
from wmloc.utils import downsample_mask
from experiments.small_region_diagnosis.common import metadata,masks128,bucket,select,mean,save
from experiments.native_localizer.evaluate import image_metrics
from .proposals import boxes_to_search,boxmask

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,default=Path(__file__).resolve().parent/'results/runs');p.add_argument('--families',nargs='+',default=['pixelseal','trustmark']);a=p.parse_args()
    meta=metadata();gt=masks128();base=Path(__file__).resolve().parent.parent/'region_data/results/runs';allsummary={}
    for family,variant in [('pixelseal','control-adapt'),('trustmark','objects-adapt')]:
        if family not in a.families:continue
        proposals=json.loads((a.runs/family/'proposals.json').read_text());maps={};folds={}
        for fold in [1,2,3]:
            path=base/family/variant/f'fold{fold}'
            split=json.loads((path/'split.json').read_text());assert not set(split['validation'])&set(split['train'])
            with np.load(path/'maps.npz') as z:
                assert set(z['names'])==set(split['validation'])
                for n,prob,target in zip(z['names'],z['probabilities'],z['targets']):
                    assert np.array_equal(target.squeeze().astype(bool),gt[n]);maps[n]=prob.squeeze();folds[n]=fold
        assert len(proposals)==len({r['image_name'] for r in proposals})==170
        assert {r['image_name'] for r in proposals}==set(maps)
        rows={k:[] for k in ['baseline','fixed_crop90','wrong_message90']};diagnostic=[]
        for proposal in proposals:
            name=proposal['image_name'];y=gt[name];baseline=maps[name]>=.5
            candidate_masks=[boxmask(b) for b in boxes_to_search([r['box'] for r in proposal['raw_proposals']])]
            def iou(m):
                u=(m|y).sum();return float((m&y).sum()/u) if u else 1.
            d={'image_name':name,'true_area':float(y.mean()),'bucket':bucket(float(y.mean())),'full_watermark':meta[name]['full_watermark'],'proposal_oracle_iou':max([iou(m) for m in candidate_masks],default=0),'search_count':proposal['search_count']}
            diagnostic.append(d)
            for mode,key in [('baseline',None),('fixed_crop90','correct_top'),('wrong_message90','wrong_top')]:
                pred=baseline.copy();accepted=False
                if key and proposal[key] and proposal[key][0]['bit_accuracy']>=.9:
                    pred |= boxmask(proposal[key][0]['box']);accepted=True
                r=image_metrics(pred.astype(np.float32),y);r.update(image_name=name,bucket=d['bucket'],full_watermark=d['full_watermark'],fold=folds[name],accepted=accepted);rows[mode].append(r)
        result={}
        for mode,rr in rows.items():
            result[mode]={g:{'n':len(select(rr,g)),**{k:mean(select(rr,g),k) for k in ['miou','fg_iou','clean_iou','empty','accepted']}} for g in ['all','le1','1to5','5to20','gt20','clean_holes']}
            save(a.runs/family/f'{mode}_per_image.json',rr)
        result['proposal_diagnostics']={g:{'n':len(select(diagnostic,g)),'recall_iou50':float(np.mean([r['proposal_oracle_iou']>=.5 for r in select(diagnostic,g)])) if select(diagnostic,g) else None,'oracle_mask_iou':mean(select(diagnostic,g),'proposal_oracle_iou')} for g in ['all','le1','1to5']}
        d=np.array([b['miou']-a['miou'] for a,b in zip(rows['baseline'],rows['fixed_crop90'])]);rng=np.random.default_rng(42)
        result['paired_delta']={'mean':float(d.mean()),'ci95':np.quantile(d[rng.integers(0,len(d),(5000,len(d)))].mean(1),[.025,.975]).tolist()}
        allsummary[family]=result;save(a.runs/family/'diagnostic_per_image.json',diagnostic)
        print(family,json.dumps(result),flush=True)
    save(a.runs.parent/f'{a.runs.name}_summary.json',allsummary)
if __name__=='__main__':main()
