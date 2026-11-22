"""Freeze one SS candidate from completed OOF before any test prediction."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import fallback,scores
from experiments.native_localizer.common import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).parent/'results');p.add_argument('--cluster-root',type=Path,required=True);a=p.parse_args();r=a.results
    assert not (r/'selected_recipe.json').exists(),'Preserve the frozen selection'
    cv=json.loads((r/'ss-oof-150.json').read_text());base=json.loads((r/'ss-oof-0.json').read_text());assert len(cv['per_image'])==len({x['image_name'] for x in cv['per_image']})==170
    adapted=cv['paired_delta']['mean']>=.01 and cv['paired_delta']['ci95'][0]>0
    step=150 if adapted else 0;all_delta=[];per_image=[]
    for fold in (1,2,3):
        done=json.loads((r/f'stegastamp/fold{fold}/completed.json').read_text());assert done['step']==150 and done['stop']=='steps_complete'
        z=np.load(r/f'stegastamp/fold{fold}/maps-{step}.npz')
        for name,prob,gt in zip(z['names'],z['probabilities'],z['targets']):
            before=scores(prob>=.5,gt.astype(bool));after=scores(fallback(prob,'top1pct'),gt.astype(bool));delta=after['miou']-before['miou'];all_delta.append(delta);per_image.append({'image_name':str(name),'before':before,'after':after,'delta':delta})
    assert len(all_delta)==170;delta=np.array(all_delta);rng=np.random.default_rng(42);ci=np.quantile(delta[rng.integers(170,size=(5000,170))].mean(1),[.025,.975]);use_fallback=float(delta.mean())>0 and ci[0]>0
    checkpoint=a.cluster_root/('next-ideas-runs/stegastamp/all170/last.pt' if adapted else 'runs/stegastamp/pilot-30min/last.pt')
    recipe={'family':'stegastamp','recipe_frozen':True,'checkpoint':str(checkpoint),'needs_all170_refit':bool(adapted),'updates':150 if adapted else 200,'threshold':.5,'weights':'raw','tta':False,'precision':'FP32;TF32off','postprocess':'top1pct' if use_fallback else 'none','fallback_policy':'only empty/full masks;164 highest minority-probability cells;fixed before adaptation results','adaptation_selection':{'delta':cv['paired_delta']['mean'],'ci95':cv['paired_delta']['ci95'],'milestone':.01,'selected':bool(adapted)},'fallback_selection':{'mean_delta':float(delta.mean()),'ci95':ci.tolist(),'selected':bool(use_fallback)},'validation_miou_before_fallback':(cv if adapted else base)['summary']['all']['miou'],'validation_miou_final':(cv if adapted else base)['summary']['all']['miou']+(float(delta.mean()) if use_fallback else 0),'preserve_families':['pixelseal','trustmark'],'base_submission_id':2675,'selection_scope':'Reused development folds, exploratory postprocessing; not independent test estimate','test_labels_used':False,'evidence_sha256':sha256(r/'ss-oof-150.json')}
    assert adapted or use_fallback,'No measured positive candidate; retain previous submission'
    atomic_json(r/'selected_recipe.json',recipe);atomic_json(r/'selected_fallback_per_image.json',per_image);print(json.dumps(recipe,indent=2))


if __name__=='__main__':main()
