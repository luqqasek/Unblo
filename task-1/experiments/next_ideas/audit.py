"""A1/A5: honest error accounting from existing label-disjoint predictions."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label,uniform_filter
from experiments.native_localizer.common import atomic_json
from experiments.native_localizer.evaluate import image_metrics
from experiments.triforensic_wm.evaluate import summarize
from experiments.small_zoom.proposals import boxmask
from experiments.local_presence.verification import verified
from wmloc.data import LocalisationDataset


def load_maps(task,family):
    result={};folds={}
    if family=='stegastamp':
        paths=[task/'outputs/native_localizer/artifacts/runs/stegastamp/pilot-30min/real_validation_all/maps.npz']
        protocol='unchanged synthetic-only production checkpoint;170 real images independent of training'
    else:
        variant='control-adapt' if family=='pixelseal' else 'objects-adapt'
        paths=[task/f'experiments/region_data/results/runs/{family}/{variant}/fold{k}/maps.npz' for k in (1,2,3)]
        protocol='historical #2409 three-fold OOF;newer #2675 PS continuation has only a57-image screen' if family=='pixelseal' else '#2675 base three-fold OOF plus unchanged GT-free crop corrections'
    for fold,path in enumerate(paths,1):
        z=np.load(path)
        for n,p,y in zip(z['names'],z['probabilities'],z['targets']):assert str(n) not in result;result[str(n)]=(p.squeeze().copy(),y.squeeze().astype(bool));folds[str(n)]=fold
    assert len(result)==170
    if family=='trustmark':
        paths=[task/'experiments/small_zoom/results/finalscale/trustmark/proposals.json']+[task/f'experiments/local_presence/results/tm-scan-{k}/proposals.json' for k in (0,1)]
        for path in paths:
            for r in json.loads(path.read_text()):
                top=r.get('correct_top',[])
                if top and verified(top[0]):result[r['image_name']][0][boxmask(top[0]['box'])]=1
    return result,protocol


def scores(pred,gt):
    values=[]
    for c in (0,1):
        a,b=pred==c,gt==c;u=np.logical_or(a,b).sum();values.append(float(np.logical_and(a,b).sum()/u) if u else 1.)
    return {'clean_iou':values[0],'fg_iou':values[1],'miou':sum(values)/2}


def fallback(p,mode):
    m=p>=.5
    if m.any() and not m.all():return m
    evidence=1-p if m.all() else p;region=np.zeros_like(m)
    if mode=='top1pct':region.ravel()[np.argsort(evidence.ravel(),kind='stable')[-164:]]=True
    else:
        # Fixed13x13 contiguous region (~1%); rank using probabilities, never GT.
        sums=uniform_filter(evidence.astype(float),size=13,mode='constant');sums[:6]=-np.inf;sums[-6:]=-np.inf;sums[:,:6]=-np.inf;sums[:,-6:]=-np.inf
        y,x=np.unravel_index(np.argmax(sums),sums.shape);region[y-6:y+7,x-6:x+7]=True
    return ~region if m.all() else region


def main():
    task=Path(__file__).resolve().parents[2];out=Path(__file__).parent/'results';out.mkdir(exist_ok=True);allrows=[];summaries={};fallbacks={};cache={}
    for family in ('pixelseal','trustmark','stegastamp'):
        maps,protocol=load_maps(task,family);ds=LocalisationDataset(task/'data','train',methods=[family]);rows=[];fallback_rows={k:[] for k in ('top1pct','best13box')}
        for i in range(len(ds)):
            it=ds[i];name=it['image_name'];p,y=maps[name];assert np.array_equal(y,it['mask128']);m=it['mask'].astype(bool);yy,xx=np.where(m);rect=len(yy)>0 and m.sum()==(yy.max()-yy.min()+1)*(xx.max()-xx.min()+1)
            components,n=label(~y);border=np.unique(np.concatenate([components[0],components[-1],components[:,0],components[:,-1]]));enclosed=np.isin(components,[k for k in range(1,n+1) if k not in border])
            kind='restored_full_canvas' if it['full_watermark'] else 'rectangular_support' if rect else 'fragment_support'
            r=image_metrics(p,y);r.update(image_name=name,family=family,shape_proxy=kind,full_watermark=bool(it['full_watermark']),exact_rectangle=bool(rect),enclosed_clean_area=float(enclosed.mean()),loss_clean=(1-r['clean_iou'])/2,loss_wm=(1-r['fg_iou'])/2)
            rows.append(r);allrows.append(r)
            for mode in fallback_rows:
                s=scores(fallback(p,mode),y);s.update(image_name=name,delta=s['miou']-r['miou'],changed=bool(r['empty'] or r['full']),original_empty=bool(r['empty']),original_full=bool(r['full']));fallback_rows[mode].append(s)
        expected={'pixelseal':.5856553376906618,'trustmark':.6570460378911762,'stegastamp':.6949617064018395}
        assert abs(np.mean([r['miou'] for r in rows])-expected[family])<1e-10,'Reconstructed baseline does not match accepted recipe'
        cache[family]=(maps,rows)
        table=[]
        for kind in ('restored_full_canvas','rectangular_support','fragment_support'):
            rr=[r for r in rows if r['shape_proxy']==kind];table.append({'family':family,'proxy':kind,'n':len(rr),'miou':float(np.mean([r['miou'] for r in rr])) if rr else None,'mean_clean_loss':float(np.mean([r['loss_clean'] for r in rr])) if rr else None,'mean_wm_loss':float(np.mean([r['loss_wm'] for r in rr])) if rr else None,'contribution_to_510_error_clean':sum(r['loss_clean'] for r in rr)/510,'contribution_to_510_error_wm':sum(r['loss_wm'] for r in rr)/510})
        summaries[family]={'protocol':protocol,'metrics':summarize(rows),'error_budget':table,'both_classes_GT':sum(0<r['true_area']<1 for r in rows),'empty_predictions':sum(r['empty'] for r in rows),'full_predictions':sum(r['full'] for r in rows)}
        fallbacks[family]={}
        for mode,rr in fallback_rows.items():
            delta=np.array([r['delta'] for r in rr]);rng=np.random.default_rng(42);ci=np.quantile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975]);fallbacks[family][mode]={'miou':float(np.mean([r['miou'] for r in rr])),'delta':float(delta.mean()),'ci95':ci.tolist(),'improved':int((delta>1e-10).sum()),'worse':int((delta<-1e-10).sum()),'changed_masks':sum(r['changed'] for r in rr),'per_image':rr}
        print('ERROR_BUDGET',family,json.dumps(summaries[family]),flush=True);print('FALLBACK',family,json.dumps({k:{x:y for x,y in v.items() if x!='per_image'} for k,v in fallbacks[family].items()}),flush=True)
    total=sum(r['loss_clean']+r['loss_wm'] for r in allrows)/510;assert abs(total-(1-np.mean([r['miou'] for r in allrows])))<1e-12
    atomic_json(out/'error_budget.json',{'families':summaries,'combined_proxy_miou':1-total,'sum_weighted_error':total,'n':510,'important':'PS uses historical #2409 OOF, not the newer #2675 full-data checkpoint. TM and SS match current production recipes. Shape proxies are NOT verified edit labels;full_watermark describes initial embedding/restoration only. No test labels used.'});atomic_json(out/'per_image.json',allrows);atomic_json(out/'fallback.json',{'families':fallbacks,'protocol':'two predeclared fixed1%-area probability-ranked fallback rules;only originally empty/full masks changed;exploratory existing OOF,not a hidden-test guarantee'})


if __name__=='__main__':main()
