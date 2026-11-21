"""Phases C/D: nested threshold/orientation selection, spatial controls, paired diagnostics."""
import argparse
import pandas as pd
from common import *

METRICS=['miou','fg_iou','clean_iou','auc','ap','clean_ap','topk_precision','topk_lift','boundary_f1',
         'internal_boundary_f1','hole_recall','hole_component_recall','clean_fpr','tp_coverage',
         'false_positive_area','complete_miss','gt_area','pred_area','components','largest_component_area','spatial_correlation']

def summarize(df):
    rows=[]
    for family in ['all',*METHODS]:
        f=df if family=='all' else df[df.family==family]
        for label,select in [('all',np.ones(len(f),bool)),('le1pct',f.gt_area<=.01),
                             ('1to5pct',(f.gt_area>.01)&(f.gt_area<=.05)),('5to20pct',(f.gt_area>.05)&(f.gt_area<=.2)),
                             ('20to50pct',(f.gt_area>.2)&(f.gt_area<=.5)),('gt50pct',f.gt_area>.5),('small_le5pct',f.gt_area<=.05)]:
            for variant,g in f[select].groupby('variant'):
                rows.append(dict(family=family,stratum=label,variant=variant,n=len(g),
                                 n_auc=int(g.auc.notna().sum()),hole_images=int((g.hole_pixels>0).sum()),
                                 **{m:float(g[m].mean()) for m in METRICS}))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='trufor'); args=ap.parse_args(); model=args.model
    records,b,g,splits,provenance=cohort(); names=[r['image_id'] for r in records]
    files=[OUT/model/f'{n}.npz' for n in names]; assert all(p.exists() for p in files), 'Incomplete inference'
    f=np.stack([np.load(p)['probability'].astype('float32') for p in files]); assert f.shape==g.shape==(510,128,128)
    assert np.isfinite(f).all() and f.min()>=0 and f.max()<=1
    out=OUT/f'eval_{model}'; out.mkdir(exist_ok=True)
    thresholds={}; choices=[]
    for family in METHODS:
        for fold,s in enumerate(splits[family],1):
            cal=np.array([names.index(n) for n in s['inner_validation']]); val=[names.index(n) for n in s['validation']]
            assert not set(cal)&set(val)
            opts=[]
            for orientation,p in [('direct',f),('inverse',1-f)]:
                cuts=np.unique(np.r_[0.,np.quantile(p[cal],np.linspace(0,1,101)),np.linspace(.05,.95,19),1.000001]).astype('float32')
                scores=np.mean([curve(p[i],g[i],cuts) for i in cal],axis=0); k=int(scores.argmax())
                thresholds[family,fold,orientation]=float(cuts[k])
                opts.append(dict(family=family,fold=fold,orientation=orientation,threshold=float(cuts[k]),
                                 inner_miou=float(scores[k]),calibration_names=s['inner_validation'],validation_names=s['validation']))
            selected=max(opts,key=lambda x:x['inner_miou']); thresholds[family,fold,'selected']=selected['orientation']
            choices.extend([dict(o,selected=o['orientation']==selected['orientation']) for o in opts])
    rows=[]; complement=[]; runtimes=[]; selected_probs=[]; selected_threshold=[]; oracle_rows=[]
    for i,r in enumerate(records):
        family=r['family']; fold=r['fold']; z=np.load(files[i]); p=f[i]; target=g[i]
        td=thresholds[family,fold,'direct']; ti=thresholds[family,fold,'inverse']; orient=thresholds[family,fold,'selected']
        chosen=p if orient=='direct' else 1-p; tc=td if orient=='direct' else ti
        selected_probs.append(chosen); selected_threshold.append(tc)
        variants={'baseline':(b[i],r['baseline_threshold']),'direct_fixed':(p,.5),'inverse_fixed':(1-p,.5),
                  'direct_calibrated':(p,td),'inverse_calibrated':(1-p,ti),'selected':(chosen,tc),
                  'native_binary':(z['native_binary128'].astype('float32'),.5),
                  'native_binary_inverse':(1-z['native_binary128'].astype('float32'),.5)}
        local={}
        for v,(prob,t) in variants.items():
            m=measure(prob,target,t,oracle=v in ('direct_fixed','inverse_fixed'))
            local[v]=m; rows.append(dict(**r,model=model,variant=v,**m))
        fixed='direct_fixed' if local['direct_fixed']['miou']>=local['inverse_fixed']['miou'] else 'inverse_fixed'
        best='direct_fixed' if local['direct_fixed']['oracle_threshold_miou']>=local['inverse_fixed']['oracle_threshold_miou'] else 'inverse_fixed'
        po=p if best=='direct_fixed' else 1-p
        om=measure(po,target,local[best]['oracle_threshold']); rows.append(dict(**r,model=model,variant='oracle_orientation_threshold',**om))
        rows.append(dict(**r,model=model,variant='oracle_orientation_fixed',**local[fixed]))
        empty=measure(np.zeros_like(p),target,.5); full=measure(np.ones_like(p),target,.5)
        trivial=max(empty['miou'],full['miou']); rows.append(dict(**r,model=model,variant='oracle_empty_full',**(empty if empty['miou']>=full['miou'] else full)))
        # Fixed coordinate shifts retain each map's histogram, area and shape. No new images are synthesized.
        shift=[]; shift_fixed=[]
        for dy,dx in [(32,0),(0,32),(64,64),(32,64)]:
            shifted=np.roll(p,(dy,dx),(0,1))
            auc=roc_auc_score(target.ravel(),shifted.ravel()) if target.any() and not target.all() else np.nan
            shift.append(max(auc,1-auc))
            shift_fixed.append(max(curve(shifted,target,[.5])[0],curve(1-shifted,target,[.5])[0]))
        oracle_rows.append(dict(**r,fixed_orientation=fixed,threshold_orientation=best,
                                fixed_miou=local[fixed]['miou'],best_threshold_miou=om['miou'],trivial_miou=trivial,
                                gain_over_trivial=om['miou']-trivial,fixed_gain_over_trivial=local[fixed]['miou']-trivial,
                                oracle_auc=max(local['direct_fixed']['auc'],local['inverse_fixed']['auc']),
                                shift_oracle_auc=float(np.mean(shift)),shift_fixed_miou=float(np.mean(shift_fixed)),
                                direct_threshold=local['direct_fixed']['oracle_threshold'],inverse_threshold=local['inverse_fixed']['oracle_threshold']))
        bp=b[i]>=r['baseline_threshold']; fp=chosen>=tc; eb=bp!=target; ef=fp!=target
        corr=float(np.corrcoef(eb.ravel(),ef.ravel())[0,1]) if eb.std()>0 and ef.std()>0 else np.nan
        complement.append(dict(**r,baseline_miou=local['baseline']['miou'],forensic_miou=local['selected']['miou'],
                                delta_miou=local['selected']['miou']-local['baseline']['miou'],
                                error_correlation=corr,error_overlap_iou=float((eb&ef).sum()/max(1,(eb|ef).sum())),
                                baseline_wrong_forensic_right=float((eb&~ef).mean()),baseline_right_forensic_wrong=float((~eb&ef).mean()),
                                oracle_best_system_miou=max(local['baseline']['miou'],local['selected']['miou']),
                                oracle_fixed_geometry_miou=max(local['direct_fixed']['miou'],local['inverse_fixed']['miou']),
                                gt_area=float(target.mean()),orientation=orient))
        runtimes.append(dict(**r,seconds=float(z['seconds']),peak_allocated_mb=float(z['peak_allocated_mb']),peak_reserved_mb=float(z['peak_reserved_mb']),
                             confidence=float(z['confidence'].mean()) if 'confidence' in z else np.nan,
                             detection=float(z['detection']) if 'detection' in z else np.nan))
        if (i+1)%50==0: print(model,'evaluated',i+1,flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'per_image_metrics.csv',index=False)
    summary=summarize(df); summary.to_csv(out/'summary_by_family_area.csv',index=False)
    folds=df.groupby(['family','fold','variant'])[METRICS].mean().reset_index(); folds.to_csv(out/'per_fold.csv',index=False)
    comp=pd.DataFrame(complement); comp.to_csv(out/'complementarity.csv',index=False)
    ora=pd.DataFrame(oracle_rows); ora.to_csv(out/'oracle_diagnostics.csv',index=False)
    rt=pd.DataFrame(runtimes); rt.to_csv(out/'runtime.csv',index=False)
    save(out/'calibration.json',choices)
    np.savez_compressed(out/'selected.npz',names=names,probability=np.stack(selected_probs),threshold=selected_threshold)
    paired=[]
    for fam in ['all',*METHODS]:
        base=df[df.variant=='baseline'].set_index('image_id'); base=base if fam=='all' else base[base.family==fam]
        for v in df.variant.unique():
            other=df[df.variant==v].set_index('image_id').loc[base.index]; delta=(other.miou-base.miou).to_numpy()
            paired.append(dict(family=fam,variant=v,delta_miou=float(delta.mean()),ci95=ci(delta),n=len(delta)))
    save(out/'paired_bootstrap.json',paired)
    # Transparent feasibility gate, not final proof; explicit geometric and held-out complementarity requirements.
    geom=(ora.fixed_gain_over_trivial>.05)
    gain=comp.oracle_fixed_geometry_miou-comp.baseline_miou
    useful=(gain>.05)&geom
    special=(comp.gt_area<=.05)|(comp.gt_area>.5)
    gate=dict(model=model,n=510,spatial_images=int(geom.sum()),complementary_images=int(useful.sum()),
              complementary_small_or_large=int((useful&special).sum()),
              folds_with_complementary_evidence=sorted(comp.loc[useful&special,'fold'].unique().tolist()),
              oracle_auc=float(ora.oracle_auc.mean()),shift_oracle_auc=float(ora.shift_oracle_auc.mean()),
              mean_shift_adjusted_oracle_auc=float((ora.oracle_auc-ora.shift_oracle_auc).mean()),
              auc_shift_ci95=ci(ora.oracle_auc-ora.shift_oracle_auc),
              rationale='Exploratory feasibility: >=10 complementary images, >=5 small/large, all3 folds, shift-adjusted oracle AUC>0.03; never an inference selector.')
    gate['pass']=bool(gate['complementary_images']>=10 and gate['complementary_small_or_large']>=5 and len(gate['folds_with_complementary_evidence'])==3 and gate['mean_shift_adjusted_oracle_auc']>.03)
    save(out/'gate.json',gate)
    save(out/'summary.json',dict(model=model,gate=gate,metrics=summary.to_dict('records'),runtime=dict(mean_seconds=float(rt.seconds.mean()),median_seconds=float(rt.seconds.median()),peak_allocated_mb=float(rt.peak_allocated_mb.max())),paired=paired))
    print(summary[(summary.stratum=='all')&(summary.family=='all')][['variant','n','miou','fg_iou','clean_iou','auc','ap','boundary_f1']].to_string(index=False),flush=True)
    print(json.dumps(gate,indent=2),flush=True)

if __name__=='__main__': main()
