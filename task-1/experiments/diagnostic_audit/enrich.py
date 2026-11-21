"""Additional controls that prevent image-level and frame-edge confounding."""
from common import *
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from metrics_audit import boot

def main():
    df=pd.read_csv(OUT/'per_image_metrics.csv');summary=pd.read_csv(OUT/'summary_metrics.csv');rows=[];corr=[]
    for col in ['boundary_internal_undefined_reason','precision_undefined_reason','recall_undefined_reason','logit_source']:
        df[col]=df[col].fillna('').astype(object) if col in df else ''
    for (method,variant,seed),sub in df.groupby(['watermark_method','variant','seed'],sort=False):
        c=load_cohort(method,variant,int(seed));ids={n:i for i,n in enumerate(c['names'])}
        cache=torch.load(cv.model.OUT/f'{method}_features.pt',weights_only=True)
        logits=np.zeros_like(c['prob'])
        for fold,path in enumerate(c['checkpoints'],1):
            checkpoint=torch.load(path,weights_only=True);head=cv.model.Head(checkpoint['channels']);head.load_state_dict(checkpoint['state_dict']);head.eval()
            ii=np.where(c['fold']==fold)[0]
            with torch.inference_mode():
                for start in range(0,len(ii),8):
                    batch=ii[start:start+8];z=head([f[batch] for f in cache['features']]);logits[batch]=z.numpy()
                    np.testing.assert_allclose(z.sigmoid().numpy(),c['prob'][batch],atol=1e-7,rtol=0)
        for index,row in sub.iterrows():
            i=ids[row.image_id];g=c['gt'][i];p=c['prob'][i]>=c['threshold'][i]
            z=logits[i].astype(np.float64)
            df.loc[index,'mean_logit_inside']=float(z[g].mean()) if g.any() else np.nan
            df.loc[index,'mean_logit_outside']=float(z[~g].mean()) if (~g).any() else np.nan
            df.loc[index,'logit_separation']=df.loc[index,'mean_logit_inside']-df.loc[index,'mean_logit_outside']
            df.loc[index,'log_loss']=float(np.mean(np.logaddexp(0,z)-g*z))
            df.loc[index,'logit_source']='recomputed checkpoint logits; no inverse-sigmoid clipping'
            a=g&~ndi.binary_erosion(g,border_value=1);b=p&~ndi.binary_erosion(p,border_value=1)
            for tol in (1,2,4):
                if a.any() and b.any():
                    precision=(ndi.distance_transform_edt(~a)[b]<=tol).mean();recall=(ndi.distance_transform_edt(~b)[a]<=tol).mean()
                    f=2*precision*recall/(precision+recall) if precision+recall else 0.
                elif a.any() or b.any():f=0.
                else:f=np.nan
                df.loc[index,f'boundary_internal_f1_{tol}']=f
            df.loc[index,'boundary_internal_undefined_reason']='both masks have no internal boundary' if not a.any() and not b.any() else ''
            df.loc[index,'precision_undefined_reason']='prediction empty' if not p.any() else ''
            df.loc[index,'recall_undefined_reason']='GT empty' if not g.any() else ''
        for label,select in [('all',np.ones(len(sub),bool)),('small_le5pct',sub.gt_foreground_fraction<=.05),('mid_5to50pct',(sub.gt_foreground_fraction>.05)&(sub.gt_foreground_fraction<=.5)),('large_gt50pct',sub.gt_foreground_fraction>.5)]:
            z=sub[select];r=dict(method=method,variant=variant,seed=seed,stratum=label,n=len(z))
            for col in ['official_miou','iou_watermark','iou_clean','top_k_precision','top_k_random_baseline','roc_auc','ap','ap_prevalence_lift','gt_foreground_fraction','predicted_foreground_fraction','probability_separation','delta_vs_empty']:
                r[col]=float(z[col].mean())
            r['top_k_lift']=float((z.top_k_precision/z.gt_foreground_fraction).mean());rows.append(r)
        mask=(summary.watermark_method==method)&(summary.variant==variant)&(summary.seed==seed)
        updated=df.loc[sub.index]
        for tol in (1,2,4):summary.loc[mask,f'boundary_internal_f1_{tol}_mean']=updated[f'boundary_internal_f1_{tol}'].mean()
        for key in ['log_loss','logit_separation']:
            summary.loc[mask,key+'_mean']=updated[key].mean();summary.loc[mask,key+'_median']=updated[key].median()
    df.to_csv(OUT/'per_image_metrics.csv',index=False);summary.to_csv(OUT/'summary_metrics.csv',index=False)
    dec=pd.read_csv(OUT/'decoder_metrics.csv')
    for method in METHODS:
        v='stable' if method=='stegastamp' else 'original';z=df[(df.watermark_method==method)&(df.variant==v)&(df.seed==42)].merge(dec,on='image_id')
        for label,g in z.groupby('size_bin'):
            for y in ['official_miou','iou_watermark','predicted_foreground_fraction']:
                reason='constant input' if g.bit_accuracy.nunique()<2 or g[y].nunique()<2 else ''
                stat=spearmanr(g.bit_accuracy,g[y]) if not reason else None
                corr.append(dict(method=method,size_bin=label,n=len(g),x='bit_accuracy',y=y,rho=float(stat.statistic) if stat else np.nan,p_value=float(stat.pvalue) if stat else np.nan,undefined_reason=reason,unreliable=len(g)<10))
    pd.DataFrame(rows).to_csv(OUT/'ranking_stratified_metrics.csv',index=False);pd.DataFrame(corr).to_csv(OUT/'decoder_correlations_by_area.csv',index=False)
    # Add probe comparison with matched current-head fold and area strata, not overall scores from other cohorts.
    probe_rows=[]
    for path in sorted((OUT/'cache').glob('probe_*.npz')):
        method=path.stem.split('_')[1];variant=path.stem[len('probe_'+method+'_'):];z=np.load(path)
        c=load_cohort(method,'stable' if method=='stegastamp' else 'original');ids=[c['names'].index(n) for n in z['names']];g=c['gt'][ids];p=z['probability'];t=float(z['threshold'])
        from probes import report_prob
        for label,selection in [('all',np.ones(len(g),bool)),('small_le5pct',g.mean((1,2))<=.05),('mid_5to50pct',(g.mean((1,2))>.05)&(g.mean((1,2))<=.5)),('large_gt50pct',g.mean((1,2))>.5)]:
            if not selection.any():continue
            vals=report_prob(p[selection],g[selection],t);base=ious(c['prob'][ids][selection]>=c['threshold'][ids][selection,None,None],g[selection])[0]
            delta=ious(p[selection]>=t,g[selection])[0]-base
            probe_rows.append(dict(method=method,variant=variant,stratum=label,**vals,current_head_miou=float(base.mean()),delta=float(delta.mean()),ci_low=boot(delta)[0],ci_high=boot(delta)[1]))
    pd.DataFrame(probe_rows).to_csv(OUT/'probe_stratified_metrics.csv',index=False)
    doc=json.loads((OUT/'diagnostic_results.json').read_text(encoding='utf-8'));doc['summary']=summary.to_dict('records');doc['scope']=dict(unique_images=510,per_image_rows=len(df),primary='original PixelSeal/TrustMark seed42; stable StegaStamp seed42',secondary='original StegaStamp seed42 plus stable seeds7,123; repeated images are not independent samples')
    save_json(OUT/'diagnostic_results.json',doc)

if __name__=='__main__':main()
