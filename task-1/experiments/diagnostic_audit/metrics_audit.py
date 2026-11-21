"""All-image OOF localization, geometry, calibration, and visualization audit."""
from common import *
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import html

def ratio(a,b): return float(a/b) if b else np.nan
def mean(a): return float(np.mean(a)) if len(a) else np.nan
def boundary(m): return m & ~ndi.binary_erosion(m, border_value=0)
def box(m):
    y,x=np.where(m)
    return (int(y.min()),int(x.min()),int(y.max()+1),int(x.max()+1)) if len(y) else None

def geometry(g,p):
    gl,gn=ndi.label(g,np.ones((3,3))); pl,pn=ndi.label(p,np.ones((3,3)))
    gb,pb=box(g),box(p); bg,bp=boundary(g),boundary(p)
    sizes=np.sort(np.bincount(pl.ravel())[1:])[::-1]
    holes=ndi.binary_fill_holes(g)&~g
    dist=ndi.distance_transform_edt(np.pad(g,1))[1:-1,1:-1]
    ridges=(dist==ndi.maximum_filter(dist,size=3))&g
    d=dict(gt_components=gn,predicted_components=pn,gt_holes=ndi.label(holes)[1],
        gt_hole_fraction=float(holes.mean()),gt_perimeter_area=ratio(bg.sum(),g.sum()),
        thickness_min= float(2*dist[ridges].min()) if ridges.any() else np.nan,
        thickness_median=float(2*np.median(dist[ridges])) if ridges.any() else np.nan,
        largest_component_pixels=int(sizes[0]) if len(sizes) else 0,
        second_component_pixels=int(sizes[1]) if len(sizes)>1 else 0,
        largest_component_fraction=ratio(sizes[0],p.sum()) if len(sizes) else np.nan,
        gt_component_overlap=ratio(len(set(gl[p].tolist())-{0}),gn),
        centroid_distance=float(np.linalg.norm(np.asarray(ndi.center_of_mass(g))-ndi.center_of_mass(p))) if g.any() and p.any() else np.nan,
        bbox_iou=np.nan,gt_bbox_width=np.nan,gt_bbox_height=np.nan,gt_bbox_aspect=np.nan,gt_bbox_fill=np.nan)
    if gb:
        h,w=gb[2]-gb[0],gb[3]-gb[1]
        d.update(gt_bbox_width=w,gt_bbox_height=h,gt_bbox_aspect=w/h,gt_bbox_fill=g.sum()/(w*h))
    if gb and pb:
        inter=max(0,min(gb[2],pb[2])-max(gb[0],pb[0]))*max(0,min(gb[3],pb[3])-max(gb[1],pb[1]))
        d['bbox_iou']=ratio(inter,(gb[2]-gb[0])*(gb[3]-gb[1])+(pb[2]-pb[0])*(pb[3]-pb[1])-inter)
    for tol in (1,2,4):
        if bg.any() and bp.any():
            prec=float((ndi.distance_transform_edt(~bg)[bp]<=tol).mean())
            rec=float((ndi.distance_transform_edt(~bp)[bg]<=tol).mean())
            f=2*prec*rec/(prec+rec) if prec+rec else 0.
        elif bg.any() or bp.any(): prec=0. if bp.any() else np.nan;rec=0. if bg.any() else np.nan;f=0.
        else: prec=rec=f=np.nan
        d.update({f'boundary_precision_{tol}':prec,f'boundary_recall_{tol}':rec,f'boundary_f1_{tol}':f})
    d['boundary_undefined_reason']='both masks empty' if not bg.any() and not bp.any() else ('one boundary empty; F1=0' if not bg.any() or not bp.any() else '')
    return d

def one(p,g,t):
    m=p>=t; mi,fg,bg,tp,fp,fn,tn=ious(m,g)
    # Call the actual starter-kit implementation for every image.
    actual=official.miou_image(m,g); assert abs(actual-mi)<1e-12
    f=float(g.mean()); a=float(m.mean()); mixed=g.any() and not g.all()
    z=np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1))
    inside,outside=p[g],p[~g]
    sep=mean(inside)-mean(outside)
    denom=np.sqrt((np.var(inside)+np.var(outside))/2) if mixed else np.nan
    k=int(g.sum())
    top=float(g.ravel()[np.argsort(-p.ravel(),kind='stable')[:k]].mean()) if k else np.nan
    empty=official.miou_image(np.zeros_like(g),g)
    d=dict(gt_foreground_fraction=f,predicted_foreground_fraction=a,official_miou=actual,iou_watermark=float(fg),iou_clean=float(bg),
        foreground_precision=ratio(tp,tp+fp),foreground_recall=ratio(tp,tp+fn),foreground_f1=ratio(2*tp,2*tp+fp+fn),
        roc_auc=roc_auc_score(g.ravel(),p.ravel()) if mixed else np.nan,
        ap=average_precision_score(g.ravel(),p.ravel()) if mixed else np.nan,
        ranking_undefined_reason='' if mixed else ('GT empty' if not g.any() else 'GT full; AP=1 degenerate, excluded'),
        mean_probability_inside=mean(inside),mean_probability_outside=mean(outside),
        mean_logit_inside=mean(z[g]),mean_logit_outside=mean(z[~g]),probability_separation=sep,
        logit_separation=mean(z[g])-mean(z[~g]),effect_size=sep/denom if denom>1e-12 else np.nan,
        top_k_precision=top,top_k_recall=top,top_k_random_baseline=f if k else np.nan,
        top_k_tied_at_cutoff=int((p==np.sort(p.ravel())[-k]).sum()) if k else 0,
        empty_baseline_miou=empty,full_baseline_miou=official.miou_image(np.ones_like(g),g),delta_vs_empty=actual-empty,
        predicted_gt_area_ratio=ratio(a,f),absolute_area_error=abs(a-f),prediction_empty=not m.any(),prediction_nearly_full=a>=.99,
        brier=float(np.mean((p-g)**2)),log_loss=float(-np.mean(g*np.log(np.clip(p,1e-7,1))+(~g)*np.log(np.clip(1-p,1e-7,1)))),
        tp=tp,fp=fp,fn=fn,tn=tn,threshold=float(t))
    d['ap_prevalence_lift']=ratio(d['ap'],f)
    for label,v in [('inside',inside),('outside',outside)]:
        for q in (5,25,50,75,95): d[f'probability_{label}_q{q}']=float(np.percentile(v,q)) if len(v) else np.nan
    d.update(geometry(g,m))
    return d

def sizebin(f):
    if f==0:return 'empty'
    if f<=.01:return '(0,1%]'
    if f<=.05:return '(1,5%]'
    if f<=.2:return '(5,20%]'
    if f<=.5:return '(20,50%]'
    return '(50,100%]'

def boot(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    if not len(x):return [np.nan,np.nan]
    rng=np.random.default_rng(20261121)
    return np.quantile(x[rng.integers(len(x),size=(2000,len(x)))].mean(1),[.025,.975]).tolist()

def reliability(p,g):
    bins=np.minimum((p.ravel()*15).astype(int),14); rows=[]
    for b in range(15):
        ix=bins==b
        rows.append(dict(bin=b,n=int(ix.sum()),confidence=mean(p.ravel()[ix]),positive_rate=mean(g.ravel()[ix])))
    ece=sum(r['n']*abs(r['confidence']-r['positive_rate']) for r in rows if r['n'])/p.size
    return rows,ece

def save_example(c,i,r):
    rgb=np.asarray(Image.open(ROOT/'data/train/images'/f"{c['names'][i]}.png").convert('RGB').resize((128,128)))/255
    g=c['gt'][i]; p=c['prob'][i]; m=p>=c['threshold'][i]
    overlay=rgb.copy();overlay[g]=.6*overlay[g]+.4*np.array([0,1,0]);overlay[m]=.6*overlay[m]+.4*np.array([1,.6,0])
    errors=rgb*.25;errors[m&~g]=[1,0,0];errors[~m&g]=[0,.4,1]
    fig,axs=plt.subplots(1,6,figsize=(15,3.3))
    for ax,img,title in zip(axs,[rgb,g,p,m,overlay,errors],['RGB','GT: watermark=1','P(watermark)','Prediction','GT green / pred amber','FP red / FN blue']):
        ax.imshow(img,cmap='viridis' if title.startswith('P(') else 'gray',vmin=0,vmax=1);ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle(f"{c['method']} | {c['names'][i]} | fold {r['fold']} | {c['variant']} seed {c['seed']}\nmIoU {r['official_miou']:.3f} | FG IoU {r['iou_watermark']:.3f} | AUC {r['roc_auc']:.3f} | AP {r['ap']:.3f} | GT {r['gt_foreground_fraction']:.1%} | pred {r['predicted_foreground_fraction']:.1%} | delta empty {r['delta_vs_empty']:+.3f}",fontsize=10)
    fig.tight_layout();path=f"{c['method']}_{c['variant']}_{c['seed']}_{i:03}.jpg";fig.savefig(OUT/'examples'/path,dpi=95);plt.close(fig)
    return path

def main():
    ds=cv.model.LocalisationDataset(ROOT/'data','train')
    cohorts=[load_cohort(m) for m in METHODS]+[load_cohort('stegastamp','stable',s) for s in (42,7,123)]
    allrows=[];summaries=[];calrows=[];size_rows=[];corrrows=[];manifest=[];gallery=[];spatial_rows=[]
    for c in cohorts:
        method,variant,seed=c['method'],c['variant'],c['seed']; tag=f'{method}_{variant}_{seed}'
        print('METRICS',tag,flush=True)
        cache=torch.load(cv.model.OUT/f'{method}_features.pt',weights_only=True)
        assert c['names']==cache['names']
        np.testing.assert_array_equal(c['gt'],np.stack([ds.m128[n] for n in c['names']]))
        # Reuse, but verify, all saved checkpoints; never retrain here.
        for fold,ck in enumerate(c['checkpoints'],1):
            checkpoint=torch.load(ck,weights_only=True);head=cv.model.Head(checkpoint['channels']);head.load_state_dict(checkpoint['state_dict'])
            ids=np.where(c['fold']==fold)[0];actual=cv.predict(head,cache['features'],ids)
            np.testing.assert_allclose(actual,c['prob'][ids],atol=1e-7,rtol=0)
            np.testing.assert_allclose(c['threshold'][ids],checkpoint['threshold'],atol=0,rtol=0)
            manifest.append(dict(cohort=tag,fold=fold,path=str(ck.relative_to(ROOT)),sha256=sha(ck),epoch=checkpoint['epoch'],threshold=checkpoint['threshold']))
        curve=threshold_scores(c['prob'],c['gt']);global_index=curve.mean(0).argmax()
        rel,ece=reliability(c['prob'],c['gt']);pd.DataFrame(rel).to_csv(OUT/f'reliability_{tag}.csv',index=False)
        rows=[]
        for i,(p,g,t) in enumerate(zip(c['prob'],c['gt'],c['threshold'])):
            r=one(p,g,t);r.update(image_id=c['names'][i],watermark_method=method,variant=variant,fold=int(c['fold'][i]),seed=seed,
                checkpoint=str(c['checkpoints'][c['fold'][i]-1].relative_to(ROOT)),global_decoder_bit_accuracy=cache['bit_accuracy'][i],size_bin=sizebin(float(g.mean())),
                oracle_threshold=float(THRESHOLDS[curve[i].argmax()]),oracle_threshold_miou=float(curve[i].max()),
                global_resubstitution_threshold=float(THRESHOLDS[global_index]),global_resubstitution_miou=float(curve[i,global_index]))
            r['oracle_threshold_gap']=r['oracle_threshold_miou']-r['official_miou']
            # GT-only coarse reconstruction diagnostic; not model quality or strict bound.
            coarse=torch.nn.functional.avg_pool2d(torch.from_numpy(g.astype(np.float32))[None,None],4)
            rec=torch.nn.functional.interpolate(coarse,size=(128,128),mode='bilinear',align_corners=False)[0,0].numpy()
            r['gt_coarse32_reconstruction_miou']=official.miou_image(rec>=.5,g)
            r['gt_coarse32_reconstruction_fg_iou']=float(ious(rec>=.5,g)[1]); rows.append(r)
        df=pd.DataFrame(rows);allrows.extend(rows)
        summary=dict(watermark_method=method,variant=variant,seed=seed,n=len(df),
            pooled_roc_auc=roc_auc_score(c['gt'].ravel(),c['prob'].ravel()),pooled_ap=average_precision_score(c['gt'].ravel(),c['prob'].ravel()),
            prevalence=float(c['gt'].mean()),ece=ece,undefined_auc=int(df.roc_auc.isna().sum()),undefined_ap=int(df.ap.isna().sum()),
            delta_positive_fraction=float((df.delta_vs_empty>1e-12).mean()),delta_negative_fraction=float((df.delta_vs_empty< -1e-12).mean()),delta_equal_fraction=float((df.delta_vs_empty.abs()<=1e-12).mean()))
        for col in ['official_miou','iou_watermark','iou_clean','foreground_precision','foreground_recall','foreground_f1','roc_auc','ap','ap_prevalence_lift','probability_separation','logit_separation','effect_size','top_k_precision','empty_baseline_miou','full_baseline_miou','delta_vs_empty','predicted_foreground_fraction','gt_foreground_fraction','predicted_gt_area_ratio','absolute_area_error','prediction_empty','prediction_nearly_full','boundary_f1_1','boundary_f1_2','boundary_f1_4','centroid_distance','bbox_iou','gt_components','predicted_components','largest_component_fraction','gt_component_overlap','oracle_threshold_miou','oracle_threshold_gap','brier','log_loss','global_decoder_bit_accuracy','gt_coarse32_reconstruction_miou','gt_coarse32_reconstruction_fg_iou']:
            v=df[col].astype(float); summary[col+'_mean']=float(v.mean()); summary[col+'_median']=float(v.median());summary[col+'_n']=int(v.notna().sum())
        summary['official_miou_std']=float(df.official_miou.std())
        for col in ['official_miou','delta_vs_empty','roc_auc','ap','iou_watermark']:
            summary[col+'_ci_low'],summary[col+'_ci_high']=boot(df[col])
        tp,fp,fn,tn=(df[k].sum() for k in ('tp','fp','fn','tn'))
        summary.update(pooled_fg_iou=ratio(tp,tp+fp+fn),pooled_clean_iou=ratio(tn,tn+fp+fn),pooled_precision=ratio(tp,tp+fp),pooled_recall=ratio(tp,tp+fn),pooled_f1=ratio(2*tp,2*tp+fp+fn))
        summary['pooled_ap_prevalence_lift']=summary['pooled_ap']/summary['prevalence']
        summaries.append(summary)
        for label in ['empty','(0,1%]','(1,5%]','(5,20%]','(20,50%]','(50,100%]']:
            sub=df[df.size_bin==label]; row=dict(watermark_method=method,variant=variant,seed=seed,size_bin=label,n=len(sub),unreliable=len(sub)<10)
            for col in ['official_miou','iou_watermark','ap','roc_auc','delta_vs_empty','prediction_empty','predicted_gt_area_ratio','top_k_precision','gt_foreground_fraction','boundary_f1_2','gt_coarse32_reconstruction_fg_iou']:
                row[col]=float(sub[col].astype(float).mean());row[col+'_defined_n']=int(sub[col].notna().sum())
            size_rows.append(row)
        for fold in (1,2,3):
            ids=c['fold']==fold
            # Cross-fold posthoc calibration shares upstream training data: NOT independent nested calibration.
            selected=curve[~ids].mean(0).argmax()
            calrows.append(dict(watermark_method=method,variant=variant,seed=seed,fold=fold,n=int(ids.sum()),operational_miou=float(df.loc[ids,'official_miou'].mean()),
                operational_threshold=float(c['threshold'][ids][0]),global_resubstitution_threshold=float(THRESHOLDS[global_index]),global_resubstitution_miou=float(curve[ids,global_index].mean()),
                cross_oof_threshold=float(THRESHOLDS[selected]),cross_oof_diagnostic_miou=float(curve[ids,selected].mean()),oracle_miou=float(curve[ids].max(1).mean()),
                fold_oracle_global_threshold=float(THRESHOLDS[curve[ids].mean(0).argmax()]),caveat='cross-OOF has training overlap; only operational threshold is nested'))
        pd.DataFrame(dict(threshold=THRESHOLDS,miou=curve.mean(0))).to_csv(OUT/f'threshold_curve_{tag}.csv',index=False)
        for x in ['global_decoder_bit_accuracy','gt_foreground_fraction','gt_components','gt_holes','gt_perimeter_area','thickness_median']:
            for y in ['official_miou','iou_watermark','ap','predicted_foreground_fraction','delta_vs_empty']:
                sub=df[[x,y]].dropna(); stat=spearmanr(sub[x],sub[y])
                corrrows.append(dict(watermark_method=method,variant=variant,seed=seed,x=x,y=y,n=len(sub),spearman_rho=float(stat.statistic),p_value=float(stat.pvalue)))
        primary=variant=='original' and method!='stegastamp' or variant=='stable' and seed==42
        if primary:
            # Small shifts of probability maps with edge padding; all GT-selected results explicitly oracle.
            choices=[('identity',0,0,'none')]+[(f'shift_{dy}_{dx}',dy,dx,'none') for dy,dx in [(0,-4),(0,-2),(0,2),(0,4),(-4,0),(-2,0),(2,0),(4,0)]]+[(k,0,0,k) for k in ('horizontal','vertical','both')]
            scores=[]
            for label,dy,dx,flip in choices:
                p=c['prob']
                if dy or dx:p=ndi.shift(p,(0,dy,dx),order=0,mode='nearest')
                if flip in ('horizontal','both'):p=p[:,:,::-1]
                if flip in ('vertical','both'):p=p[:,::-1,:]
                scores.append(ious(p>=c['threshold'][:,None,None],c['gt'])[0])
                spatial_rows.append(dict(method=method,transform=label,mean_miou=float(scores[-1].mean()),delta=float((scores[-1]-df.official_miou).mean()),oracle=True))
            ss=np.stack(scores,1);pd.DataFrame(dict(image_id=c['names'],best_oracle_transform=[choices[k][0] for k in ss.argmax(1)],oracle_miou=ss.max(1),delta=ss.max(1)-df.official_miou)).to_csv(OUT/f'spatial_oracle_{method}.csv',index=False)
            selections={}
            for cat,idx in [('best_fg',df.nlargest(5,'iou_watermark').index),('worst_fg',df.nsmallest(5,'iou_watermark').index),('small',df[df.gt_foreground_fraction>0].nsmallest(5,'gt_foreground_fraction').index),('high_auc_low_iou',df[df.roc_auc>.7].nsmallest(5,'iou_watermark').index),('low_auc',df.nsmallest(5,'roc_auc').index),('empty_competitive',df.nsmallest(5,'delta_vs_empty').index)]:
                for i in idx: selections.setdefault(i,[]).append(cat)
            # All 510 primary OOF images, not just hand-picked successes.
            for i in range(len(df)):
                file=save_example(c,i,rows[i]);gallery.append(dict(method=method,name=c['names'][i],file=file,categories=selections.get(i,[]),**{k:rows[i][k] for k in ['official_miou','iou_watermark','roc_auc','ap','delta_vs_empty','gt_foreground_fraction']}))
            fig,axs=plt.subplots(2,3,figsize=(15,8))
            for g,label,color in [(True,'Inside GT','tab:orange'),(False,'Outside GT','tab:blue')]:
                axs[0,0].hist(c['prob'][c['gt']==g],bins=30,density=True,alpha=.5,label=label,color=color)
            axs[0,0].legend();axs[0,0].set(title='Probability distribution',xlabel='Probability')
            for col,marker in [('iou_watermark','o'),('ap','x'),('delta_vs_empty','+')]:axs[0,1].scatter(df.gt_foreground_fraction,df[col],s=12,marker=marker,label=col)
            axs[0,1].legend(fontsize=8);axs[0,1].set(xlabel='GT area',title='Localization versus area')
            axs[0,2].scatter(df.gt_foreground_fraction,df.predicted_foreground_fraction,s=12);axs[0,2].plot([0,1],[0,1],'k--');axs[0,2].set(xlabel='GT area',ylabel='Predicted area')
            axs[1,0].plot(THRESHOLDS,curve.mean(0));axs[1,0].axhline(df.official_miou.mean(),color='gray',linestyle='--');axs[1,0].set(xlabel='Threshold',ylabel='Official mIoU',title='OOF resubstitution curve (optimistic)')
            axs[1,1].plot([r['confidence'] for r in rel],[r['positive_rate'] for r in rel],'o-');axs[1,1].plot([0,1],[0,1],'k--');axs[1,1].set(xlabel='Mean probability',ylabel='Observed positive fraction',title=f'Reliability / ECE {ece:.3f}')
            axs[1,2].scatter(df.global_decoder_bit_accuracy,df.iou_watermark,c=df.gt_foreground_fraction,cmap='viridis',s=15);axs[1,2].set(xlabel='Decoder bit accuracy',ylabel='Foreground IoU',title='Color: GT area')
            fig.suptitle(tag);fig.tight_layout();fig.savefig(OUT/'figures'/f'{method}_diagnostics.png',dpi=140);plt.close(fig)
        print(tag,'mIoU',summary['official_miou_mean'],'FG',summary['iou_watermark_mean'],'AUC',summary['roc_auc_mean'],'AP',summary['ap_mean'],'empty',summary['empty_baseline_miou_mean'],flush=True)
    for filename,rows in [('per_image_metrics',allrows),('summary_metrics',summaries),('size_stratified_metrics',size_rows),('calibration_results',calrows),('correlations',corrrows),('spatial_corrections',spatial_rows)]:pd.DataFrame(rows).to_csv(OUT/f'{filename}.csv',index=False)
    df=pd.DataFrame(allrows);paired=[]
    old=df[(df.watermark_method=='stegastamp')&(df.variant=='original')].set_index('image_id')
    for seed in (42,7,123):
        new=df[(df.variant=='stable')&(df.seed==seed)].set_index('image_id').loc[old.index]
        delta=new.official_miou-old.official_miou
        paired.append(dict(seed=seed,comparison='stable seed vs original seed42; only seed42 strictly paired training seed',mean_delta=float(delta.mean()),ci=boot(delta)))
    save_json(OUT/'diagnostic_results.json',dict(summary=summaries,paired=paired,checkpoints=manifest,fold_manifest_sha256=sha(cv.OUT/'folds.json'),official_metric_sha256=sha(ROOT/'starter-kit/task-1/wmloc/utils.py'),bootstrap='2000 image resamples, seed20261121; conditional on fitted OOF models, no model-refit uncertainty'))
    save_json(OUT/'examples/index.json',gallery)
    cards=[]
    for r in gallery:
        cats=', '.join(r['categories'])
        cards.append(f'<article data-method="{r["method"]}" data-hard="{bool(cats)}"><h3>{html.escape(r["method"]+" / "+r["name"])}</h3><p>{cats or "pozostałe OOF"}</p><img loading="lazy" src="{r["file"]}" alt="RGB, GT, probability, prediction, overlay, FP and FN" /></article>')
    page='''<!doctype html><html lang="pl"><meta charset="utf-8"><title>Audyt lokalizacji — 510 obrazów OOF</title><style>body{font:16px system-ui;background:#f4f6f8;color:#18212b;margin:25px}article{background:white;padding:12px;margin:20px 0}img{width:100%;height:auto}select{padding:8px}header{position:sticky;top:0;background:#f4f6f8;padding:12px}h3{font-size:14px}</style><header><h1>Diagnostyka 510 obrazów OOF</h1><p>PixelSeal i TrustMark: original seed42. StegaStamp: stable seed42. Zielony = GT, bursztynowy = predykcja; czerwony = FP, niebieski = FN.</p><label>Metoda <select id="method"><option value="">Wszystkie</option><option>pixelseal</option><option>trustmark</option><option>stegastamp</option></select></label> <label><input type="checkbox" id="hard"> Tylko wybrane przypadki</label><span id="count"></span></header>'''+''.join(cards)+'''<script>function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{let show=(!method.value||a.dataset.method===method.value)&&(!hard.checked||a.dataset.hard==='True');a.hidden=!show;if(show)n++});count.textContent=' '+n+' obrazów'}method.onchange=hard.onchange=filter;filter()</script></html>'''
    (OUT/'examples/index.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
