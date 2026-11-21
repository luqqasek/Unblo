"""Non-training controls: spatial information, geometry, message correlations, provenance."""
from common import *
from execution_audit import inverse_feature_grid
from metrics_audit import boot,one
import pandas as pd
from scipy.stats import spearmanr
import torch.nn.functional as F
import importlib.metadata
import subprocess

def main():
    rows=[];detail=[];feature_stats=[];correlations=[];geometry=[]
    allmetrics=pd.read_csv(OUT/'per_image_metrics.csv')
    decoder=pd.read_csv(OUT/'decoder_metrics.csv')
    for method in METHODS:
        variant='stable' if method=='stegastamp' else 'original';c=load_cohort(method,variant)
        p,g,t=c['prob'],c['gt'],c['threshold'];base=ious(p>=t[:,None,None],g)[0]
        for label,pred in [('empty',np.zeros_like(g)),('full',np.ones_like(g)),('image_mean_constant',np.broadcast_to(p.mean((1,2))[:,None,None]>=t[:,None,None],g.shape)),('image_median_constant',np.broadcast_to(np.median(p,axis=(1,2))[:,None,None]>=t[:,None,None],g.shape))]:
            score=ious(pred,g)[0];delta=base-score
            rows.append(dict(method=method,control=label,n=170,miou=float(score.mean()),model_minus_control=float(delta.mean()),paired_ci_low=boot(delta)[0],paired_ci_high=boot(delta)[1],interpretation='fixed no-GT rule; inherited OOF head probabilities'))
            for i,n in enumerate(c['names']):detail.append(dict(method=method,image_id=n,control=label,miou=score[i],model_minus_control=delta[i]))
        oracle=np.maximum(ious(np.zeros_like(g),g)[0],ious(np.ones_like(g),g)[0])
        rows.append(dict(method=method,control='oracle_empty_or_full',n=170,miou=float(oracle.mean()),model_minus_control=float((base-oracle).mean()),interpretation='GT-dependent oracle: knows regime but no spatial localization'))
        rng=np.random.default_rng(20261121);values=[]
        for rep in range(5):
            shuffled=np.stack([rng.permutation(v.ravel()).reshape(128,128) for v in p])
            values.append(ious(shuffled>=t[:,None,None],g)[0])
        score=np.mean(values,axis=0);delta=base-score
        rows.append(dict(method=method,control='independent_pixel_shuffle_5_repeats',n=170,miou=float(score.mean()),model_minus_control=float(delta.mean()),paired_ci_low=boot(delta)[0],paired_ci_high=boot(delta)[1],interpretation='area/confidence preserved; spatial correspondence destroyed'))
        sub=allmetrics[(allmetrics.watermark_method==method)&(allmetrics.variant==variant)&(allmetrics.seed==42)].copy()
        dec=decoder[decoder.method==method]
        sub=sub.merge(dec,left_on='image_id',right_on='image_id',validate='one_to_one')
        for x in ['bit_accuracy','message_log_likelihood','expected_bit_probability','raw_logit_abs']:
            for y in ['official_miou','iou_watermark','gt_foreground_fraction','predicted_foreground_fraction','roc_auc','ap']:
                z=sub[[x,y]].dropna();r=spearmanr(z[x],z[y]);correlations.append(dict(method=method,x=x,y=y,n=len(z),rho=float(r.statistic),p_value=float(r.pvalue)))
        for label,select in [('one_component',sub.gt_components==1),('multiple_components',sub.gt_components>1),('has_holes',sub.gt_holes>0),('no_holes',sub.gt_holes==0),('thin_median_le8',sub.thickness_median<=8),('thin_median_gt8',sub.thickness_median>8),('elongated', (sub.gt_bbox_aspect>=4)|(sub.gt_bbox_aspect<=.25))]:
            z=sub[select];r=dict(method=method,geometry=label,n=len(z),unreliable=len(z)<10)
            for col in ['official_miou','iou_watermark','roc_auc','ap','delta_vs_empty','boundary_f1_2','gt_foreground_fraction']:r[col]=float(z[col].mean())
            geometry.append(r)
        cache=torch.load(cv.model.OUT/f'{method}_features.pt',weights_only=True)
        for layer,f in zip(cache['layers'],cache['features']):
            # Exact cache statistics over all 170 images, no sampled channels.
            feature_stats.append(dict(method=method,layer=layer,n=170,shape=str(list(f.shape)),min=float(f.min()),max=float(f.max()),mean=float(f.mean()),std=float(f.std()),zero_fraction=float((f==0).float().mean()),mean_spatial_variance=float(f.var((-2,-1)).mean()),finite=bool(torch.isfinite(f).all()),constant_channel_image_fraction=float((f.var((-2,-1))<1e-12).float().mean())))
    # Same head, inverse warp its cached features, no training and no mask-based choices.
    stn=pd.read_csv(OUT/'stn_transforms.csv').set_index('image_id');c=load_cohort('stegastamp','stable');cache=torch.load(cv.model.OUT/'stegastamp_features.pt',weights_only=True)
    ff=[f.clone() for f in cache['features']]
    for i,n in enumerate(c['names']):
        r=stn.loc[n];a=np.array([[r.a00,r.a01,r.tx],[r.a10,r.a11,r.ty]])
        for k,f in enumerate(ff):f[i:i+1]=F.grid_sample(f[i:i+1],inverse_feature_grid(a,k,f.shape[-1]),align_corners=False,padding_mode='border')
    torch.save(dict(features=ff,names=c['names']),OUT/'cache/stegastamp_aligned_features.pt')
    prob=np.zeros_like(c['prob'])
    for fold,ck in enumerate(c['checkpoints'],1):
        ch=torch.load(ck,weights_only=True);head=cv.model.Head(ch['channels']);head.load_state_dict(ch['state_dict']);ids=np.where(c['fold']==fold)[0];prob[ids]=cv.predict(head,ff,ids)
    corrected=ious(prob>=c['threshold'][:,None,None],c['gt'])[0];original=ious(c['prob']>=c['threshold'][:,None,None],c['gt'])[0]
    pd.DataFrame(dict(image_id=c['names'],original=original,corrected=corrected,delta=corrected-original)).to_csv(OUT/'stn_feature_correction.csv',index=False)
    np.savez_compressed(OUT/'cache/stn_feature_corrected_probability.npz',probability=prob,names=c['names'])
    for file,data in [('spatial_controls',rows),('spatial_controls_per_image',detail),('cached_feature_statistics',feature_stats),('decoder_correlations',correlations),('geometry_stratified_metrics',geometry)]:pd.DataFrame(data).to_csv(OUT/f'{file}.csv',index=False)
    paths=[ROOT/'starter-kit/task-1/wmloc/utils.py',ROOT/'wmloc/watermarks.py',ROOT/'wmloc/data.py',ROOT/'checkpoints/pixelseal/checkpoint.pth',ROOT/'checkpoints/pixelseal_card.yaml',ROOT/'checkpoints/stegastamp/stega_stamp_logits.onnx',cv.OUT/'folds.json',ROOT/'data/train/metadata.json',ROOT/'data/train/masks_128.npz']
    import trustmark
    tm=Path(trustmark.__file__).parent/'models';paths.extend([tm/'decoder_Q.ckpt',tm/'trustmark_Q.yaml'])
    for method in METHODS:paths.append(cv.model.OUT/f'{method}_features.pt')
    provenance={str(p):sha(p) for p in paths}
    versions={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','pandas','scipy','scikit-learn','onnx','onnxruntime','trustmark','timm','matplotlib','Pillow']}
    save_json(OUT/'provenance.json',dict(sha256=provenance,versions=versions,python=sys.version,videoseal_commit=subprocess.check_output(['git','-C',str(ROOT/'checkpoints/videoseal'),'rev-parse','HEAD'],text=True).strip(),device='CPU',threads=4))
    print(pd.DataFrame(rows).to_string(index=False));print('FEATURE UNWARP',float(corrected.mean()),float((corrected-original).mean()))

if __name__=='__main__':main()
