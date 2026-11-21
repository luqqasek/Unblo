"""All metrics are per image; topology is used only for evaluation, never inference."""
import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score,average_precision_score
HARMONIC_128=float(np.sum(1/np.arange(1,128*128+1)))


def one(p,g,t):
    g=g.astype(bool);pred=p>=t
    tp=int((pred&g).sum());fp=int((pred&~g).sum());fn=int((~pred&g).sum());tn=int((~pred&~g).sum())
    fg=tp/(tp+fp+fn) if tp+fp+fn else 1.;clean=tn/(tn+fp+fn) if tn+fp+fn else 1.
    k=int(g.sum());n=g.size;area=k/n
    cut=np.partition(p.ravel(),n-k)[n-k] if k else None
    high=p>cut if k else np.zeros_like(g);eq=p==cut if k else np.zeros_like(g)
    top=(g[high].sum()+(k-high.sum())*g[eq].mean())/k if k else None
    a=g&~ndi.binary_erosion(g,border_value=1);b=pred&~ndi.binary_erosion(pred,border_value=1)
    if a.any() and b.any():
        pr=(ndi.distance_transform_edt(~a)[b]<=2).mean();re=(ndi.distance_transform_edt(~b)[a]<=2).mean()
        bf=2*pr*re/(pr+re) if pr+re else 0.
    else:bf=0. if a.any() or b.any() else None
    holes=ndi.binary_fill_holes(g)&~g
    return dict(miou=(fg+clean)/2,fg_iou=fg,clean_iou=clean,area=area,pred_area=float(pred.mean()),
        auc=float(roc_auc_score(g.ravel(),p.ravel())) if 0<k<n else None,
        ap=float(average_precision_score(g.ravel(),p.ravel())) if 0<k<n else None,topk=float(top) if top is not None else None,
        empty_prediction=int(not pred.any()),missed=int(k>0 and tp==0),false_positive_area=fp/n,
        clean_fpr=fp/(fp+tn) if fp+tn else None,precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/k if k else None,area_error=abs(float(pred.mean())-area),boundary_f1=bf,
        hole_pixels=int(holes.sum()),hole_clean_recall=float((~pred)[holes].mean()) if holes.any() else None,
        empty_miou=(1-area)/2 if k else 1.,full_miou=area/2 if k<n else 1.,
        constant_miou=(area if p.mean()>=t else 1-area)/2 if 0<k<n else float((p.mean()>=t)==bool(k)),
        random_topk=area,random_ap=(HARMONIC_128/n+(k-1)/(n-1)*(1-HARMONIC_128/n)) if k else None)


def mean(rows,key):
    v=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def aggregate(rows):
    keys=['miou','fg_iou','clean_iou','auc','ap','topk','empty_miou','full_miou','constant_miou']
    out={k:mean(rows,k) for k in keys};out['n']=len(rows)
    for name,select,columns in [
        ('small',lambda r:0<r['area']<=.05,['miou','fg_iou','ap','auc','topk','empty_prediction','missed','false_positive_area','random_ap','random_topk','empty_miou']),
        ('medium',lambda r:.05<r['area']<=.5,['miou','precision','recall','area_error']),
        ('large',lambda r:r['area']>.5,['miou','clean_iou','clean_fpr','boundary_f1','hole_clean_recall'])]:
        subset=[r for r in rows if select(r)];out[name+'_n']=len(subset)
        out.update({name+'_'+k:mean(subset,k) for k in columns})
        if name=='large':out['large_holes_n']=sum(r['hole_pixels']>0 for r in subset)
    return out
