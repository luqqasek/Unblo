"""Per-image macro diagnostics, tie-neutral oracle Top-K, and official IoU."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score


def ious(mask,gt):
    values=[]
    for c in (False,True):
        union=((mask==c)|(gt==c)).sum()
        values.append(float(((mask==c)&(gt==c)).sum()/union) if union else 1.)
    return (sum(values)/2,values[1],values[0])


def topk(prob,gt):
    """Expected precision under uniformly random tie-breaking at the K-th score."""
    k=int(gt.sum())
    if not k:return None
    cut=np.partition(prob.ravel(),prob.size-k)[prob.size-k]
    higher=prob>cut;equal=prob==cut
    return float((gt[higher].sum()+(k-higher.sum())*gt[equal].mean())/k)


def image_metrics(prob,gt,threshold):
    gt=gt.astype(bool);mask=prob>=threshold
    miou,fg,clean=ious(mask,gt)
    k=int(gt.sum());n=gt.size;area=k/n
    mixed=0<k<n
    # Exact expected AP for a uniformly random permutation (continuous random scores).
    harmonic=float(np.sum(1/np.arange(1,n+1)))
    random_ap=harmonic/n+(k-1)/(n-1)*(1-harmonic/n) if k else None
    return dict(miou=miou,fg_iou=fg,clean_iou=clean,area=area,pred_area=float(mask.mean()),
                auc=float(roc_auc_score(gt.ravel(),prob.ravel())) if mixed else None,
                ap=float(average_precision_score(gt.ravel(),prob.ravel())) if mixed else None,
                topk=topk(prob,gt),random_auc=.5 if mixed else None,
                random_ap=random_ap,random_topk=area if k else None,
                empty_miou=ious(np.zeros_like(gt),gt)[0],
                constant_miou=ious(np.full_like(gt,prob.mean()>=threshold),gt)[0])


def avg(rows,key):
    x=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(x)) if x else None


def aggregate(rows):
    out={k:avg(rows,k) for k in ('miou','fg_iou','clean_iou','auc','ap','topk','random_auc',
                                 'random_ap','random_topk','empty_miou','constant_miou')}
    small=[r for r in rows if 0<r['area']<=.05]
    large=[r for r in rows if r['area']>.5]
    out.update(n=len(rows),small_n=len(small),large_n=len(large))
    out.update({f'small_{k}':avg(small,k) for k in ('ap','auc','topk','fg_iou','miou','random_ap','random_topk','empty_miou')})
    out['large_clean_iou']=avg(large,'clean_iou')
    return out
