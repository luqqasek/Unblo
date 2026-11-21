"""Train-only cohort, immutable OOF sources, official scoring and forensic diagnostics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ('pixelseal', 'trustmark', 'stegastamp')
metric_source=ROOT/'starter-kit/task-1/wmloc/utils.py'
if not metric_source.exists():metric_source=HERE/'reference_metric.py'
spec = importlib.util.spec_from_file_location('official_forensics_metric', metric_source)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)

def save(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k,x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)): return [clean(x) for x in v]
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def cohort():
    base = HERE.parent / 'frozen_extractor/results/cv3'
    if not base.exists():
        # Portable evaluation/reporting from the self-contained result snapshot.
        z=np.load(OUT/'inputs/baseline_oof.npz')
        records=json.loads((OUT/'cohort.json').read_text())
        assert z['names'].tolist()==[r['image_id'] for r in records] and len(records)==510
        return records,z['probability'],z['target'].astype(bool),json.loads((OUT/'inputs/splits.json').read_text()),json.loads((OUT/'inference_trufor.json').read_text())['baseline_sources']
    splits = json.loads((base / 'folds.json').read_text())
    meta = {r['image_name']:r for r in json.loads((ROOT/'data/train/metadata.json').read_text())}
    records=[]; probabilities=[]; targets=[]; provenance=[]
    for family in METHODS:
        z = np.load(base / f'{family}_oof.npz')
        names=z['names'].tolist(); p=z['probability'].copy(); g=z['target'].astype(bool); t=z['threshold'].copy()
        folds=np.zeros(len(names),int)
        provenance.append(dict(path=str(base/f'{family}_oof.npz'),sha256=sha(base/f'{family}_oof.npz')))
        for fold,s in enumerate(splits[family],1):
            assert not set(s['train']) & set(s['validation'])
            ids=[names.index(n) for n in s['validation']]; folds[ids]=fold
            if family=='stegastamp':
                path=HERE.parent/'synthetic_pretrain_2k/results'/f'synthetic_fold{fold}_seed42.npz'
                zz=np.load(path); assert zz['names'].tolist()==s['validation']
                assert np.array_equal(zz['target'].astype(bool),g[ids])
                p[ids]=zz['probability']; t[ids]=float(zz['threshold'])
                provenance.append(dict(path=str(path),sha256=sha(path)))
        for i,n in enumerate(names):
            assert meta[n]['method']==family and folds[i]>0
            records.append(dict(image_id=n,family=family,fold=int(folds[i]),baseline_threshold=float(t[i]),
                                full_watermark=meta[n]['full_watermark']))
        probabilities.extend(p); targets.extend(g)
    assert len(records)==510 and len({r['image_id'] for r in records})==510
    official_gt=np.load(ROOT/'data/train/masks_128.npz')
    gt_lookup=dict(zip(official_gt['names'],official_gt['masks']))
    assert all(np.array_equal(gt_lookup[r['image_id']].astype(bool),y) for r,y in zip(records,targets))
    return records,np.stack(probabilities),np.stack(targets),splits,provenance

def curve(p,g,thresholds):
    thresholds=np.asarray(thresholds,dtype=p.dtype)
    ix=np.argsort(p.ravel(),kind='stable'); s=p.ravel()[ix]; y=g.ravel()[ix]
    c=np.r_[0,np.cumsum(y)]; k=np.searchsorted(s,thresholds,side='left')
    fn=c[k]; tp=c[-1]-fn; tn=k-fn; fp=len(y)-k-tp
    fg=np.divide(tp,tp+fp+fn,out=np.ones(len(k)),where=(tp+fp+fn)!=0)
    bg=np.divide(tn,tn+fp+fn,out=np.ones(len(k)),where=(tn+fp+fn)!=0)
    return (fg+bg)/2

def bf1(g,p):
    # Ignore exterior image edges; tolerance two evaluation-grid pixels (=8 RGB pixels).
    a=g & ~ndi.binary_erosion(g,border_value=1); b=p & ~ndi.binary_erosion(p,border_value=1)
    if not a.any() or not b.any(): return 0. if a.any() or b.any() else np.nan
    pr=(ndi.distance_transform_edt(~a)[b]<=2).mean(); re=(ndi.distance_transform_edt(~b)[a]<=2).mean()
    return float(2*pr*re/(pr+re)) if pr+re else 0.

def measure(p,g,t,oracle=False):
    pred=p>=t; tp,fp,fn,tn=[int(x.sum()) for x in (pred&g,pred&~g,~pred&g,~pred&~g)]
    fg=tp/(tp+fp+fn) if tp+fp+fn else 1.; bg=tn/(tn+fp+fn) if tn+fp+fn else 1.
    score=float(official.miou_image(pred,g)); assert abs(score-(fg+bg)/2)<1e-12
    both=g.any() and not g.all(); k=int(g.sum()); top=np.nan
    if both:
        cut=np.partition(p.ravel(),-k)[-k]; above=p>cut; ties=p==cut
        top=float(((above&g).sum()+(k-above.sum())*g[ties].mean())/k)
    holes=ndi.binary_fill_holes(g)&~g
    labels,n=ndi.label(pred); sizes=np.bincount(labels.ravel())[1:]
    holes_labels,nh=ndi.label(holes)
    recovered=sum((~pred)[holes_labels==j].mean()>=.5 for j in range(1,nh+1))
    corr=float(np.corrcoef(g.ravel(),p.ravel())[0,1]) if both and np.std(p)>1e-10 else np.nan
    r=dict(miou=score,fg_iou=fg,clean_iou=bg,auc=float(roc_auc_score(g.ravel(),p.ravel())) if both else np.nan,
           ap=float(average_precision_score(g.ravel(),p.ravel())) if both else np.nan,
           clean_ap=float(average_precision_score((~g).ravel(),(1-p).ravel())) if both else np.nan,
           topk_precision=top,topk_lift=top/g.mean() if both else np.nan,spatial_correlation=corr,
           boundary_f1=bf1(g,pred),internal_boundary_f1=bf1(holes,ndi.binary_fill_holes(pred)&~pred) if holes.any() else np.nan,
           hole_pixels=int(holes.sum()),hole_recall=float((~pred)[holes].mean()) if holes.any() else np.nan,
           hole_component_recall=recovered/nh if nh else np.nan,clean_fpr=fp/(fp+tn) if fp+tn else np.nan,
           tp_coverage=tp/(tp+fn) if tp+fn else np.nan,false_positive_area=fp/g.size,
           complete_miss=float(tp==0) if g.any() else np.nan,gt_area=float(g.mean()),pred_area=float(pred.mean()),
           components=n,largest_component_area=float(sizes.max()/g.size) if n else 0.,threshold=float(t),
           tp=tp,fp=fp,fn=fn,tn=tn)
    if oracle:
        # Exact best threshold over every distinct probability, including all/empty.
        cuts=np.r_[np.unique(p),np.nextafter(np.float32(p.max()),np.float32(np.inf))]
        scores=curve(p,g,cuts); j=int(scores.argmax())
        r.update(oracle_threshold=float(cuts[j]),oracle_threshold_miou=float(scores[j]))
    return r

def ci(delta):
    d=np.asarray(delta,float); d=d[np.isfinite(d)]
    if not len(d): return [np.nan,np.nan]
    rng=np.random.default_rng(20261121)
    return np.quantile(d[rng.integers(0,len(d),(5000,len(d)))].mean(1),[.025,.975]).tolist()
