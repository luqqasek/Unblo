"""Shared read-only sources and official scoring for the diagnostic audit."""
import importlib.util
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
for p in (OUT, OUT/'figures', OUT/'examples', OUT/'cache'):
    p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent/'frozen_extractor'))
import cross_validate as cv
spec = importlib.util.spec_from_file_location('official_scoring', ROOT/'starter-kit/task-1/wmloc/utils.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
torch.set_num_threads(4)
METHODS = ('pixelseal', 'trustmark', 'stegastamp')
THRESHOLDS = np.r_[np.linspace(0, 1, 101), 1.000001]

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save_json(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load_cohort(method, variant='original', seed=42):
    z = np.load(cv.OUT/f'{method}_oof.npz')
    names = z['names'].tolist()
    prob, gt, thresholds = z['probability'], z['target'].astype(bool), z['threshold']
    splits = json.loads((cv.OUT/'folds.json').read_text())[method]
    folds = np.zeros(len(names), int)
    checkpoints=[]
    for fold, split in enumerate(splits, 1):
        ids = [names.index(n) for n in split['validation']]
        folds[ids]=fold
        assert not set(split['validation']) & set(split['train'])
        if variant=='stable':
            path=HERE.parent/'stegastamp_stable_selection/results'/f'stegastamp_fold{fold}_seed{seed}.npz'
            zz=np.load(path)
            assert zz['names'].tolist()==split['validation']
            prob[ids]=zz['probability']; thresholds[ids]=zz['threshold']
            ck=path.with_suffix('.pt')
        else: ck=cv.OUT/f'{method}_fold{fold}_head.pt'
        checkpoints.append(ck)
    assert np.all(folds>0) and len(set(names))==170
    return dict(method=method,variant=variant,seed=seed,names=names,prob=prob,gt=gt,threshold=thresholds,fold=folds,checkpoints=checkpoints)

def ious(pred,gt):
    tp=np.count_nonzero(pred & gt,axis=(-2,-1)).astype(float)
    fp=np.count_nonzero(pred & ~gt,axis=(-2,-1)).astype(float)
    fn=np.count_nonzero(~pred & gt,axis=(-2,-1)).astype(float)
    tn=np.count_nonzero(~pred & ~gt,axis=(-2,-1)).astype(float)
    fg=np.divide(tp,tp+fp+fn,out=np.ones_like(tp),where=tp+fp+fn!=0)
    bg=np.divide(tn,tn+fp+fn,out=np.ones_like(tn),where=tn+fp+fn!=0)
    return (fg+bg)/2,fg,bg,tp,fp,fn,tn

def threshold_scores(p,g):
    return np.stack([ious(p>=t,g)[0] for t in THRESHOLDS],axis=-1)
