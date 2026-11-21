"""Immutable real-only sources, matched checkpoints and official training settings."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE.parent/'frozen_extractor'))
import cross_validate as cv
METHODS=('pixelseal','trustmark','stegastamp')
SEEDS=(42,7,123)


def save(path,value):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def feature_path(method):
    return HERE.parent/'stegastamp_stn_alignment/results/aligned_features.pt' if method=='stegastamp' else cv.model.OUT/f'{method}_features.pt'


def load_data(method,device='cpu'):
    cache=torch.load(feature_path(method),weights_only=True)
    z=np.load(cv.OUT/f'{method}_oof.npz')
    assert cache['names']==z['names'].tolist()
    return dict(names=cache['names'],features=[f.to(device) for f in cache['features']],
                gt=z['target'].astype(bool),channels=[f.shape[1] for f in cache['features']])


def splits(method):
    if method=='stegastamp':return json.loads((HERE.parent/'stegastamp_stn_alignment/results/splits.json').read_text())
    return json.loads((cv.OUT/'folds.json').read_text())[method]


def split_ids(data,method,fold):
    s=splits(method)[fold-1];idx={n:i for i,n in enumerate(data['names'])}
    fit=np.array([idx[n] for n in s['train']]);val=np.array([idx[n] for n in s['validation']])
    parts=s['inner_validation'] if method=='stegastamp' else [s['inner_validation']]
    parts=[np.array([idx[n] for n in part]) for part in parts]
    trains=[np.setdiff1d(fit,p) for p in parts] if method=='stegastamp' else [np.array([idx[n] for n in s['inner_train']])]
    assert not set(fit)&set(val)
    for tr,va in zip(trains,parts):assert not set(tr)&set(va) and set(tr)|set(va)==set(fit)
    return fit,val,trains,parts


def baseline_path(method,fold,seed):
    if method=='stegastamp':return HERE.parent/'stegastamp_stn_alignment/results'/f'stegastamp_fold{fold}_seed{seed}.pt'
    if seed==42:return cv.OUT/f'{method}_fold{fold}_head.pt'
    return HERE.parent/'rgb_srm/results'/f'{method}_A_fold{fold}_seed{seed}.pt'


def baseline(method,fold,seed,data,device):
    path=baseline_path(method,fold,seed)
    ck=torch.load(path,weights_only=True)
    state=ck['state_dict']
    if any(k.startswith('head.') for k in state):state={k.removeprefix('head.'):v for k,v in state.items()}
    model=cv.model.Head(data['channels']).to(device);model.load_state_dict(state);model.eval()
    return model,ck


def regime(gt):return np.digitize(gt.mean((-2,-1)),[.05,.5],right=True)


@torch.inference_mode()
def predict(head,features,ids,clean=False):
    head.eval()
    p=torch.cat([head([f[ids[j:j+8]] for f in features]).sigmoid() for j in range(0,len(ids),8)])
    return (1-p if clean else p).cpu().numpy()


def miou(p,g,t=.5):
    if np.ndim(t):t=np.asarray(t)[:,None,None]
    pred=p>=t;v=[]
    for c in (0,1):
        u=((pred==c)|(g==c)).sum((-2,-1));i=((pred==c)&(g==c)).sum((-2,-1))
        v.append(np.divide(i,u,out=np.ones_like(i,dtype=float),where=u>0))
    return (v[0]+v[1])/2


def setup_gpu(device):
    torch.set_num_threads(2);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True


def provenance():
    paths=[cv.OUT/'folds.json',ROOT/'data/train/metadata.json',ROOT/'data/train/masks_128.npz',
           HERE.parent/'stegastamp_stn_alignment/results/splits.json']
    paths += [feature_path(m) for m in METHODS]
    paths += [baseline_path(m,f,s) for m in METHODS for f in (1,2,3) for s in SEEDS]
    paths += [p for name in ('frozen_extractor','stegastamp_stable_selection','decoder_evidence')
              for p in (HERE.parent/name/'results/full_train').glob('*') if p.is_file()]
    config=dict(regimes='small <=.05; medium (.05,.5]; large >.5',seeds=list(SEEDS),
        data='real train only; no competition test or synthetic data',
        baseline='historical frozen PS/TM seed42; existing matched RGB-experiment A seeds7/123; active inverse-STN SS all seeds',
        gate_models=['majority','bit_lr','decoder_lr','decoder_hgb','pooled_lr'],
        gate_selection='lowest saved-inner log loss among bit_lr, decoder_lr, decoder_hgb, pooled_lr; outer labels never select model',
        gate_go='OOF selected macroF1 > majority + .15, small recall >= .60, large recall >= .85, small->large and large->small <= .10',
        calibration='global t in [.3,.4,.5,.6,.7]; 27 fixed offsets [-.7,0,.7]^3 applied to global threshold logit using cross-fitted regime probabilities',
        experts='shared frozen baseline projection trunk; frozen universal fuse; S/L copy universal fuse, L final logit sign reversed; S image weight4 small, L weight4 large; unchanged BCE+Dice on foreground/true clean respectively',
        expert_selection='saved inner splits; max50 patience8; mean mIoU@.5 selects epoch, fresh copy of existing outer M then fit selected epochs',
        promotion='fold1 seed42: A2 small AP and TopK improve and mIoU >= A0-.01; A3 large cleanIoU and internalBoundaryF1 improve and mIoU >= A0-.01; A4 either localization gate and mIoU >= A0-.01; A5/A6/hard and U retained as controls for confirmed families',
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},torch=str(torch.__version__))
    path=OUT/'config.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==config
    else:save(path,config)
    save(OUT/'splits.json',{m:splits(m) for m in METHODS})
    return config
