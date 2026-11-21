"""Resume-safe GPU workers. Stage one screens B/C; stage two confirms D and gate passers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from models import Model, prepare_input, srm_kernels, cv

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
ROOT=HERE.parents[1]
METHODS=('pixelseal','trustmark','stegastamp')
SEEDS=(42,7,123)


def save(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def feature_path(method):
    return (HERE.parent/'stegastamp_stn_alignment/results/aligned_features.pt' if method=='stegastamp'
            else cv.model.OUT/f'{method}_features.pt')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[cv.OUT/'folds.json',HERE.parent/'stegastamp_stable_selection/results/splits.json',
             ROOT/'data/train/masks_128.npz',ROOT/'data/train/metadata.json',
             HERE/'models.py',HERE/'run.py',cv.HERE/'run.py',cv.HERE/'cross_validate.py']
    sources += [feature_path(m) for m in METHODS]
    # Record retained best artifacts and all source models, then verify at completion.
    sources += [p for exp in ('frozen_extractor','stegastamp_stable_selection','decoder_evidence')
                for p in (HERE.parent/exp/'results/full_train').glob('*') if p.is_file()]
    config=dict(training={**cv.model.CONFIG,'device':'cuda','precision':'float32; TF32 disabled'},
        seeds=list(SEEDS),variants=['A','B','C','D'],input_size=256,filter_size=512,
        branch_channels=[16,24,32,32],branch_strides=[1,2,2,2],srm=srm_kernels().tolist(),
        fusion='D adds four RGB/SRM projections to four frozen-extractor projections at 32x32; unchanged fuse and output',
        selection='unchanged saved inner80/20 PS/TM; saved three inner folds SS; mIoU@.5 epoch, inner threshold, fresh refit',
        screening='fold1 seed42 A/B/C before D; B/C promoted per method only if small AP AND tie-aware TopK exceed A and macro AP >= A - .01',
        confirmation='A,D all three folds and seeds; gate-passing B/C all three folds and seeds; screening fold reported separately from folds2/3',
        sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        torch=str(torch.__version__),cuda=torch.version.cuda,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    path=OUT/'config.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==config, 'Configuration/source changed: refuse stale resume'
    else:
        save(path,config)
    splits=json.loads((cv.OUT/'folds.json').read_text())
    splits['stegastamp']=json.loads((HERE.parent/'stegastamp_stable_selection/results/splits.json').read_text())
    save(OUT/'splits.json',splits)
    return digest(path)


def load_data(method,device):
    cache=torch.load(feature_path(method),weights_only=True)
    names=cache['names']
    source=np.load(cv.OUT/f'{method}_oof.npz')
    assert names==source['names'].tolist() and len(set(names))==170
    gt=source['target'].astype(bool)
    masks=np.load(ROOT/'data/train/masks_128.npz')
    lookup=dict(zip(masks['names'].tolist(),masks['masks']))
    assert all(np.array_equal(lookup[n],g) for n,g in zip(names,gt))
    path=OUT/f'{method}_rgb_srm.pt'
    if path.exists():
        cached=torch.load(path,weights_only=True)
        assert cached['names']==names and cached['preprocessing_sha256']==digest(HERE/'models.py')
        rgb=cached['input'].to(device)
    else:
        chunks=[]
        for start in range(0,len(names),8):
            a=np.stack([np.asarray(Image.open(ROOT/'data/train/images'/f'{n}.png').convert('RGB'))
                        for n in names[start:start+8]])
            assert a.shape[1:]==(512,512,3)
            t=torch.from_numpy(a).permute(0,3,1,2).to(device,dtype=torch.float32)/255
            chunks.append(prepare_input(t).cpu())
        cached=dict(names=names,input=torch.cat(chunks),preprocessing_sha256=digest(HERE/'models.py'))
        # Multiple device workers may prepare the same cache; atomic replace.
        tmp=path.with_suffix(f'.{os.getpid()}.tmp')
        torch.save(cached,tmp);os.replace(tmp,path)
        rgb=cached['input'].to(device)
    return dict(names=names,features=[f.to(device) for f in cache['features']],rgb=rgb,
                gt=gt,target=torch.from_numpy(gt).to(device,dtype=torch.float32))


def initialize(variant,data,seed,device):
    torch.manual_seed(seed)
    model=Model(variant,[f.shape[1] for f in data['features']]).to(device)
    return model,torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001),np.random.default_rng(seed)


def train_epoch(model,opt,rng,data,ids):
    model.train()
    shuffled=rng.permutation(ids)
    for start in range(0,len(shuffled),8):
        batch=shuffled[start:start+8]
        opt.zero_grad(set_to_none=True)
        logits=model([f[batch] for f in data['features']],data['rgb'][batch])
        target=data['target'][batch]
        p=logits.sigmoid()
        dice=1-((2*(p*target).sum((1,2))+1)/(p.sum((1,2))+target.sum((1,2))+1)).mean()
        loss=.5*F.binary_cross_entropy_with_logits(logits,target)+.5*dice
        loss.backward();opt.step()


@torch.inference_mode()
def predict(model,data,ids):
    model.eval()
    return torch.cat([model([f[ids[j:j+8]] for f in data['features']],data['rgb'][ids[j:j+8]]).sigmoid()
                      for j in range(0,len(ids),8)])


def gpu_score(p,g,t):
    pred=p>=t;g=g.bool()
    values=[]
    for c in (False,True):
        union=((pred==c)|(g==c)).sum((1,2));inter=((pred==c)&(g==c)).sum((1,2))
        values.append(torch.where(union>0,inter.double()/union,1.))
    return float(((values[0]+values[1])/2).mean())


def fit(method,variant,fold,seed,device,data,config_hash):
    stem=f'{method}_{variant}_fold{fold}_seed{seed}'
    path=OUT/f'{stem}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['config_sha256']==config_hash
        return r
    start=time.perf_counter()
    split=json.loads((OUT/'splits.json').read_text())[method][fold-1]
    index={n:i for i,n in enumerate(data['names'])}
    fit_ids=np.array([index[n] for n in split['train']]);val=np.array([index[n] for n in split['validation']])
    inner=split['inner_validation'] if method=='stegastamp' else [split['inner_validation']]
    parts=[np.array([index[n] for n in ns]) for ns in inner]
    assert not set(fit_ids)&set(val)
    assert all(set(part)<=set(fit_ids) for part in parts)
    trains=[np.setdiff1d(fit_ids,p) for p in parts]
    if method!='stegastamp':assert set(trains[0])=={index[n] for n in split['inner_train']}
    else:assert sorted(np.concatenate(parts).tolist())==sorted(fit_ids.tolist())
    # Preserve saved training order for the original 80/20 pipeline.
    if method!='stegastamp':trains=[np.array([index[n] for n in split['inner_train']])]
    models=[initialize(variant,data,seed,device) for _ in parts]
    target=data['target'][np.concatenate(parts)]
    best,stale,history=-1,0,[]
    for epoch in range(1,51):
        probs=[]
        for (model,opt,rng),train_ids,valid_ids in zip(models,trains,parts):
            train_epoch(model,opt,rng,data,train_ids)
            probs.append(predict(model,data,valid_ids))
        pooled=torch.cat(probs)
        metric=gpu_score(pooled,target,.5);history.append(metric)
        if metric>best:
            best,stale,selected_epoch=metric,0,epoch
            best_prob=pooled.clone()
        else:stale+=1
        if epoch%5==0:print(f'{stem} inner epoch={epoch} miou={metric:.5f}',flush=True)
        if stale>=8:break
    scores=[dict(threshold=t,miou=gpu_score(best_prob,target,t)) for t in cv.model.CONFIG['threshold_candidates']]
    threshold=max(scores,key=lambda r:r['miou'])['threshold']
    del models,pooled,probs,model,opt
    model,opt,rng=initialize(variant,data,seed,device)
    for _ in range(selected_epoch):train_epoch(model,opt,rng,data,fit_ids)
    prob=predict(model,data,val).cpu().numpy()
    assert np.isfinite(prob).all()
    official=cv.model.metrics(prob,data['gt'][val],threshold)
    assert abs(official['miou']-gpu_score(torch.from_numpy(prob),torch.from_numpy(data['gt'][val]),threshold))<1e-12
    params=sum(p.numel() for p in model.parameters())
    ck=dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},variant=variant,
            channels=[f.shape[1] for f in data['features']],epoch=selected_epoch,threshold=threshold,
            train_names=split['train'],validation_names=split['validation'],config_sha256=config_hash)
    torch.save(ck,OUT/f'{stem}.pt')
    restored=Model(variant,ck['channels']).to(device);restored.load_state_dict(ck['state_dict'])
    reproduction=float(np.max(np.abs(predict(restored,data,val).cpu().numpy()-prob)))
    assert reproduction<1e-6
    np.savez_compressed(OUT/f'{stem}.npz',names=split['validation'],probability=prob,
                        mask=prob>=threshold,target=data['gt'][val],threshold=threshold)
    from evaluate import image_metrics, aggregate
    per_image=[dict(name=n,**image_metrics(p,g,threshold)) for n,p,g in zip(split['validation'],prob,data['gt'][val])]
    result=dict(method=method,variant=variant,fold=fold,seed=seed,epoch=selected_epoch,threshold=threshold,
                inner_history=history,inner_threshold_scores=scores,parameters_trainable=params,
                seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(device),
                config_sha256=config_hash,checkpoint_reproduction_max_error=reproduction,
                metrics=aggregate(per_image),per_image=per_image)
    save(path,result)
    print(f'COMPLETE {stem}: {result["metrics"]} seconds={result["seconds"]:.1f}',flush=True)
    return result


def worker(args):
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config_hash=digest(OUT/'config.json')
    jobs=json.loads(Path(args.jobs).read_text())
    previous=None;data=None
    for job in jobs:
        method=job['method']
        if method!=previous:
            del data;torch.cuda.empty_cache()
            data=load_data(method,torch.device(f'cuda:{args.device}'));previous=method
        fit(**job,device=torch.device(f'cuda:{args.device}'),data=data,config_hash=config_hash)


def dispatch(jobs,stage):
    pending=[j for j in jobs if not (OUT/f'{j["method"]}_{j["variant"]}_fold{j["fold"]}_seed{j["seed"]}.json').exists()]
    # Faster RTX gets twice the work; independent jobs, no inter-GPU communication.
    bins=[[],[]]
    for i,j in enumerate(pending):bins[1 if i%3==2 else 0].append(j)
    processes=[]
    for gpu,items in enumerate(bins):
        if not items:continue
        items.sort(key=lambda j:(j['method'],j['fold'],j['seed'],j['variant']))
        jobpath=OUT/f'{stage}_gpu{gpu}_jobs.json';save(jobpath,items)
        log=(OUT/f'{stage}_gpu{gpu}.log').open('a',encoding='utf-8')
        p=subprocess.Popen([sys.executable,str(HERE/'run.py'),'--jobs',str(jobpath),'--device',str(gpu)],
                           stdout=log,stderr=subprocess.STDOUT)
        processes.append((p,log))
    for p,log in processes:
        code=p.wait();log.close()
        if code:raise RuntimeError(f'GPU worker {p.pid} failed: {code}; see {stage} logs')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs');ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--stage',choices=['screen','confirm','all'],default='all');args=ap.parse_args()
    if args.jobs:return worker(args)
    assert torch.cuda.is_available() and torch.cuda.device_count()>=2
    prepare()
    if args.stage in ('screen','all'):
        dispatch([dict(method=m,variant=v,fold=1,seed=42) for m in METHODS for v in ('A','B','C')],'screen')
    gates=[]
    for m in METHODS:
        a=json.loads((OUT/f'{m}_A_fold1_seed42.json').read_text())['metrics']
        for v in ('B','C'):
            c=json.loads((OUT/f'{m}_{v}_fold1_seed42.json').read_text())['metrics']
            passed=c['small_ap']>a['small_ap'] and c['small_topk']>a['small_topk'] and c['ap']>=a['ap']-.01
            gates.append(dict(method=m,variant=v,passed=bool(passed),baseline=a,candidate=c))
    save(OUT/'screening.json',gates)
    if args.stage in ('confirm','all'):
        choices=[(m,v) for m in METHODS for v in ('A','D')]+[(g['method'],g['variant']) for g in gates if g['passed']]
        dispatch([dict(method=m,variant=v,fold=f,seed=s) for m,v in choices for f in (1,2,3) for s in SEEDS],'confirm')
        config=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
        assert all(digest(ROOT/p)==h for p,h in config['sources_sha256'].items())
        save(OUT/'completion.json',dict(status='complete',expected_runs=len(choices)*9+sum(not g['passed'] for g in gates),
            source_hashes_unchanged=True))
    print('STAGE COMPLETE',args.stage,flush=True)


if __name__=='__main__':main()
