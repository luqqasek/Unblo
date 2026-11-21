"""Deterministic full factorial: 2k/4k x 2500/5000 updates, then matched nested real FT."""
import argparse
import time
import torch.nn.functional as F
from scaling_common import *

cv = module('scaling_cv', HERE.parent / 'frozen_extractor/cross_validate.py')

class Data:
    def __init__(self, features, target, device, names=None):
        self.features=features; self.target=target; self.device=torch.device(device); self.names=names
        self.channels=[f.shape[1] for f in features]
    def batch(self, ids):
        xs=[]
        for f in self.features:
            xs.append(f[ids] if isinstance(f,torch.Tensor) else torch.from_numpy(np.asarray(f[ids])).to(self.device))
        y=self.target[ids] if isinstance(self.target,torch.Tensor) else torch.from_numpy(np.asarray(self.target[ids])).to(self.device,dtype=torch.float32)
        return xs,y

def initialize(data, seed, state=None):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head=cv.model.Head(data.channels).to(data.device)
    if state is not None:head.load_state_dict(state)
    return head,torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001),np.random.default_rng(seed)

def state_cpu(head):return {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}

def epoch(head,opt,rng,data,ids):
    head.train(); order=rng.permutation(ids); total=0.
    for start in range(0,len(order),8):
        batch=order[start:start+8];xs,y=data.batch(batch)
        opt.zero_grad(set_to_none=True);logits=head(xs);p=logits.sigmoid()
        dice=1-((2*(p*y).sum((1,2))+1)/(p.sum((1,2))+y.sum((1,2))+1)).mean()
        loss=.5*F.binary_cross_entropy_with_logits(logits,y)+.5*dice
        loss.backward();opt.step();total+=float(loss.detach())*len(batch)
    return total/len(order)

@torch.inference_mode()
def predict(head,data,ids):
    head.eval();probs=[]
    for start in range(0,len(ids),8):
        xs,_=data.batch(ids[start:start+8]);probs.append(head(xs).sigmoid())
    return torch.cat(probs)

def gpu_score(p,g,t):
    pred=p>=t;g=g.bool();values=[]
    for c in (False,True):
        union=((pred==c)|(g==c)).sum((1,2));inter=((pred==c)&(g==c)).sum((1,2))
        values.append(torch.where(union>0,inter.double()/union,1.))
    return float(((values[0]+values[1])/2).mean())

def load_real(method,device):
    c=torch.load(real_source(method),weights_only=True)
    z=np.load(HERE.parent/f'frozen_extractor/results/cv3/{method}_oof.npz')
    assert c['names']==z['names'].tolist()
    gt=z['target'].astype(np.uint8)
    return Data([f.to(device) for f in c['features']],torch.from_numpy(gt).to(device,dtype=torch.float32),device,c['names'])

def load_synthetic(method,device):
    dest=OUT/'data'/method
    assert (dest/'complete.json').exists()
    features=[np.load(dest/f'features_{k}.npy',mmap_mode='r') for k in range(4)]
    targets=np.load(dest/'targets.npy',mmap_mode='r')
    assert all(len(f)==4200 for f in features) and len(targets)==4200
    return Data(features,targets,device)

def pretrain(method,n,seed,data,config_hash):
    dest=OUT/'runs'/method;prefix=f'pretrain_n{n}_seed{seed}'
    last=dest/f'{prefix}_latest.pt';end=20 if n==2000 else 10
    milestones={10:'n2k_s2500',20:'n2k_s5000'} if n==2000 else {5:'n4k_s2500',10:'n4k_s5000'}
    if all((dest/f'{v}_seed{seed}_pretrain.pt').exists() for v in milestones.values()):return
    head,opt,rng=initialize(data,seed);start_epoch=0;history=[];seconds=0.
    if last.exists():
        ck=torch.load(last,weights_only=True,map_location=data.device)
        assert ck['config_sha256']==config_hash
        head.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng']
        start_epoch=ck['epoch'];history=ck['history'];seconds=ck['seconds']
    ids=train_ids(n);valid=np.arange(2000,2200)
    _,valid_gt=data.batch(valid)
    for e in range(start_epoch+1,end+1):
        started=time.perf_counter();loss=epoch(head,opt,rng,data,ids)
        p=predict(head,data,valid);score=gpu_score(p,valid_gt,.5)
        seconds+=time.perf_counter()-started
        history.append(dict(epoch=e,loss=loss,synthetic_validation_miou=score))
        payload=dict(state_dict=state_cpu(head),optimizer=opt.state_dict(),rng=rng.bit_generator.state,epoch=e,
                     seed=seed,n=n,updates=e*n//8,history=history,seconds=seconds,config_sha256=config_hash)
        torch.save(payload,last.with_suffix('.tmp'));os.replace(last.with_suffix('.tmp'),last)
        if e in milestones:
            variant=milestones[e];path=dest/f'{variant}_seed{seed}_pretrain.pt'
            torch.save(dict(state_dict=payload['state_dict'],channels=data.channels,seed=seed,n=n,epoch=e,
                            updates=e*n//8,config_sha256=config_hash),path)
            np.savez_compressed(dest/f'{variant}_seed{seed}_synthetic_validation.npz',probability=p.cpu().numpy(),target=valid_gt.cpu().numpy(),threshold=.5)
            save(dest/f'{variant}_seed{seed}_pretrain.json',dict(method=method,variant=variant,seed=seed,n=n,epochs=e,updates=e*n//8,
                 synthetic_validation_miou=score,seconds=seconds,history=history,checkpoint_sha256=sha(path),config_sha256=config_hash))
            if method=='stegastamp' and variant=='n2k_s2500':
                old=torch.load(OLD/f'results/pretrain_seed{seed}.pt',weights_only=True)
                errors={k:float((v-old['state_dict'][k]).abs().max()) for k,v in payload['state_dict'].items()}
                save(dest/f'prior_reproduction_seed{seed}.json',dict(max_parameter_error=max(errors.values()),errors=errors))
                assert max(errors.values())<1e-6,'Original 2k reproduction mismatch: inspect before claiming a matched comparison.'
        print(f'{method} PRETRAIN n={n} seed={seed} epoch={e}/{end} loss={loss:.4f} validation={score:.5f}',flush=True)

def fit(method,variant,seed,split,data,state,config_hash):
    dest=OUT/'runs'/method;stem=f'{variant}_fold{split["fold"]}_seed{seed}'
    path=dest/f'{stem}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['config_sha256']==config_hash;return
    index={n:i for i,n in enumerate(data.names)}
    fit_ids=np.array([index[n] for n in split['train']]);val=np.array([index[n] for n in split['validation']])
    inner=split['inner_validation'] if method=='stegastamp' else [split['inner_validation']]
    parts=[np.array([index[n] for n in part]) for part in inner]
    train_parts=[np.setdiff1d(fit_ids,p) for p in parts] if method=='stegastamp' else [np.array([index[n] for n in split['inner_train']])]
    assert not set(fit_ids)&set(val)
    assert all(not set(a)&set(b) and set(a)|set(b)==set(fit_ids) for a,b in zip(train_parts,parts))
    started=time.perf_counter();models=[initialize(data,seed,state) for _ in parts]
    target=data.target[np.concatenate(parts)];best,stale=-1,0;history=[]
    for e in range(1,51):
        probs=[];losses=[]
        for (head,opt,rng),train,valid in zip(models,train_parts,parts):
            losses.append(epoch(head,opt,rng,data,train));probs.append(predict(head,data,valid))
        pooled=torch.cat(probs);score=gpu_score(pooled,target,.5)
        history.append(dict(epoch=e,miou=score,loss=float(np.mean(losses))))
        if score>best:best,stale,selected=score,0,e;best_prob=pooled.clone()
        else:stale+=1
        if stale>=8:break
    scores=[dict(threshold=t,miou=gpu_score(best_prob,target,t)) for t in (.3,.4,.5,.6,.7)]
    threshold=max(scores,key=lambda r:r['miou'])['threshold']
    del models,head,opt,pooled,probs
    head,opt,rng=initialize(data,seed,state)
    for _ in range(selected):epoch(head,opt,rng,data,fit_ids)
    prob=predict(head,data,val).cpu().numpy();gt=data.target[val].cpu().numpy().astype(np.uint8)
    measured=cv.model.metrics(prob,gt,threshold)
    assert abs(measured['miou']-gpu_score(torch.from_numpy(prob),torch.from_numpy(gt),threshold))<1e-12
    ck=dict(state_dict=state_cpu(head),channels=data.channels,method=method,variant=variant,seed=seed,epoch=selected,threshold=threshold,
            train_names=split['train'],validation_names=split['validation'],config_sha256=config_hash,
            pretrain_sha256=None if state is None else sha(dest/f'{variant}_seed{seed}_pretrain.pt'))
    torch.save(ck,dest/f'{stem}.pt')
    reloaded,_,_=initialize(data,seed,torch.load(dest/f'{stem}.pt',weights_only=True)['state_dict'])
    np.testing.assert_array_equal(predict(reloaded,data,val).cpu().numpy(),prob)
    np.savez_compressed(dest/f'{stem}.npz',probability=prob,target=gt,names=split['validation'],threshold=threshold)
    save(path,dict(method=method,variant=variant,seed=seed,fold=split['fold'],epoch=selected,threshold=threshold,
                  miou=measured['miou'],fg_iou=measured['iou_watermark'],clean_iou=measured['iou_clean'],
                  per_image=measured['per_image'],names=split['validation'],inner_history=history,inner_threshold_scores=scores,
                  fixed_half_miou=cv.model.metrics(prob,gt,.5)['miou'],seconds=time.perf_counter()-started,config_sha256=config_hash))
    print(f'{method} COMPLETE {stem} mIoU={measured["miou"]:.6f} FG={measured["iou_watermark"]:.6f} epoch={selected} t={threshold}',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=METHODS,required=True);ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args();setup(args.device);method=args.method
    dest=OUT/'runs'/method;dest.mkdir(parents=True,exist_ok=True)
    config=dict(method=method,seeds=[42,7,123],variants=list(VARIANTS),device=args.device,gpu=torch.cuda.get_device_name(torch.device(args.device)),
                torch=str(torch.__version__),tf32=False,deterministic=True,batch_size=8,pretrain_lr=.001,finetune_lr=.001,
                fine_max_epochs=50,patience=8,thresholds=[.3,.4,.5,.6,.7],code_sha256=sha(__file__),common_sha256=sha(HERE/'scaling_common.py'),
                dataset_config_sha256=sha(OUT/'data'/method/'config.json'),manifest_sha256=sha(OUT/'data'/method/'manifest.json'),
                real_source_sha256=sha(real_source(method)),head_code_sha256=sha(cv.HERE/'run.py'),
                existing_submission_sha256=sha(OLD/'results/full_train/submission.pt'),folds=splits(method))
    if (dest/'config.json').exists():assert json.loads((dest/'config.json').read_text())==config,'Do not mix stale checkpoints.'
    else:save(dest/'config.json',config)
    config_hash=sha(dest/'config.json');real=load_real(method,args.device);synthetic=load_synthetic(method,args.device)
    for seed in (42,7,123):
        for n in (2000,4000):pretrain(method,n,seed,synthetic,config_hash)
        for variant in VARIANTS:
            state=None if variant=='real_only' else torch.load(dest/f'{variant}_seed{seed}_pretrain.pt',weights_only=True)['state_dict']
            if state is not None and not (dest/f'{variant}_seed{seed}_synthetic_only.npz').exists():
                head,_,_=initialize(real,seed,state);p=predict(head,real,np.arange(170)).cpu().numpy()
                np.savez_compressed(dest/f'{variant}_seed{seed}_synthetic_only.npz',probability=p,target=real.target.cpu().numpy(),names=real.names,threshold=.5)
                del head
            for split in config['folds']:fit(method,variant,seed,split,real,state,config_hash)
    assert sha(OLD/'results/full_train/submission.pt')==config['existing_submission_sha256']
    save(dest/'complete.json',dict(method=method,runs=45,pretrains=12,config_sha256=config_hash))
    print(method,'ALL 45 NESTED RUNS COMPLETE',flush=True)

if __name__=='__main__':main()
