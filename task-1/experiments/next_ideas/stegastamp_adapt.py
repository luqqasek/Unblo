"""Predeclared short real/replay adaptation of the unchanged native RGB model."""
import argparse,json,os,signal,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from experiments.native_localizer.model import NativeLocalizer
from experiments.native_localizer.generator import EncodedDataset
from experiments.native_localizer.losses import segmentation,local_balanced
from experiments.native_localizer.common import atomic_json,append_jsonl,rgb_tensor,seed_all,rng_state,restore_rng,sha256
from experiments.native_localizer.evaluate import image_metrics
from experiments.triforensic_wm.evaluate import summarize
from wmloc.data import LocalisationDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fold',type=int,choices=(1,2,3),default=1);p.add_argument('--all-real',action='store_true');p.add_argument('--steps',type=int,default=150);p.add_argument('--resume',action='store_true');p.add_argument('--wall-seconds',type=int,default=1500);a=p.parse_args()
    torch.set_num_threads(6);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'next-ideas-runs/stegastamp'/('all170' if a.all_real else f'fold{a.fold}');out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists()
    started=time.monotonic();stop=[False]
    def halt(*_):stop[0]=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    ds=LocalisationDataset('data','train',methods=['stegastamp']);items=[ds[i] for i in range(len(ds))]
    fold=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families']['stegastamp'][a.fold-1]
    valid=[] if a.all_real else [i for i,r in enumerate(items) if r['image_name'] in fold['validation']]
    training=[i for i in range(170) if i not in valid];assert a.all_real or len(valid) in (56,57)
    source=a.root/'runs/stegastamp/pilot-30min/last.pt';state=torch.load(source,map_location='cpu',weights_only=False)
    model=NativeLocalizer(checkpointing=True,amp=False).cuda();model.load_state_dict(state['model'],strict=True);del state
    optimizer=torch.optim.AdamW(model.parameter_groups(3e-6,1e-5,.01))
    manifest=a.root/'encoded/stegastamp/pilot-train/encoded.jsonl';replay=EncodedDataset(manifest,'stegastamp');assert all(r['partition']=='train' for r in replay.rows);replay.preload(workers=4)
    cfg={'family':'stegastamp','fold':None if a.all_real else a.fold,'all_real':a.all_real,'seed':42,'updates':a.steps,'effective_batch':16,'synthetic_views':12,'real_views':4,'microbatch':16,'backbone_lr':3e-6,'new_lr':1e-5,'weight_decay':.01,'precision':'FP32;TF32 disabled','evaluation_weights':'raw','threshold':.5,'geometry_augmentation':False,'loss':'existing per-image0.5BCE+0.5dualDice; local auxiliary0.04','base_checkpoint':str(source),'base_sha256':sha256(source),'train_manifest_sha256':sha256(manifest),'synthetic_sources':len({r['source_id'] for r in replay.rows}),'real_train_n':len(training),'validation_n':len(valid),'selection':'150-update endpoint declared before results;50 is progress only;3fold OOF improvement>=.01 to justify final refit','new_embedder_calls':0}
    atomic_json(out/'config.json',cfg);atomic_json(out/'split.json',{'train':[items[i]['image_name'] for i in training],'validation':[items[i]['image_name'] for i in valid]})
    initial=0
    if a.resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert state['config']==cfg;model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);initial=state['step'];restore_rng(state['rng']);del state
    @torch.inference_mode()
    def evaluate(step):
        model.eval();rows=[];names=[];maps=[];targets=[]
        for begin in range(0,len(valid),4):
            ids=valid[begin:begin+4];x=torch.stack([rgb_tensor(items[i]['image']) for i in ids]).cuda();probs=model(x)['logits'].sigmoid().cpu().numpy()
            for i,prob in zip(ids,probs):
                it=items[i];r=image_metrics(prob,it['mask128']);r.update(image_name=it['image_name'],family='stegastamp',fold=a.fold);rows.append(r);names.append(it['image_name']);maps.append(prob);targets.append(it['mask128'])
        if not rows:return None
        record={'step':step,'summary':summarize(rows),'per_image':rows,'protocol':'existing outer folds; fixed150-update endpoint, threshold0.5; repeatedly inspected development folds'}
        if step==0:
            z=np.load(a.root/'runs/stegastamp/pilot-30min/real_validation_all/maps.npz');old={str(n):v for n,v in zip(z['names'],z['probabilities'])};error=max(float(np.max(np.abs(prob-old[n]))) for n,prob in zip(names,maps));record['archived_map_max_abs_difference']=error
            assert error<1e-4,f'Baseline reproduction gap: {error}; stop before training'
        atomic_json(out/f'real-{step}.json',record);np.savez_compressed(out/f'maps-{step}.npz',names=names,probabilities=maps,targets=targets);model.train();print('SS_VALIDATION',a.fold,step,json.dumps(record['summary']),flush=True);return record
    def save(step):
        tmp=out/'last.tmp';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':cfg,'rng':rng_state()},tmp);os.replace(tmp,out/'last.pt')
    if not a.resume:evaluate(0)
    exposure=[]
    def batch(step):
        rng=np.random.default_rng(72042+step);x,y,_,_,meta=replay.batch(rng,12,0,False)
        ids=rng.choice(training,4,replace=True);xx=torch.stack([rgb_tensor(items[i]['image']) for i in ids]);yy=torch.tensor(np.stack([items[i]['mask128'] for i in ids])).float()
        return torch.cat([x,xx]),torch.cat([y,yy]),meta,[items[i]['image_name'] for i in ids]
    pool=ThreadPoolExecutor(max_workers=1);future=pool.submit(batch,initial);step=initial;reason='steps_complete';train_seconds=0
    for update in range(initial,a.steps):
        if stop[0] or time.monotonic()-started>a.wall_seconds:reason='paused';break
        x,y,meta,real_names=future.result()
        if update+1<a.steps:future=pool.submit(batch,update+1)
        model.train();optimizer.zero_grad(set_to_none=True);t=time.monotonic();value=0.
        for j in range(0,16,16):
            pred=model(x[j:j+16].cuda());target=y[j:j+16].cuda();loss=(segmentation(pred['logits'],target)+.04*local_balanced(pred['local_logits'],target));loss.backward();value+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step();torch.cuda.synchronize();elapsed=time.monotonic()-t;train_seconds+=elapsed;step=update+1
        exposure.append({'step':step,'synthetic':meta,'real':real_names})
        if step==1 or step%10==0:
            row={'step':step,'loss':value,'gradient_norm':float(norm),'seconds_per_update':train_seconds/(step-initial),'eta_training_seconds':(a.steps-step)*train_seconds/(step-initial),'peak_GB':torch.cuda.max_memory_allocated()/1e9};append_jsonl(out/'train.jsonl',row);print('SS_TRAIN',a.fold,json.dumps(row),flush=True)
        if step in (50,a.steps):save(step);evaluate(step)
    pool.shutdown(wait=True);save(step);atomic_json(out/'exposure.json',{'views':step*16,'newly_logged_updates':exposure,'unique_synthetic_sources':len({m['source_id'] for r in exposure for m in r['synthetic']})})
    atomic_json(out/('completed.json' if reason=='steps_complete' else 'paused.json'),{'step':step,'stop':reason,'seconds':time.monotonic()-started,'training_seconds':train_seconds,'config':cfg});print('SS_END',a.fold,step,reason,flush=True)


if __name__=='__main__':main()
