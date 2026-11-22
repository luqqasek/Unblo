"""One optimizer / OneCycle / one backward for co-batched full and paired views."""
import argparse,copy,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from .model import TriForensicWM
from .data import Engine,device_batch
from .losses import objective
from .evaluate import real_evaluate,paired_evaluate,synthetic_evaluate,old_rectangles
from .audit import fidelity
from experiments.native_localizer.common import atomic_json,append_jsonl,seed_all,rng_state,restore_rng,sha256

@torch.no_grad()
def ema_update(ema,model):
    for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p.detach(),.005)
    for e,p in zip(ema.buffers(),model.buffers()):e.copy_(p)

def save(path,model,ema,opt,sched,step):
    tmp=path.with_suffix('.tmp');torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'step':step,'rng':rng_state(),'batch_seed':'421007+step'},tmp);os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True,choices=['pixelseal','trustmark']);p.add_argument('--steps',type=int,default=3500);p.add_argument('--run',default='unified-v1');p.add_argument('--resume',action='store_true');p.add_argument('--carrier-no-checkpointing',action='store_true');p.add_argument('--wall-seconds',type=int,default=1650);a=p.parse_args()
    torch.set_num_threads(6);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==4,'One node with4 GPUs required'
    out=a.root/f'triforensic-runs/{a.family}/{a.run}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'completed.json').exists(),'Preserve completed run'
    started=time.monotonic();engine=Engine(a.root,a.family);variant='control' if a.family=='pixelseal' else 'objects';base=a.root/f'region-runs/{a.family}/{variant}-adapt/fold1/model.pt'
    model=TriForensicWM(a.family,base).cuda();model.carrier.checkpointing=not a.carrier_no_checkpointing;ema=copy.deepcopy(model).eval().requires_grad_(False);parallel=torch.nn.DataParallel(model);evalparallel=torch.nn.DataParallel(ema)
    opt=torch.optim.AdamW(model.parameter_groups());sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[g['lr'] for g in opt.param_groups],total_steps=a.steps,pct_start=.05)
    step=0;prior_seconds=0.;prior_train_seconds=0.
    if a.resume:
        state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);step=state['step'];restore_rng(state['rng']);del state
        if (out/'paused.json').exists():
            previous=json.loads((out/'paused.json').read_text());prior_seconds=previous['seconds'];prior_train_seconds=previous['train_seconds']
    config={'family':a.family,'seed':42,'fold':1,'steps':a.steps,'full_batch':16,'synthetic_full':12,'real_full':4,'paired_pairs':64,'views_per_update':144,'one_optimizer':'AdamW','one_schedule':'OneCycleLR pct_start0.05','base_max_lr':2e-5,'new_max_lr':4e-4,'ema':.995,'threshold':.5,'base_checkpoint':str(base),'base_sha256':sha256(base),'all_base_trainable_from_update0':True,'carrier_rf':29,'carrier_params':sum(p.numel() for p in model.carrier.parameters()),'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),'normalization':'per-position8 channel groups','refiner_body_rf128':7,'refiner_effective_context':'ring9 plus frozen NoiseprintRF35 and global spatial median; NOT local','precision':'FP32 observed pixels,filters,JND,first projections,loss; BF16 deeper convolutions','pool_n':len(engine.train.rows),'source_n':len({r['source_id'] for r in engine.train.rows}),'no_augmentation':'no post-quantization blur,JPEG,color,rotation or geometric image augmentation','gpu':[torch.cuda.get_device_name(i) for i in range(4)],'max_wall_seconds':a.wall_seconds,'gates':{'step':1000,'ps_local_real_tiny_auc':.55,'thin_interior_synthetic_auc':.65}}
    if not (out/'config.json').exists():atomic_json(out/'config.json',config)
    atomic_json(out/f'runtime-{step}.json',{'resume_step':step,'carrier_checkpointing':model.carrier.checkpointing,'wall_seconds':a.wall_seconds,'same_optimizer_and_schedule':True})
    atomic_json(out/'split.json',{'train':[engine.real[i]['image_name'] for i in engine.training],'validation':[engine.real[i]['image_name'] for i in engine.valid]})
    rectangles=old_rectangles(a.root) if a.family=='trustmark' else {}
    if not a.resume:
        fidelity(model,engine,out/'fidelity.json')
        # FP32 original model provides the exact#2409 matched-fold baseline.
        baseline=real_evaluate(torch.nn.DataParallel(model.base),engine,out,'baseline',True,rectangles)
        initial=real_evaluate(evalparallel,engine,out,0,rectangles=rectangles)
        print('INITIALIZATION_DELTA',initial['summary']['all']['miou']-baseline['summary']['all']['miou'],flush=True)
    executor=ThreadPoolExecutor(max_workers=1);future=executor.submit(engine.batch,step);train_started=time.monotonic();initial_step=step;train_seconds=0.;stop='steps_complete';gate=json.loads((out/'gate1000.json').read_text()) if (out/'gate1000.json').exists() else None
    for update in range(step,a.steps):
        if time.monotonic()-started>a.wall_seconds:stop='wall_budget_resumable';break
        t=time.monotonic();batch=device_batch(future.result())
        if update+1<a.steps:future=executor.submit(engine.batch,update+1)
        model.train();opt.zero_grad(set_to_none=True)
        before_rng=rng_state()
        try:
            pred=parallel(batch['x'],batch['bits'],batch['npmap'],batch['pairs']);loss,terms=objective(pred,batch['y'],batch['pair_y']);loss.backward()
        except torch.cuda.OutOfMemoryError:
            if model.carrier.checkpointing:raise
            # No optimizer update occurred. Retry identical views/RNG with
            # activation recomputation; the training objective is unchanged.
            pred=loss=terms=None;opt.zero_grad(set_to_none=True)
            import gc
            gc.collect();torch.cuda.empty_cache();model.carrier.checkpointing=True;restore_rng(before_rng)
            append_jsonl(out/'runtime-events.jsonl',{'step':update,'event':'carrier_checkpointing_enabled_after_OOM','optimizer_unchanged':True})
            pred=parallel(batch['x'],batch['bits'],batch['npmap'],batch['pairs']);loss,terms=objective(pred,batch['y'],batch['pair_y']);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm),'Non-finite gradient';opt.step();sched.step();ema_update(ema,model);step=update+1
        torch.cuda.synchronize();elapsed=time.monotonic()-t;train_seconds+=elapsed
        if step%25==0 or step==1:
            row={'step':step,'loss':float(loss),'terms':{k:float(v) for k,v in terms.items()},'grad_norm':float(norm),'seconds':time.monotonic()-started,'train_seconds':train_seconds,'step_seconds':elapsed,'eta_train_seconds':(a.steps-step)*train_seconds/(step-initial_step),'lr':sched.get_last_lr(),'memory_peak_GB':[torch.cuda.max_memory_allocated(i)/1e9 for i in range(4)],'views':step*144}
            append_jsonl(out/'train.jsonl',row);print('TRAIN',json.dumps(row),flush=True)
        del pred,loss,terms,batch
        if step in (100,200,1000,2000,3000,a.steps):
            save(out/'resume.pt',model,ema,opt,sched,step)
            if step==200:atomic_json(out/'profile.json',{'steps':200,'train_seconds':train_seconds,'seconds_per_update':train_seconds/200,'remaining_training_seconds_estimate':(a.steps-200)*train_seconds/200,'peak_GB':[torch.cuda.max_memory_allocated(i)/1e9 for i in range(4)]})
            if step in (100,1000,2000,3000,a.steps):
                real=real_evaluate(evalparallel,engine,out,step,rectangles=rectangles)
                if step==1000:
                    paired=paired_evaluate(ema,engine,out,step);syn=synthetic_evaluate(evalparallel,engine,out,step)
                    auc=real['local_summary']['fg_le1']['roc_auc'];interior=paired['tiny_interior']['auc']
                    gate={'real_tiny_auc':auc,'synthetic_interior_auc':interior,'passed':(auc is not None and auc>.55 and interior is not None and interior>=.65) if a.family=='pixelseal' else (interior is not None and interior>=.65)}
                    atomic_json(out/'gate1000.json',gate)
                    if not gate['passed']:stop='gate_failed_debug_required';break
    executor.shutdown(wait=True);save(out/'resume.pt',model,ema,opt,sched,step)
    real=real_evaluate(evalparallel,engine,out,step,rectangles=rectangles)
    if step!=1000:paired=paired_evaluate(ema,engine,out,step);syn=synthetic_evaluate(evalparallel,engine,out,step)
    record={'step':step,'stop':stop,'seconds':prior_seconds+time.monotonic()-started,'session_seconds':time.monotonic()-started,'train_seconds':prior_train_seconds+train_seconds,'final':real['summary'],'local':real['local_summary'],'gate':gate,'views':step*144,'optimizer_updates':step}
    atomic_json(out/('paused.json' if stop=='wall_budget_resumable' else 'completed.json'),record);print('RUN_END',json.dumps(record),flush=True)
if __name__=='__main__':main()
