"""One optimizer/schedule, two-GPU DDP, bounded end-to-end region screen."""
import argparse,json,os,time,copy
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from experiments.native_localizer.common import atomic_json,append_jsonl,seed_all,rng_state
from .model import RegionPresence
from .geometry import candidates,tight,jitter,targets
from .data import make_plan,prep,Bank,pack
from .evaluate import search,recognition,recognition_summary,real_diagnostic,scan_real,finish


def init_distributed(world):
    if world>1:dist.init_process_group('nccl')
def barrier():
    if dist.is_initialized():dist.barrier()
def reduce_max(tensor):
    if dist.is_initialized():dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
def destroy_distributed():
    if dist.is_initialized():dist.destroy_process_group()


def gather(rows):
    if not dist.is_initialized():return rows
    all_rows=[None]*dist.get_world_size();dist.all_gather_object(all_rows,rows);return [r for part in all_rows for r in part]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--steps',type=int,default=800);p.add_argument('--max-train-seconds',type=int,default=1080);a=p.parse_args();rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));torch.cuda.set_device(rank);device=torch.device('cuda',rank);init_distributed(world);torch.set_num_threads(4);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'pixelseal-region-runs/native-v1';start=time.monotonic()
    if rank==0:make_plan(a.root,out)
    barrier();prep(a.root,out,rank,world);barrier();bank=Bank(out)
    model=RegionPresence().to(device);ddp=DDP(model,device_ids=[rank]) if world>1 else model;ema=copy.deepcopy(model).eval().requires_grad_(False);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01);schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-4,total_steps=a.steps,pct_start=.1)
    if rank==0:
        atomic_json(out/'config.json',{'steps_budget':a.steps,'max_training_seconds':a.max_train_seconds,'world':world,'batch_views_per_update':12,'paired_views':8,'real_views':4,'optimizer':'AdamW','max_lr':.0003,'weight_decay':.01,'schedule':'OneCycleLR','precision':'FP32 pixels, residual filtering, first projection, integral statistics and losses; BF16 deeper layers','trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'initialization':'fresh learned native filters; no real-label-pretrained carrier','loss':'class-balanced physical region fraction BCE + balanced box-IoU BCE; sampled high-score negative mining and periodic whole-search mining','base':'#3188 PS global EMA1561; frozen masks','acceptance':'one small rectangle OR on baseline-predicted area<=10%; threshold inner23 official mIoU; final reused57-image screen gain>=.005 with tiny-FG improvement and no>.01 clean-hole regression','source_counts':{k:{'views':len(v),'unique_sources':len({r['source'] for r in v}),'real_clean_views':sum(r['kind']=='real_clean' for r in v)} for k,v in bank.rows.items()}})
    rng=np.random.default_rng(4201+rank);cache={};trainstart=time.monotonic();losses=[];step=0
    for step in range(1,a.steps+1):
        ddp.train();items=bank.batch(rng);items=items+bank.batch(rng) if world==1 else items;boxes=[];hws=[]
        for x,y,b in items:
            hw=y.shape
            if hw not in cache:cache[hw]=candidates(*hw,step=2)
            full=cache[hw];idx=rng.choice(len(full),min(1024,len(full)),replace=False);arr=np.concatenate([full[idx],jitter(b,*hw,rng)])
            boxes.append(torch.as_tensor(arr,device=device));hws.append(hw)
        if step%100==0:
            with torch.inference_mode():
                top,_,_,_=search(ema,items[0][0],device,{})
            extra=jitter(top,*hws[0],rng,48)
            # Both twins receive identical searched negative geometry.
            for j in (0,1):boxes[j]=torch.cat([boxes[j],torch.as_tensor(extra,device=device)])
        rgb=pack(items,device);labels=[targets(y,b) for (_,y,_),b in zip(items,boxes)]
        with torch.autocast('cuda',dtype=torch.bfloat16):outputs=ddp(rgb,boxes,hws)
        loss=0
        for z,t in zip(outputs,labels):
            zz=z.float();pos=t[:,1]>.01;neg=~pos
            per=F.binary_cross_entropy_with_logits(zz,t,reduction='none').mean(1)
            parts=[]
            if pos.any():parts.append(per[pos].mean())
            if neg.any():
                hardness=zz.detach().sigmoid().prod(1);ids=torch.where(neg)[0];ids=ids[hardness[ids].topk(min(48,len(ids))).indices];parts.append(per[ids].mean())
            loss=loss+sum(parts)/len(parts)
        loss=loss/len(items);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();schedule.step()
        with torch.no_grad():
            for e,v in zip(ema.parameters(),model.parameters()):e.lerp_(v,.01)
            for e,v in zip(ema.buffers(),model.buffers()):e.copy_(v)
        losses.append(float(loss.detach()))
        if step%25==0:
            elapsed=time.monotonic()-trainstart
            if rank==0:
                row={'step':step,'loss':float(np.mean(losses[-25:])),'gradient_norm':float(norm),'elapsed_seconds':elapsed,'steps_per_second':step/elapsed,'lr':schedule.get_last_lr()[0],'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9};append_jsonl(out/'train.jsonl',row);print('TRAIN',json.dumps(row),flush=True)
        if step in (200,400):
            rr=gather(recognition(ema,bank,device,rank,world));dg=gather(real_diagnostic(ema,bank,device,rank,world))
            if rank==0:
                rs=recognition_summary(rr);audit={'step':step,'recognition':rs,'real_known_geometry_n':len(dg),'real_within_image_rank':float(np.mean([r['within_image_rank'] for r in dg])) if dg else None,'per_real':dg,'per_pair':rr};atomic_json(out/f'gate-{step}.json',audit);print('GATE',json.dumps({k:v for k,v in audit.items() if not k.startswith('per_')}),flush=True)
            # The declared early-stop gate uses overall held-out pair ROC-AUC.
            bad=recognition_summary(rr)['all']['auc']<.55 and step==400
            if bad:break
        elapsed=torch.tensor(time.monotonic()-trainstart,device=device);reduce_max(elapsed)
        if float(elapsed)>a.max_train_seconds:break
    barrier();model=ema.eval();rr=gather(recognition(model,bank,device,rank,world));dg=gather(real_diagnostic(model,bank,device,rank,world));rows=gather(scan_real(model,bank,device,rank,world,a.root/'pcmv-runs/base-candidate/all-real-maps.npz'))
    if rank==0:
        torch.save({'model':model.cpu().state_dict(),'train_model':(ddp.module if world>1 else ddp).state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'rng':rng_state(),'numpy_generator':rng.bit_generator.state,'step':step,'config':json.loads((out/'config.json').read_text())},out/'checkpoint.pt')
        result=finish(rows,rr,dg,bank,out,step,a.root/'pcmv-runs/base-candidate/all-real-maps.npz');atomic_json(out/'completed.json',{'step':step,'seconds_total':time.monotonic()-start,'seconds_train_including_gates':float(elapsed),'candidate_for_test':result['candidate_for_test'],'new_validation_miou':result['summary']['all']['miou'],'delta':result['delta'],'gpu_processes':world,'job_id':os.environ.get('SLURM_JOB_ID'),'stopped_reason':'recognition_gate_or_time_budget' if step<a.steps else 'fixed_endpoint'})
    barrier();destroy_distributed()


if __name__=='__main__':main()
