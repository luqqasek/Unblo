"""Short synthetic pilot with immutable raw-view cache, per-image losses and held-out real eval."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from experiments.native_localizer.common import atomic_json,append_jsonl,seed_all,rng_state,rgb_tensor,restore_rng,sha256
from experiments.native_localizer.generator import EncodedDataset,pair_views
from experiments.native_localizer.losses import segmentation,pair_loss
from experiments.native_localizer.evaluate import image_metrics,summarize,gallery
from wmloc.data import LocalisationDataset
from wmloc.utils import score
from .model import MessageLocalizer

@torch.inference_mode()
def evaluate(model,family,root,out,dev=False,limit=170):
    out.mkdir(parents=True,exist_ok=True);model.eval();rows=[];maps=[];targets=[];names=[]
    if dev:
        ds=EncodedDataset(root/f'encoded/{family}/pilot-dev/encoded.jsonl',family);rng=np.random.default_rng(2026)
        items=[]
        for i in range(min(limit,len(ds.rows))):
            clean,marked,r=ds.load(i);xs,ys,_=pair_views(clean,marked,r,rng,simple=False)
            items.append({'image':xs[0],'mask128':ys[0],'message':r['message'],'image_name':r['realization_id']})
    else:items=LocalisationDataset('data','train',methods=[family])
    for i,item in enumerate(items):
        x=rgb_tensor(item['image'])[None].cuda();b=torch.tensor(item['message'],device='cuda')[None].float()
        p=model(x,b)['logits'].sigmoid()[0].cpu().numpy();gt=item['mask128']
        r=image_metrics(p,gt,.5);r.update(image_name=item['image_name'],family=family);rows.append(r);maps.append(p);targets.append(gt);names.append(item['image_name'])
        if i<6:gallery(out/f'example_{i}.png',item['image'],gt,p)
    summary=summarize(rows);atomic_json(out/'metrics.json',summary);atomic_json(out/'per_image.json',rows)
    np.savez_compressed(out/'maps.npz',names=names,probabilities=np.stack(maps),targets=np.stack(targets))
    model.train();return summary['groups']['all']['miou']['mean']

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--run',default='matched-v1');p.add_argument('--steps',type=int,default=400);p.add_argument('--batch',type=int,default=32);p.add_argument('--gpus',type=int,default=2);p.add_argument('--freeze-steps',type=int,default=50);p.add_argument('--overfit',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
 out=a.root/f'message-runs/{a.family}/{a.run}';out.mkdir(parents=True,exist_ok=True)
 if not a.resume and ((out/'last.pt').exists() or (out/'train.jsonl').exists()):raise FileExistsError(out)
 torch.set_num_threads(4);seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=vars(a).copy();cfg.pop('resume');cfg['root']=str(a.root);cfg.update(seed=42,precision='FP32',threshold=.5,extractor_lr=2e-5,new_lr=3e-4,loss='per-image .5BCE+.5both-classDice + ramped pair ranking; no dense bit auxiliary',generator='existing exact encoded sources; procedural irregular masks enabled from update0',real_label_training=False)
 atomic_json(out/'config.json',cfg)
 ds=EncodedDataset(a.root/f'encoded/{a.family}/pilot-train/encoded.jsonl',a.family);ds.preload(workers=4)
 messages={r['realization_id']:r['message'] for r in ds.rows}
 model=MessageLocalizer(a.family).cuda();model.train();model.freeze_encoder(True)
 atomic_json(out/'model.json',{'total_parameters':sum(p.numel() for p in model.parameters()),'native_decoder_size':model.encoder.native,'channels':model.encoder.channels,'BN_running_stats':'frozen during all training','inference_inputs':['RGB','message'],'pretrained_extractor':'unchanged original competition weights at initialization'})
 enc=list(model.encoder.net.parameters());ids={id(p) for p in enc};new=[p for p in model.parameters() if id(p) not in ids and p.requires_grad]
 optimizer=torch.optim.AdamW([{'params':enc,'lr':2e-5},{'params':new,'lr':3e-4}],weight_decay=.01)
 forward=torch.nn.DataParallel(model,device_ids=list(range(a.gpus))) if a.gpus>1 else model
 rng=np.random.default_rng(42);start=time.monotonic();fixed=None
 if a.overfit:fixed=ds.batch(rng,a.batch,curriculum_update=5000,paired=True,fixed_indices=list(range(a.batch)))
 start_step=0
 if a.resume:
    state=torch.load(a.resume,map_location='cuda',weights_only=False)
    assert state['config']==cfg
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng']);rng.bit_generator.state=state['data_rng'];start_step=state['step']
    model.freeze_encoder(start_step<a.freeze_steps)
 for step in range(start_step,a.steps):
    if step==a.freeze_steps:model.freeze_encoder(False)
    factor=min(1.,(step+1)/20)*(.2+.8*.5*(1+math.cos(math.pi*step/a.steps)))
    for group,lr in zip(optimizer.param_groups,(2e-5,3e-4)):group['lr']=lr*factor
    began=time.monotonic();images,y,pairs,_,meta=fixed or ds.batch(rng,a.batch,curriculum_update=5000,paired=True)
    b=torch.tensor([messages[m['realization_id']] for m in meta],device='cuda',dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True);z=forward(images.cuda(),b)['logits'];target=y.cuda()
    seg=segmentation(z,target);pair=pair_loss(z,target,pairs);loss=seg+.1*min(1,(step+1)/100)*pair
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
    optimizer.step();torch.cuda.synchronize()
    if (step+1)%10==0 or step==0:
        row={'step':step+1,'seg':float(seg.detach()),'pair':float(pair.detach()),'grad_norm':float(norm),'seconds_step':time.monotonic()-began,'elapsed':time.monotonic()-start,'frozen':model.frozen,'peak_bytes':torch.cuda.max_memory_allocated(),'train_miou':score(z.detach().sigmoid().cpu().numpy()>=.5,y.numpy())}
        print(json.dumps(row),flush=True);append_jsonl(out/'train.jsonl',row)
    if (step+1)%100==0 or step+1==a.steps:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'config':cfg,'rng':rng_state(),'data_rng':rng.bit_generator.state},out/'last.pt')
        if not a.overfit:
            val=evaluate(model,a.family,a.root,out/f'dev/{step+1}',dev=True,limit=64)
            print(json.dumps({'step':step+1,'synthetic_dev_miou':val}),flush=True);append_jsonl(out/'dev.jsonl',{'step':step+1,'miou':val})
 if not a.overfit:
    val=evaluate(model,a.family,a.root,out/'real',dev=False)
    print(json.dumps({'REAL_VALIDATION':True,'family':a.family,'n':170,'miou':val}),flush=True)
 atomic_json(out/'completed.json',{'seconds':time.monotonic()-start,'steps':a.steps,'views':a.steps*a.batch})
if __name__=='__main__':main()
