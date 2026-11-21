"""Fixed-schedule three-fold real adaptation with 75% synthetic replay; no outer selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from wmloc.data import LocalisationDataset
from experiments.native_localizer.common import atomic_json,append_jsonl,seed_all,rgb_tensor
from experiments.native_localizer.generator import EncodedDataset
from experiments.native_localizer.losses import segmentation,pair_loss
from experiments.native_localizer.evaluate import image_metrics,summarize,gallery
from .model import MessageLocalizer

@torch.inference_mode()
def eval_indices(model,ds,indices,out):
 model.eval();rows=[];maps=[];targets=[];names=[];out.mkdir(parents=True,exist_ok=True)
 for i in indices:
  item=ds[int(i)];b=torch.tensor(item['message'],device='cuda')[None].float()
  p=model(rgb_tensor(item['image'])[None].cuda(),b)['logits'].sigmoid()[0].cpu().numpy()
  r=image_metrics(p,item['mask128'],.5);r['image_name']=item['image_name'];rows.append(r);maps.append(p);targets.append(item['mask128']);names.append(item['image_name'])
  if len(rows)<=4:gallery(out/f'example_{len(rows)}.png',item['image'],item['mask128'],p)
 atomic_json(out/'metrics.json',summarize(rows));atomic_json(out/'per_image.json',rows)
 np.savez_compressed(out/'maps.npz',names=names,probabilities=np.stack(maps),targets=np.stack(targets))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--steps',type=int,default=150);p.add_argument('--gpus',type=int,default=2);p.add_argument('--full-refit',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=a.root/f"message-runs/{a.family}/{'matched-v1-full-refit' if a.full_refit else 'matched-v1-adapt'}";out.mkdir(parents=True,exist_ok=True)
 config={'family':a.family,'steps_per_fold':a.steps,'seed':42,'folds':3,'batch':32,'synthetic_views':24,'real_views':8,'threshold':.5,'encoder_lr':2e-6,'decoder_lr':2e-5,'selection':'fixed before outer evaluation; no checkpoint/threshold search; three original folds; not nested tuning'}
 config.update(full_refit=a.full_refit)
 if a.full_refit:config.update(folds=0,selection='Fixed 150-step recipe selected from completed OOF; all 170 real training images; no test-label use')
 atomic_json(out/'config.json',config)
 ds=LocalisationDataset('data','train',methods=[a.family]);lookup={r['image_name']:i for i,r in enumerate(ds.meta)}
 splits=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families'][a.family]
 if a.full_refit:splits=[{'validation':[]}]
 synth=EncodedDataset(a.root/f'encoded/{a.family}/pilot-train/encoded.jsonl',a.family);synth.preload(workers=4);msgs={r['realization_id']:r['message'] for r in synth.rows}
 state=torch.load(a.root/f'message-runs/{a.family}/matched-v1/last.pt',map_location='cpu',weights_only=False)
 assert state['step']==400
 allrows=[];started=time.monotonic()
 for fold,split in enumerate(splits,1):
  dest=out if a.full_refit else out/f'fold{fold}'
  if a.full_refit and (dest/'last.pt').exists():
   print('Completed full refit already exists; preserving checkpoint',flush=True);return
  if (dest/'per_image.json').exists():allrows.extend(json.loads((dest/'per_image.json').read_text()));continue
  seed_all(42);rng=np.random.default_rng(42);model=MessageLocalizer(a.family).cuda();model.load_state_dict(state['model']);model.freeze_encoder(False);model.train()
  forward=torch.nn.DataParallel(model,device_ids=list(range(a.gpus))) if a.gpus>1 else model
  enc=list(model.encoder.net.parameters());ids={id(p) for p in enc};new=[p for p in model.parameters() if id(p) not in ids and p.requires_grad]
  optimizer=torch.optim.AdamW([{'params':enc,'lr':2e-6},{'params':new,'lr':2e-5}],weight_decay=.01)
  valid=[lookup[n] for n in split['validation']];train=[i for i in range(len(ds)) if i not in set(valid)]
  assert not set(train)&set(valid) and len(train)+len(valid)==170
  dest.mkdir(exist_ok=True);atomic_json(dest/'split.json',{'train':[ds.meta[i]['image_name'] for i in train],'validation':split['validation']})
  for step in range(a.steps):
   x,y,pairs,_,meta=synth.batch(rng,24,curriculum_update=5000,paired=True)
   items=[ds[int(i)] for i in rng.choice(train,8,replace=True)]
   x=torch.cat([x,torch.stack([rgb_tensor(it['image']) for it in items])]).cuda()
   y=torch.cat([y,torch.stack([torch.tensor(it['mask128']).float() for it in items])]).cuda()
   b=torch.tensor([msgs[m['realization_id']] for m in meta]+[it['message'].tolist() for it in items],device='cuda',dtype=torch.float32)
   optimizer.zero_grad(set_to_none=True);z=forward(x,b)['logits'];loss=segmentation(z,y)+.05*pair_loss(z,y,pairs)
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
   optimizer.step()
   if (step+1)%25==0:print(json.dumps({'family':a.family,'fold':fold,'step':step+1,'loss':float(loss.detach()),'elapsed':time.monotonic()-started}),flush=True)
  torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'fold':fold,'step':a.steps,'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'data':rng.bit_generator.state}},dest/'last.pt')
  if a.full_refit:
   print(json.dumps({'FULL_REFIT_COMPLETE':True,'family':a.family,'seconds':time.monotonic()-started}),flush=True);return
  rows=eval_indices(model,ds,valid,dest);allrows.extend(rows)
  print(json.dumps({'OUTER_VALIDATION':True,'family':a.family,'fold':fold,'miou':float(np.mean([r['miou'] for r in rows]))}),flush=True)
  del model,forward,optimizer;torch.cuda.empty_cache()
 assert len(allrows)==len({r['image_name'] for r in allrows})==170
 atomic_json(out/'metrics.json',summarize(allrows));atomic_json(out/'per_image.json',allrows)
 print(json.dumps({'OOF_COMPLETE':True,'family':a.family,'miou':float(np.mean([r['miou'] for r in allrows])),'seconds':time.monotonic()-started}),flush=True)
if __name__=='__main__':main()
