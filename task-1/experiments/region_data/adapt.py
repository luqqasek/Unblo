"""Matched three-fold adaptation, only generator variant differs between arms."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from wmloc.data import LocalisationDataset
from experiments.native_localizer.common import atomic_json,seed_all,rgb_tensor
from experiments.native_localizer.losses import segmentation,pair_loss
from experiments.native_localizer.evaluate import summarize
from experiments.message_localizer.model import MessageLocalizer
from experiments.message_localizer.adapt import eval_indices
from .data import RegionData

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--folds',type=int,nargs='+',default=[1,2,3]);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=RegionData(a.root,a.family,'train');ds=LocalisationDataset('data','train',methods=[a.family]);lookup={r['image_name']:i for i,r in enumerate(ds.meta)}
 splits=json.loads(Path('experiments/native_localizer/reconstructed_splits.json').read_text())['families'][a.family]
 started=time.monotonic()
 # Interleave arms by fold so the first matched real result arrives quickly.
 for fold in a.folds:
  valid=[lookup[n] for n in splits[fold-1]['validation']];train=[i for i in range(len(ds)) if i not in set(valid)]
  assert not set(train)&set(valid)
  for variant in ['control','objects']:
   out=a.root/'region-runs'/a.family/f'{variant}-adapt';dest=out/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
   if (dest/'per_image.json').exists():continue
   config={'variant':variant,'family':a.family,'fold':fold,'steps':150,'batch':32,'synthetic':24,'real':8,'encoder_lr':2e-6,'new_lr':2e-5,'threshold':.5,'seed':42,'selection':'fixed150 updates and threshold, no outer selection'}
   atomic_json(out/'config.json',config);atomic_json(dest/'split.json',{'train':[ds.meta[i]['image_name'] for i in train],'validation':splits[fold-1]['validation']})
   state=torch.load(a.root/f'region-runs/{a.family}/{variant}/last.pt',map_location='cpu',weights_only=False)
   seed_all(42);rng=np.random.default_rng(42);model=MessageLocalizer(a.family).cuda();model.load_state_dict(state['model']);del state;model.freeze_encoder(False);model.train();forward=torch.nn.DataParallel(model,device_ids=[0,1])
   enc=list(model.encoder.net.parameters());ids={id(p) for p in enc};new=[p for p in model.parameters() if id(p) not in ids and p.requires_grad]
   opt=torch.optim.AdamW([{'params':enc,'lr':2e-6},{'params':new,'lr':2e-5}],weight_decay=.01)
   # Separate RNGs prevent different generator draw counts changing real training exposure.
   real_rng=np.random.default_rng(12342)
   for step in range(150):
    x,y,pairs,meta=data.batch(rng,24,variant=='objects');items=[ds[int(i)] for i in real_rng.choice(train,8,replace=True)]
    x=torch.cat([x,torch.stack([rgb_tensor(it['image']) for it in items])]).cuda();y=torch.cat([y,torch.stack([torch.tensor(it['mask128']).float() for it in items])]).cuda()
    b=torch.tensor([data.messages[m['realization_id']] for m in meta]+[it['message'].tolist() for it in items],device='cuda').float()
    opt.zero_grad(set_to_none=True);z=forward(x,b)['logits'];loss=segmentation(z,y)+.05*pair_loss(z,y,pairs);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
    opt.step()
    if (step+1)%50==0:print(json.dumps({'family':a.family,'variant':variant,'fold':fold,'step':step+1,'seconds':time.monotonic()-started}),flush=True)
   torch.save({'model':model.state_dict(),'config':config},dest/'model.pt')
   rows=eval_indices(model,ds,valid,dest)
   print(json.dumps({'OUTER_VALIDATION':True,'family':a.family,'variant':variant,'fold':fold,'miou':float(np.mean([r['miou'] for r in rows]))}),flush=True)
   del model,forward,opt;torch.cuda.empty_cache()
 for variant in ['control','objects']:
  out=a.root/'region-runs'/a.family/f'{variant}-adapt';paths=[out/f'fold{f}/per_image.json' for f in [1,2,3]]
  if all(p.exists() for p in paths):
   rows=sum([json.loads(p.read_text()) for p in paths],[]);assert len(rows)==len({r['image_name'] for r in rows})==170
   atomic_json(out/'per_image.json',rows);atomic_json(out/'metrics.json',summarize(rows))
   print(json.dumps({'OOF_COMPLETE':True,'family':a.family,'variant':variant,'miou':float(np.mean([r['miou'] for r in rows]))}),flush=True)
if __name__=='__main__':main()
