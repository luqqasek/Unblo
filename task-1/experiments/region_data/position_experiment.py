"""Bounded matched 100-step screen of counterfactual position-pair data."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from experiments.native_localizer.common import seed_all,atomic_json,append_jsonl
from experiments.native_localizer.losses import segmentation,pair_loss
from experiments.message_localizer.model import MessageLocalizer
from experiments.message_localizer.train import evaluate
from .data import RegionData
from .pair_data import PositionPairs

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=RegionData(a.root,a.family,'train');positions=PositionPairs(data)
 initial=torch.load(a.root/f'region-runs/{a.family}/objects/last.pt',map_location='cpu',weights_only=False)
 for variant in ['objects_plus100','position_pairs100']:
  out=a.root/'region-runs'/a.family/variant;out.mkdir(parents=True,exist_ok=True)
  if (out/'completed.json').exists():continue
  seed_all(73);rng=np.random.default_rng(73);model=MessageLocalizer(a.family).cuda();model.load_state_dict(initial['model']);model.freeze_encoder(False);model.train();forward=torch.nn.DataParallel(model,device_ids=[0,1])
  enc=list(model.encoder.net.parameters());ids={id(p) for p in enc};new=[p for p in model.parameters() if id(p) not in ids and p.requires_grad]
  opt=torch.optim.AdamW([{'params':enc,'lr':5e-6},{'params':new,'lr':5e-5}],weight_decay=.01)
  atomic_json(out/'config.json',{'family':a.family,'variant':variant,'seed':73,'initial':'objects600 checkpoint, no real training','steps':100,'batch':32,'encoder_lr':5e-6,'new_lr':5e-5,'loss':'same segmentation +0.1 changed-cell ranking','threshold':.5,'data':'half views equal-area same-content position pairs; pair minor bins70% .25-1%,30%1-5%, equal FG/clean polarity' if variant=='position_pairs100' else 'unchanged objects data'})
  counts=Counter();start=time.monotonic()
  for step in range(100):
   x,y,pairs,meta=positions.batch(rng) if variant=='position_pairs100' else data.batch(rng,32,True)
   bits=torch.tensor([data.messages[m['realization_id']] for m in meta],device='cuda').float()
   opt.zero_grad(set_to_none=True);z=forward(x.cuda(),bits)['logits'];loss=segmentation(z,y.cuda())+.1*pair_loss(z,y.cuda(),pairs);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
   opt.step()
   for m in meta:
    f=m['fg_fraction'];counts['views']+=1;counts['fg_le1']+=int(0<f<=.01);counts['fg_1to5']+=int(.01<f<=.05);counts['empty']+=int(f==0);counts['full']+=int(f==1)
   if (step+1)%25==0:print(json.dumps({'family':a.family,'variant':variant,'step':step+1,'seconds':time.monotonic()-start}),flush=True)
   if step<4:append_jsonl(out/'sampled_provenance.jsonl',{'step':step+1,'views':meta})
  torch.save({'model':model.state_dict(),'config':{'family':a.family,'variant':variant},'step':100},out/'model.pt')
  real=evaluate(model,a.family,a.root,out/'real',dev=False);atomic_json(out/'completed.json',{'real_miou':real,'seconds':time.monotonic()-start,'views':dict(counts)})
  print(json.dumps({'POSITION_SCREEN_COMPLETE':True,'family':a.family,'variant':variant,'real_miou':real}),flush=True)
  del model,forward,opt;torch.cuda.empty_cache()
if __name__=='__main__':main()
