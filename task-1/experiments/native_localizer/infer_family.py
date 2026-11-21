"""Export one family's native RGB predictions; not a complete leaderboard file."""
import argparse
from pathlib import Path
import torch
from wmloc.data import LocalisationDataset
from .common import atomic_json,rgb_tensor,sha256
from .model import NativeLocalizer

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--family',required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 assert state['config']['family']==a.family and state['update']==200
 model=NativeLocalizer(checkpointing=False).cuda().eval(); model.load_state_dict(state['model'])
 ds=LocalisationDataset(a.data,'test',methods=[a.family]); assert len(ds)==340
 masks=[]
 with torch.inference_mode():
  for i in range(0,len(ds),16):
   x=torch.stack([rgb_tensor(ds[j]['image']) for j in range(i,min(i+16,len(ds)))]).cuda()
   masks.append((model(x)['logits'].sigmoid()>=.5).to(torch.uint8).cpu())
 names=[r['image_name'] for r in ds.meta]; masks=torch.cat(masks)
 a.out.parent.mkdir(parents=True,exist_ok=True); torch.save({'image_names':names,'masks':masks},a.out)
 assert masks.shape==(340,128,128) and len(set(names))==340
 atomic_json(a.out.with_suffix('.json'),{'family':a.family,'threshold':.5,'weights':'raw','update':200,'checkpoint_sha256':sha256(a.checkpoint),'artifact_sha256':sha256(a.out),'n':340,'shape':list(masks.shape),'not_complete_submission':True})
 print('FAMILY_PREDICTIONS_READY',a.out,flush=True)
if __name__=='__main__':main()
