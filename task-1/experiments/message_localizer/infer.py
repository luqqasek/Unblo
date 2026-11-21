"""Fixed full-refit inference; known message, no TTA, threshold 0.5."""
import argparse,json
from pathlib import Path
import torch
from wmloc.data import LocalisationDataset
from watermark_localization.submission import save_submission,validate_submission
from experiments.native_localizer.common import rgb_tensor,atomic_json,sha256
from .model import MessageLocalizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',required=True,choices=['pixelseal','trustmark']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=a.root/f'message-runs/{a.family}/matched-v1-full-refit/last.pt'
    out=Path('experiments/message_localizer/results/submission');out.mkdir(parents=True,exist_ok=True)
    dest=out/f'{a.family}-subset.pt'
    if dest.exists():raise FileExistsError(dest)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['config']['family']==a.family and state['config']['full_refit'] and state['step']==150
    model=MessageLocalizer(a.family).cuda().eval();model.load_state_dict(state['model'])
    ds=LocalisationDataset('data','test',methods=[a.family]);assert len(ds)==340
    names=[];masks=[]
    with torch.inference_mode():
        for start in range(0,len(ds),16):
            items=[ds[i] for i in range(start,min(start+16,len(ds)))]
            x=torch.stack([rgb_tensor(it['image']) for it in items]).cuda()
            bits=torch.tensor([it['message'].tolist() for it in items],device='cuda',dtype=torch.float32)
            prob=model(x,bits)['logits'].sigmoid()
            assert torch.isfinite(prob).all()
            masks.append((prob>=.5).to(torch.uint8).cpu());names.extend(it['image_name'] for it in items)
    masks=torch.cat(masks);save_submission(dest,names,masks,names)
    validate_submission(torch.load(dest,weights_only=True),names)
    atomic_json(out/f'{a.family}-inference.json',{'family':a.family,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'threshold':.5,'tta':False,'n':len(names),'predicted_area':float(masks.float().mean()),'subset_sha256':sha256(dest)})
    print('INFERENCE_COMPLETE',a.family,len(names),flush=True)
if __name__=='__main__':main()
