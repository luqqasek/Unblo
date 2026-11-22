"""Native RGB inference with frozen recipe; optional test execution is explicit.

One worker per GPU can receive --part0..3. No image resizing/augmentation outside
the existing extractor and detector input contracts. Ground truth is never read.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .model import TriForensicWM
from .noiseprint import Noiseprint
from .trustmark_refine import propose,refine
from experiments.small_zoom.proposals import detector,boxmask,infer as legacy_proposals
from experiments.native_localizer.common import rgb_tensor,atomic_json,sha256
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from watermark_localization.submission import save_submission,validate_submission

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--part',type=int,choices=range(4),required=True);p.add_argument('--test',action='store_true');a=p.parse_args()
    assert a.test,'Test inference requires an explicit frozen-recipe action'
    cfg=json.loads(a.recipe.read_text());checkpoint=Path(cfg['checkpoint']);assert sha256(checkpoint)==cfg['checkpoint_sha256'];assert cfg['threshold']==.5 and not cfg['tta']
    assert cfg.get('all_real_refit') and cfg.get('test_inference_allowed'),'Use the selected all170 refit for final test inference'
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);family=cfg['family'];assert family==state['family'];model=TriForensicWM(family).cuda();model.load_state_dict(state['model'],strict=True);del state;model.eval().requires_grad_(False);noise=Noiseprint().cuda()
    det=detector(a.root) if family=='trustmark' else None;wm=load_watermark(family,models_dir='checkpoints',device='cuda') if det is not None else None
    ds=LocalisationDataset('data','test',methods=[family]);assert not ds.train and len(ds)==340;names=[];probs=[];records=[];a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    for i in range(a.part,len(ds),4):
        it=ds[i];x=rgb_tensor(it['image'])[None].cuda();bits=torch.tensor(it['message'],device='cuda')[None].float()
        with torch.inference_mode():o=model(x,bits,noise(x));prob=o['logit_final'][0,0].sigmoid().cpu().numpy()
        if det is not None:
            old=legacy_proposals(it['image'],it['message'],it['image_name'],det,wm,scales=(512,1536,2560,4096))
            raw=propose(it['image'],o['l_loc'][0,0].sigmoid().cpu().numpy(),old,det);accepted,searched=refine(it['image'],it['message'],raw,old,wm);veto=o['delta_clean'][0,0].cpu().numpy()>2
            for r in accepted:prob[boxmask(r['box'])&~veto]=1
            records.append({'image_name':it['image_name'],'accepted':accepted,'search_count':len(searched)})
        assert np.isfinite(prob).all();names.append(it['image_name']);probs.append(prob)
        if len(names)%10==0:print('TEST_INFERENCE',family,a.part,len(names),'seconds',time.monotonic()-started,flush=True)
    masks=torch.from_numpy(np.array(probs)>=.5).to(torch.uint8);dest=a.out/f'part{a.part}.pt';assert not dest.exists();save_submission(dest,names,masks,names);validate_submission(torch.load(dest,weights_only=True),names)
    np.savez_compressed(a.out/f'part{a.part}.npz',names=names,probabilities=probs);atomic_json(a.out/f'part{a.part}.json',{'n':len(names),'seconds':time.monotonic()-started,'sha256':sha256(dest),'recipe':cfg,'test_GT_access':False,'proposals':records})
if __name__=='__main__':main()
