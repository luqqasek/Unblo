"""Frozen selected SS inference. No GT, post-embedding transforms, or TTA."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .audit import fallback
from experiments.native_localizer.model import NativeLocalizer
from experiments.native_localizer.common import atomic_json,rgb_tensor,sha256
from wmloc.data import LocalisationDataset
from watermark_localization.submission import save_submission,validate_submission


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--part',type=int,choices=range(4),required=True);p.add_argument('--extended',action='store_true');a=p.parse_args();recipe_path=a.root/('next-ideas-extended-runs' if a.extended else 'next-ideas-runs')/'selected_recipe.json';cfg=json.loads(recipe_path.read_text());assert cfg['recipe_frozen'] and cfg['threshold']==.5 and not cfg['tta'];torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=Path(cfg['checkpoint']);state=torch.load(source,map_location='cpu',weights_only=False)
    assert state['config']['family']=='stegastamp'
    if cfg['needs_all170_refit']:assert state['config']['all_real'] and state['config']['real_train_n']==170 and state['step']==cfg['updates']
    else:assert state['config']['pilot'] and state['update']==200
    model=NativeLocalizer(checkpointing=False,amp=False).cuda().eval();model.load_state_dict(state['model'],strict=True);del state
    ds=LocalisationDataset('data','test',methods=['stegastamp']);assert len(ds)==340 and not ds.train;out=a.root/('next-ideas-extended-production' if a.extended else 'next-ideas-production')/'stegastamp';out.mkdir(parents=True,exist_ok=True);dest=out/f'part{a.part}.pt';assert not dest.exists();names=[];masks=[];probabilities=[];changed=[];start=time.monotonic();indices=list(range(a.part,340,4))
    with torch.inference_mode():
        for begin in range(0,len(indices),4):
            items=[ds[i] for i in indices[begin:begin+4]];x=torch.stack([rgb_tensor(it['image']) for it in items]).cuda();pp=model(x)['logits'].sigmoid().cpu().numpy()
            for it,prob in zip(items,pp):
                before=prob>=.5;after=fallback(prob,'top1pct') if cfg['postprocess']=='top1pct' else before
                names.append(it['image_name']);masks.append(after.astype(np.uint8));probabilities.append(prob)
                if not np.array_equal(before,after):changed.append(it['image_name'])
    save_submission(dest,names,torch.tensor(np.stack(masks)),names);validate_submission(torch.load(dest,weights_only=True),names);np.savez_compressed(out/f'part{a.part}.npz',names=names,probabilities=probabilities)
    atomic_json(out/f'part{a.part}.json',{'n':len(names),'seconds':time.monotonic()-start,'checkpoint_sha256':sha256(source),'recipe_sha256':sha256(recipe_path),'artifact_sha256':sha256(dest),'postprocess_changed':changed,'test_GT_used':False,'recipe':cfg});print('SS_TEST_DONE',a.part,len(names),time.monotonic()-start,flush=True)


if __name__=='__main__':main()
