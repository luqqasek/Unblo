"""Fixed full-data refit and inference after validation recipe is frozen."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from watermark_localization.submission import save_submission,validate_submission
from experiments.message_localizer.model import MessageLocalizer
from experiments.native_localizer.common import atomic_json,seed_all,rgb_tensor,sha256
from experiments.region_data.data import RegionData
from .tiny_train import backward_micro,optimizer,ResidualModel
from .tiny_data import TinyData
from .proposals import detector,infer,boxmask

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--source',default='region');a=p.parse_args()
    root=a.root;family=a.family;out=root/'small-zoom-production'/family;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    is_tiny=a.source!='region';seed=941 if is_tiny else 42;seed_all(seed)
    checkpoint=root/(f'tiny-runs/{a.source}/last.pt' if is_tiny else f'region-runs/{family}/{"control" if family=="pixelseal" else "objects"}/last.pt')
    cfg={'family':family,'full_refit':True,'source':a.source,'source_checkpoint':str(checkpoint),'source_sha256':sha256(checkpoint),'steps':150,'batch':32,'microbatch':16,'synthetic':24,'real':8,'threshold':.5,'encoder_lr':2e-6,'new_lr':2e-5,'seed':seed,'tta':False,'proposal_scales':[512,1536,2560,4096] if family=='trustmark' else [],'proposal_bit_accuracy_threshold':.9,'recipe_frozen_before_test':True}
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg
    else:atomic_json(out/'config.json',cfg)
    model=ResidualModel().cuda() if is_tiny else MessageLocalizer(family).cuda()
    if not (out/'last.pt').exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);del state;model.freeze_encoder(False);model.train()
        data=TinyData(root) if is_tiny else RegionData(root,family,'train');real=LocalisationDataset('data','train',methods=[family]);opt=optimizer(model,2e-6,2e-5);rng=np.random.default_rng(seed);rrng=np.random.default_rng(12342);start=time.monotonic()
        for step in range(150):
            if is_tiny:x,y,b,pairs,d,regions,valid=data.batch(rng,24,a.source!='control')
            else:
                x,y,pairs,meta=data.batch(rng,24,family=='trustmark');b=torch.tensor([data.messages[m['realization_id']] for m in meta]).float();d=torch.zeros((24,512,512));regions=torch.zeros_like(d,dtype=torch.bool);valid=torch.zeros(24,dtype=torch.bool)
            items=[real[int(i)] for i in rrng.choice(len(real),8,replace=True)]
            x=torch.cat([x,torch.stack([rgb_tensor(it['image']) for it in items])]).cuda();y=torch.cat([y,torch.stack([torch.tensor(it['mask128']).float() for it in items])]).cuda();b=torch.cat([b,torch.tensor(np.array([it['message'] for it in items])).float()]).cuda()
            opt.zero_grad(set_to_none=True);backward_micro(model,x,y,b,pairs,d,regions,valid,a.source=='tiny_aux',.1,.05);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
            if (step+1)%50==0:print('FULL_REFIT',family,step+1,'seconds',time.monotonic()-start,flush=True)
        torch.save({'model':model.state_dict(),'config':cfg,'step':150},out/'last.pt');del data,real,opt,x,y,b;torch.cuda.empty_cache()
    else:model.load_state_dict(torch.load(out/'last.pt',map_location='cpu',weights_only=False)['model'])
    model.eval();dest=out/f'{family}-subset.pt'
    if dest.exists():print('Preserving completed test inference',dest,flush=True);return
    test=LocalisationDataset('data','test',methods=[family]);assert len(test)==340
    names=[];probs=[];start=time.monotonic()
    with torch.inference_mode():
        for begin in range(0,len(test),16):
            items=[test[i] for i in range(begin,min(begin+16,len(test)))];x=torch.stack([rgb_tensor(it['image']) for it in items]).cuda();b=torch.tensor(np.array([it['message'] for it in items]),device='cuda').float()
            prob=model(x,b)['logits'].sigmoid().cpu().numpy();assert np.isfinite(prob).all();probs.extend(prob);names.extend(it['image_name'] for it in items)
    del model,x,b;torch.cuda.empty_cache();masks=np.stack(probs)>=.5;records=[]
    if family=='trustmark':
        det=detector(root);wm=load_watermark(family,models_dir='checkpoints',device='cuda')
        for i in range(len(test)):
            it=test[i];r=infer(it['image'],it['message'],it['image_name'],det,wm,scales=tuple(cfg['proposal_scales']));records.append(r)
            if r['correct_top'] and r['correct_top'][0]['bit_accuracy']>=.9:masks[i] |= boxmask(r['correct_top'][0]['box'])
            if (i+1)%50==0:print('TEST_CROP',i+1,'seconds',time.monotonic()-start,flush=True)
        atomic_json(out/'test_proposals.json',records)
    save_submission(dest,names,torch.from_numpy(masks.astype(np.uint8)),names);validate_submission(torch.load(dest,weights_only=True),names)
    np.savez_compressed(out/'test_base_probabilities.npz',names=names,probabilities=np.stack(probs))
    atomic_json(out/'inference.json',{'n':len(names),'checkpoint_sha256':sha256(out/'last.pt'),'subset_sha256':sha256(dest),'seconds':time.monotonic()-start,'corrected_count':sum(bool(r['correct_top']) and r['correct_top'][0]['bit_accuracy']>=.9 for r in records),'shape':list(masks.shape),'binary':True,'threshold':.5})
    print('PRODUCTION_COMPLETE',family,flush=True)
if __name__=='__main__':main()
