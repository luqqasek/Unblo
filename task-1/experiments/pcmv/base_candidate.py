"""Verify stronger PS checkpoint without Noiseprint; predict only if measured gain survives."""
import argparse,json,time,os,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.triforensic_wm.model import TriForensicWM
from experiments.triforensic_wm.evaluate import summarize
from experiments.native_localizer.common import atomic_json,rgb_tensor,sha256
from experiments.native_localizer.evaluate import image_metrics
from wmloc.data import LocalisationDataset
from watermark_localization.submission import save_submission

class GlobalCarrierBase(nn.Module):
    def __init__(self,tri):super().__init__();self.tri=tri
    def forward(self,x,b):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f,l,obs=self.tri.local_forward(x);z=self.tri.global_forward(x,b,f,obs)
        return {'p':z[:,0].float().sigmoid(),'local':l[:,0].float(),'obs':torch.nn.functional.avg_pool2d(obs[:,:1],4)[:,0]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base-only',action='store_true');a=p.parse_args();root=a.root;out=root/'pcmv-runs/base-candidate';out.mkdir(parents=True,exist_ok=True)
    # User explicitly reserves physical GPUs2/3 for a separate experiment.
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    if (out/'completed.json').exists():return None if a.base_only else root
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.monotonic();source=root/'triforensic-runs/pixelseal/unified-v1/resume.pt';state=torch.load(source,map_location='cpu',weights_only=False);assert state['step']==1561
    tri=TriForensicWM('pixelseal',checkpointing=False).cuda();tri.load_state_dict(state['ema'],strict=True);del state;model=GlobalCarrierBase(tri).eval().requires_grad_(False);parallel=nn.DataParallel(model)
    split=json.loads((root/'triforensic-runs/pixelseal/unified-v1/split.json').read_text());ds=LocalisationDataset('data','train',methods=['pixelseal']);valid=set(split['validation']);assert len(valid)==57
    names=[];probs=[];local=[];obs=[];targets=[];rows=[]
    with torch.inference_mode():
        for begin in range(0,len(ds),16):
            items=[ds[i] for i in range(begin,min(begin+16,len(ds)))];x=torch.stack([rgb_tensor(it['image']) for it in items]).cuda();b=torch.tensor(np.stack([it['message'] for it in items]),device='cuda').float();d=parallel(x,b)
            for j,it in enumerate(items):
                n=it['image_name'];prob=d['p'][j].cpu().numpy();names.append(n);probs.append(prob);local.append(d['local'][j].cpu().numpy());obs.append(d['obs'][j].cpu().numpy());targets.append(it['mask128'])
                if n in valid:rows.append(dict(image_metrics(prob,it['mask128']),image_name=n))
    np.savez_compressed(out/'all-real-maps.npz',names=names,probabilities=probs,local_logits=local,observability=obs,targets=targets)
    old=np.load(root/'local-presence-runs/ps-data-small/maps-150.npz');before={str(n):image_metrics(pr,y) for n,pr,y in zip(old['names'],old['probabilities'],old['targets'])};assert set(before)==valid
    delta=np.array([r['miou']-before[r['image_name']]['miou'] for r in rows]);rng=np.random.default_rng(42);ci=np.quantile(delta[rng.integers(57,size=(5000,57))].mean(1),[.025,.975]);s=summarize(rows);s0=summarize(list(before.values()));guard=s['fg_le1']['fg_iou']>=s0['fg_le1']['fg_iou']-.01 and s['fg_1to5']['fg_iou']>=s0['fg_1to5']['fg_iou']-.01 and s['wm_over50']['clean_iou']>=s0['wm_over50']['clean_iou']-.01
    selected=bool(delta.mean()>.003)
    result={'family':'pixelseal','checkpoint':str(source),'checkpoint_sha256':sha256(source),'checkpoint_updates':1561,'weights':'EMA','architecture':'TriForensic global branch + detached carrier skips; refiner bypassed; no Noiseprint inputs or weights','baseline':s0,'summary':s,'per_image':rows,'delta':float(delta.mean()),'ci95':ci.tolist(),'minority_guard':bool(guard),'selected_for_test':selected,'selection_rule':'Overall development mIoU gain >0.003; minority regressions reported, not hard-vetoed per user priority to use the strongest overall checkpoint. Fixed before this inference.','protocol':'Existing57-image developmental fold, one seed; baseline1561 previously selected; exact deployable branch verified before test; not unbiased estimate','real_training_images':113,'all170_refit':False}
    atomic_json(out/'evaluation.json',result);print('BASE_CANDIDATE',json.dumps({k:result[k] for k in ('delta','ci95','selected_for_test')}),s['all']['miou'],flush=True)
    if selected:
        test=LocalisationDataset('data','test',methods=['pixelseal']);tp=[];tn=[]
        with torch.inference_mode():
            for begin in range(0,len(test),16):
                items=[test[i] for i in range(begin,min(begin+16,len(test)))];x=torch.stack([rgb_tensor(it['image']) for it in items]).cuda();b=torch.tensor(np.stack([it['message'] for it in items]),device='cuda').float();pr=parallel(x,b)['p'].cpu().numpy();tp.extend(pr);tn.extend(it['image_name'] for it in items)
        assert len(tn)==340 and len(set(tn))==340;save_submission(out/'pixelseal-subset.pt',tn,torch.tensor(np.array(tp)>=.5,dtype=torch.uint8),tn);np.savez_compressed(out/'test-probabilities.npz',names=tn,probabilities=tp)
        atomic_json(out/'test-inference.json',{'n':340,'checkpoint_sha256':result['checkpoint_sha256'],'artifact_sha256':sha256(out/'pixelseal-subset.pt'),'test_GT_used':False,'threshold':.5,'tta':False,'training_scope':'113 real fold-training images + original synthetic training; no all170 refit'})
    atomic_json(out/'completed.json',{'seconds':time.monotonic()-start,'selected':selected,'gpus':torch.cuda.device_count(),'validation_miou':s['all']['miou']});print('BASE_DONE',time.monotonic()-start,flush=True)
    return None if a.base_only else root
if __name__=='__main__':
    root=main()
    # Replace this process: releases all CUDA memory before feature workers,
    # retaining the same Slurm node and avoiding a second queue wait.
    if root is not None:os.execv(sys.executable,[sys.executable,'-u','-m','experiments.pcmv.screen','--root',str(root),'--world','2'])
