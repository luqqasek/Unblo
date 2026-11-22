"""Deploy the measured stronger overall checkpoint, then run one PCMV screen."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .base_candidate import GlobalCarrierBase
from experiments.triforensic_wm.model import TriForensicWM
from experiments.native_localizer.common import atomic_json,rgb_tensor,sha256
from wmloc.data import LocalisationDataset
from watermark_localization.submission import save_submission


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--infer-only',action='store_true');a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='0,1';torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'pcmv-runs/base-candidate';evaluation=json.loads((out/'evaluation.json').read_text());assert evaluation['delta']>.003
    atomic_json(out/'decision.json',{'selected':True,'reason':'User prioritized strongest overall checkpoint and immediate predictions. Overall gain>0.003; minority regression disclosed. Earlier completed job used prior minority veto and produced no test artifact.','previous_selection':evaluation['selected_for_test'],'delta':evaluation['delta'],'minority_guard':evaluation['minority_guard'],'gpu_devices':[0,1]})
    if not (out/'test-inference.json').exists():
        start=time.monotonic();source=Path(evaluation['checkpoint']);assert sha256(source)==evaluation['checkpoint_sha256'];tri=TriForensicWM('pixelseal',checkpointing=False).cuda();state=torch.load(source,map_location='cpu',weights_only=False);tri.load_state_dict(state['ema'],strict=True);del state;model=nn.DataParallel(GlobalCarrierBase(tri).eval().requires_grad_(False))
        cached=np.load(out/'all-real-maps.npz');lookup={str(n):i for i,n in enumerate(cached['names'])};ds=LocalisationDataset('data','train',methods=['pixelseal'])
        with torch.inference_mode():
            items=[ds[i] for i in range(8)];x=torch.stack([rgb_tensor(it['image']) for it in items]).cuda();b=torch.tensor(np.stack([it['message'] for it in items]),device='cuda').float();pr=model(x,b)['p'].cpu().numpy();error=max(float(np.max(np.abs(p-cached['probabilities'][lookup[it['image_name']]]))) for it,p in zip(items,pr));assert error<1e-4,error
            test=LocalisationDataset('data','test',methods=['pixelseal']);tp=[];names=[]
            for begin in range(0,len(test),16):
                items=[test[i] for i in range(begin,min(begin+16,len(test)))];x=torch.stack([rgb_tensor(it['image']) for it in items]).cuda();b=torch.tensor(np.stack([it['message'] for it in items]),device='cuda').float();tp.extend(model(x,b)['p'].cpu().numpy());names.extend(it['image_name'] for it in items)
        assert len(names)==len(set(names))==340;save_submission(out/'pixelseal-subset.pt',names,torch.tensor(np.array(tp)>=.5,dtype=torch.uint8),names);np.savez_compressed(out/'test-probabilities.npz',names=names,probabilities=tp)
        atomic_json(out/'test-inference.json',{'n':340,'checkpoint_sha256':evaluation['checkpoint_sha256'],'artifact_sha256':sha256(out/'pixelseal-subset.pt'),'test_GT_used':False,'threshold':.5,'tta':False,'training_scope':'113 real fold-training images + original synthetic training; no all170 refit','gpu_devices':[0,1],'replica_reproduction_max_probability_error':error,'seconds':time.monotonic()-start});print('PS_TEST_READY',time.monotonic()-start,flush=True)
    return None if a.infer_only else a.root


if __name__=='__main__':
    root=main()
    if root is not None:os.execv(sys.executable,[sys.executable,'-u','-m','experiments.pcmv.screen','--root',str(root),'--world','2'])
