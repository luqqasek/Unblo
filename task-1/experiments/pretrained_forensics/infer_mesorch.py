"""Official MesorchFull source and weights, bypassing training-only registry/DDP launcher."""
import argparse
import sys
import time
import subprocess
import types
from common import *
from PIL import Image
import torch
import torch.nn.functional as F

def load_model():
    vendor=OUT/'vendor/Mesorch'; sys.path.insert(0,str(vendor))
    source=(vendor/'mesorch.py').read_text(encoding='utf-8')
    # Remove only registry import/decorator. No numerical/model code is altered.
    assert source.count('from IMDLBenCo.registry import MODELS')==1
    assert source.count('@MODELS.register_module()')==1
    source=source.replace('from IMDLBenCo.registry import MODELS','').replace('@MODELS.register_module()','')
    module=types.ModuleType('official_mesorch_inference'); module.__file__=str(vendor/'mesorch.py')
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    model=module.MesorchFull()
    ck=OUT/'weights/mesorch-98.pth'; state=torch.load(ck,map_location='cpu',weights_only=False)['model']
    model.load_state_dict(state,strict=True)
    return model.eval().requires_grad_(False).cuda(0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=510); args=ap.parse_args()
    torch.set_num_threads(4); model=load_model()
    records,_,_,_,_=cohort(); cache=OUT/'mesorch';cache.mkdir(exist_ok=True)
    save(OUT/'inference_mesorch.json',dict(checkpoint_sha256=sha(OUT/'weights/mesorch-98.pth'),
         vendor_commit=subprocess.check_output(['git','-C',str(OUT/'vendor/Mesorch'),'rev-parse','HEAD'],text=True).strip(),
         preprocessing='RGB /255 external ImageNet normalization; 512 unchanged; zeros dummy mask only for unused loss',
         adaptation='MesorchFull direct constructor; remove only IMDLBenCo registry import/decorator at runtime; strict weights',
         output='already sigmoid pred_mask; native512 area average128; binary>=0.5',torch=torch.__version__,device=torch.cuda.get_device_name(0)))
    mean=torch.tensor([.485,.456,.406],device='cuda:0')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda:0')[None,:,None,None]
    dummy=torch.zeros((1,1,512,512),device='cuda:0')
    for i,r in enumerate(records[:args.limit]):
        path=cache/f'{r["image_id"]}.npz'
        if path.exists():continue
        rgb=np.asarray(Image.open(ROOT/'data/train/images'/f'{r["image_id"]}.png').convert('RGB'))
        x=(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().cuda(0)/255.-mean)/std
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode():
            p=model(x,dummy)['pred_mask']; p128=F.interpolate(p,size=(128,128),mode='area')
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        native=p[0,0].cpu().numpy(); small=p128[0,0].cpu().numpy();assert native.shape==(512,512) and np.isfinite(native).all()
        binary=small>=.5
        np.savez_compressed(path,probability=small,probability512=native,binary=binary,complement=~binary,binary512=native>=.5,
                            native_binary128=F.interpolate((p>=.5).float(),size=(128,128),mode='nearest')[0,0].cpu().numpy().astype(bool),
                            seconds=elapsed,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                            peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
        print(f'Mesorch {i+1}/510 {r["family"]}: {elapsed:.3f}s peak {torch.cuda.max_memory_allocated()/2**20:.0f}MB range {small.min():.4f}..{small.max():.4f}',flush=True)

if __name__=='__main__':main()
