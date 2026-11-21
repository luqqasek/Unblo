"""Official frozen TruFor, native 512 grid, full float32 maps; resumable train-only inference."""
import argparse
import sys
import time
import zipfile
import hashlib
import shutil
import subprocess
from common import *
from PIL import Image
import torch
import torch.nn.functional as F

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=510); args=ap.parse_args()
    torch.set_num_threads(4)
    vendor=OUT/'vendor/TruFor/TruFor_train_test'
    sys.path[:0]=[str(OUT/'deps'),str(vendor)]
    archive=OUT/'TruFor_weights.zip'
    with archive.open('rb') as f: assert hashlib.file_digest(f,'md5').hexdigest()=='7bee48f3476c75616c3c5721ab256ff8'
    weights=OUT/'weights'; weights.mkdir(exist_ok=True)
    if not list(weights.rglob('trufor.pth.tar')):
        with zipfile.ZipFile(archive) as z: z.extractall(weights)
    checkpoint=next(weights.rglob('trufor.pth.tar'))
    from lib.config import config
    from lib.utils import get_model
    config.defrost(); config.merge_from_file(str(vendor/'lib/config/trufor_ph3.yaml')); config.freeze()
    torch.backends.cudnn.enabled=config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark=config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic=config.CUDNN.DETERMINISTIC
    model=get_model(config)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict'],strict=True); model.eval().requires_grad_(False).cuda(0)
    records,base,gt,splits,provenance=cohort()
    cache=OUT/'trufor'; cache.mkdir(exist_ok=True)
    (OUT/'inputs/rgb').mkdir(parents=True,exist_ok=True); (OUT/'inputs/gt').mkdir(exist_ok=True)
    info=dict(model='TruFor',checkpoint_sha256=sha(checkpoint),archive_md5='7bee48f3476c75616c3c5721ab256ff8',
              vendor_commit=subprocess.check_output(['git','-C',str(vendor.parent),'rev-parse','HEAD'],text=True).strip(),
              torch=torch.__version__,device=torch.cuda.get_device_name(0),cudnn_enabled=torch.backends.cudnn.enabled,
              preprocessing='PIL RGB float32 CHW /256; internal ImageNet RGB normalization; no resizing',
              output='softmax channel1 manipulation, sigmoid confidence/detection; float32 native512; area average128',
              binary='native probability>=0.5; evaluation128 probability>=0.5; native binary nearest128 also saved',
              baseline_sources=provenance)
    save(OUT/'inference_trufor.json',info); save(OUT/'cohort.json',records)
    for i,r in enumerate(records[:args.limit]):
        n=r['image_id']; path=cache/f'{n}.npz'
        if path.exists(): continue
        src=ROOT/'data/train/images'/f'{n}.png'
        rgb=np.asarray(Image.open(src).convert('RGB')); assert rgb.shape==(512,512,3)
        x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().cuda(0)/256.
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        with torch.inference_mode():
            logits,conf,det,_=model(x)
            p=logits.softmax(1)[:,1:2]; c=conf.sigmoid()
            p128=F.interpolate(p,size=(128,128),mode='area'); c128=F.interpolate(c,size=(128,128),mode='area')
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        native=p[0,0].cpu().numpy(); p128=p128[0,0].cpu().numpy(); assert native.shape==(512,512)
        binary=p128>=.5
        np.savez_compressed(path,probability=p128,probability512=native,confidence=c128[0,0].cpu().numpy(),
                            confidence512=c[0,0].cpu().numpy(),detection=float(det.sigmoid().item()),
                            binary=binary,complement=~binary,binary512=native>=.5,
                            native_binary128=F.interpolate((p>=.5).float(),size=(128,128),mode='nearest')[0,0].cpu().numpy().astype(bool),
                            seconds=elapsed,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                            peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
        shutil.copy2(src,OUT/'inputs/rgb'/f'{n}.png')
        Image.fromarray(gt[i].astype('uint8')*255).save(OUT/'inputs/gt'/f'{n}.png')
        print(f'TruFor {i+1}/510 {r["family"]} {n}: {elapsed:.3f}s, peak {torch.cuda.max_memory_allocated()/2**20:.0f}MB, range {p128.min():.4f}..{p128.max():.4f}',flush=True)

if __name__=='__main__': main()
