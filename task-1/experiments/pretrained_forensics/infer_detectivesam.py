"""Official DetectiveSAM runner; GT is never supplied, including seed generation."""
import argparse
import sys
import time
import subprocess
import site
from common import *
import torch
import torch.nn.functional as F

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=510); args=ap.parse_args()
    vendor=OUT/'vendor/DetectiveSAM'; sam=OUT/'vendor/sam2'
    sys.path[:0]=[str(OUT/'deps'),str(vendor),str(sam)]
    site.addsitedir(str(OUT/'deps'))  # pywin32 .pth needed by iopath on Windows
    from detectivesam_inference.runtime import DetectiveSAMRunner
    from detectivesam_inference.dataset import prepare_sample
    torch.set_num_threads(4)
    ck=vendor/'checkpoints/detective_sam_sota.pth'; samck=vendor/'sam2configs/sam2.1_hiera_base_plus.pt'
    assert sha(ck)=='0c7c33c4c4ddb82459bc2b6bbfdfd1003561e2714d02163b3daffb25ede4e1cf'
    assert sha(samck)=='a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
    runner=DetectiveSAMRunner(checkpoint_path=ck,device='cuda:0')
    runner.model.requires_grad_(False)
    records,_,_,_,_=cohort(); cache=OUT/'detectivesam'; cache.mkdir(exist_ok=True)
    cfg=runner.config
    save(OUT/'inference_detectivesam.json',dict(checkpoint_sha256=sha(ck),sam_checkpoint_sha256=sha(samck),
         vendor_commit=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),
         sam_commit=subprocess.check_output(['git','-C',str(sam),'rev-parse','HEAD'],text=True).strip(),
         preprocessing='official prepare_sample; source=target; mask=None; RGB /255 ImageNet norm; 512; official blur/noise inference streams',
         output='sigmoid logits; CUDA official autocast; float32 cache of output probabilities; area average128',
         binary='strict >0.5 per official runner; native binary nearest128',device=torch.cuda.get_device_name(0),torch=torch.__version__))
    for i,r in enumerate(records[:args.limit]):
        path=cache/f'{r["image_id"]}.npz'
        if path.exists(): continue
        src=ROOT/'data/train/images'/f'{r["image_id"]}.png'
        start=time.perf_counter()
        sample=prepare_sample(src,src,None,cfg.img_size,cfg.perturbation_type,cfg.perturbation_intensity)
        prep=time.perf_counter()-start
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        pred=runner.predict_sample(sample)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        native=pred.probability.astype('float32'); assert native.shape==(512,512) and np.isfinite(native).all()
        p=F.interpolate(torch.from_numpy(native)[None,None],size=(128,128),mode='area')[0,0].numpy()
        binary=p>.5
        np.savez_compressed(path,probability=p,probability512=native,binary=binary,complement=~binary,
                            binary512=pred.pred_mask.astype(bool),native_binary128=F.interpolate(torch.from_numpy(pred.pred_mask.astype('float32'))[None,None],size=(128,128),mode='nearest')[0,0].numpy().astype(bool),
                            seconds=elapsed,preprocess_seconds=prep,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                            peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
        print(f'DetectiveSAM {i+1}/510 {r["family"]}: {elapsed:.3f}s peak {torch.cuda.max_memory_allocated()/2**20:.0f}MB range {p.min():.4f}..{p.max():.4f}',flush=True)

if __name__=='__main__': main()
