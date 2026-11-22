"""Single-image carrier checkpoint overhead, without optimizer/model mutations."""
import argparse,time
import torch
from .model import Carrier
from experiments.native_localizer.common import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(2)
    model=Carrier().cuda().train();x=torch.rand(1,3,512,512,device='cuda');obs=torch.rand(1,2,512,512,device='cuda');results={}
    for flag in (True,False):
        model.checkpointing=flag;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        values=[]
        for i in range(4):
            t=time.monotonic();model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):_,z=model(x,obs)
            z.square().mean().backward();torch.cuda.synchronize();values.append(time.monotonic()-t)
        results[str(flag)]={'mean_seconds':sum(values[1:])/3,'peak_bytes':torch.cuda.max_memory_allocated()}
    atomic_json(a.out,results);print(results)
if __name__=='__main__':main()
