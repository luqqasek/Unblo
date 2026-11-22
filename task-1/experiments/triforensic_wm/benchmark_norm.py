"""Equivalent local normalization kernels; read-only GPU microbenchmark."""
import argparse,json,time
import torch
from torch.nn import functional as F
from experiments.native_localizer.common import atomic_json
from .model import ChannelGroupNorm

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    norm=ChannelGroupNorm(160).cuda();x=torch.randn(2,160,128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    def fused(x):
        b,c,h,w=x.shape;v=x.permute(0,2,3,1).float().reshape(b,h,w,8,c//8)
        v=F.layer_norm(v,(c//8,),None,None,1e-5).reshape(b,h,w,c)
        return (v*norm.weight+norm.bias).permute(0,3,1,2).to(x.dtype).contiguous()
    o1=norm(x);g1=torch.autograd.grad(o1.float().square().sum(),x)[0];o2=fused(x);g2=torch.autograd.grad(o2.float().square().sum(),x)[0]
    result={'max_abs_forward':float((o1.float()-o2.float()).abs().max()),'max_abs_backward':float((g1.float()-g2.float()).abs().max())}
    for name,fn in [('explicit',norm),('fused_layernorm',fused)]:
        for _ in range(3):fn(x).float().square().mean().backward();x.grad=None
        torch.cuda.synchronize();t=time.monotonic()
        for _ in range(20):fn(x).float().square().mean().backward();x.grad=None
        torch.cuda.synchronize();result[name+'_ms']=(time.monotonic()-t)*50
    atomic_json(a.out,result);print(result)
if __name__=='__main__':main()
