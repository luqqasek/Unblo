"""FP32 local evidence: SRM30(Y), constrained RGB prediction errors, RF=27."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Bayar(nn.Module):
    def __init__(self):
        super().__init__()
        self.surround=nn.Parameter(torch.randn(8,3,24)*.01+1/24)
    def kernel(self):
        w=self.surround-self.surround.mean(-1,keepdim=True)+1/24
        return torch.cat([w[:,:,:12],-torch.ones_like(w[:,:,:1]),w[:,:,12:]],-1).reshape(8,3,5,5)/3
    def forward(self,x):
        return F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),self.kernel())

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return x+.1*self.net(x)

class Presence(nn.Module):
    receptive_field=27
    def __init__(self):
        super().__init__()
        bank=np.load(Path(__file__).with_name('SRM_Kernels.npy'))
        assert bank.shape==(30,1,5,5)
        self.register_buffer('srm',torch.tensor(bank,dtype=torch.float32))
        self.register_buffer('luma',torch.tensor([.299,.587,.114])[None,:,None,None])
        self.bayar=Bayar()
        self.input=nn.Conv2d(43,64,1)
        self.blocks=nn.Sequential(*[Residual(64) for _ in range(4)])
        self.down=nn.Sequential(nn.Conv2d(64,64,3,2,1),nn.GELU(),nn.Conv2d(64,64,3,2,1),nn.GELU())
        self.head=nn.Conv2d(64,1,1)
    def forward(self,x):
        assert x.dtype==torch.float32
        y=(x*self.luma).sum(1,keepdim=True)
        # Pixel units for small residuals; no casting raw RGB or learned image projection.
        srm=F.conv2d(F.pad(y*255,(2,2,2,2),mode='reflect'),self.srm).clamp(-3,3)
        bayar=self.bayar(x*255).clamp(-3,3)
        hp=(x-F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,1))*255
        texture=hp.abs().mean(1,keepdim=True).clamp(0,32)/8
        f=self.down(self.blocks(F.gelu(self.input(torch.cat([srm,bayar,hp.clamp(-3,3),y,texture],1)))))
        return self.head(f)[:,0],f

class Correction(nn.Module):
    def __init__(self,holes=False):
        super().__init__();self.head=nn.Conv2d(65,1,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias);self.holes=holes
    def forward(self,base,features):
        raw=torch.tanh(self.head(torch.cat([features,base[:,None].clamp(-8,8)/8],1)))[:,0]
        delta=-8*raw.clamp_min(0) if self.holes else 8*raw
        return base+delta

def balanced(logits,y):
    bce=F.binary_cross_entropy_with_logits(logits.float(),y.float(),reduction='none')
    parts=[];counts=[]
    for c in (0,1):
        m=y==c;n=m.sum((-2,-1));parts.append((bce*m).sum((-2,-1))/n.clamp_min(1));counts.append(n>0)
    return ((parts[0]+parts[1])/(counts[0].float()+counts[1].float()).clamp_min(1)).mean()

def soft_iou(logits,y):
    p=logits.float().sigmoid();loss=0
    for a,b in ((p,y),(1-p,1-y)):
        inter=(a*b).sum((-2,-1));union=(a+b-a*b).sum((-2,-1))
        loss=loss+(1-(inter+1e-3)/(union+1e-3)).mean()/2
    return loss
