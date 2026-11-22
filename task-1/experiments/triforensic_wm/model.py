"""Unified local carrier / frozen Noiseprint / trainable message localization.

Carrier RF29 is exact; the refiner convolutional body RF7 at stride4 is NOT
the effective RF of its inputs (ring contrast and Noiseprint global median).
"""
from pathlib import Path
import math
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from experiments.message_localizer.model import MessageLocalizer
from experiments.message_localizer.extractor import match_maps, resize


class ChannelGroupNorm(nn.Module):
    """Eight channel groups independently at each pixel, no spatial reduction."""
    def __init__(self, channels, groups=8):
        super().__init__(); assert channels % groups == 0
        self.groups=groups; self.weight=nn.Parameter(torch.ones(channels)); self.bias=nn.Parameter(torch.zeros(channels))
    def forward(self,x):
        b,c,h,w=x.shape; v=x.float().reshape(b,self.groups,c//self.groups,h,w)
        mean=v.mean(2,keepdim=True); var=(v-mean).square().mean(2,keepdim=True)
        v=((v-mean)*torch.rsqrt(var+1e-5)).reshape(b,c,h,w)
        return (v*self.weight[None,:,None,None]+self.bias[None,:,None,None]).to(x.dtype)


def block(ci,co,stride=1):
    return nn.Sequential(nn.Conv2d(ci,co,3,stride,1),ChannelGroupNorm(co),nn.GELU())


class Bayar(nn.Module):
    def __init__(self):
        super().__init__(); self.surround=nn.Parameter(torch.randn(8,3,24)*.01+1/24)
    def kernel(self):
        w=self.surround-self.surround.mean(-1,keepdim=True)+1/24
        return torch.cat([w[:,:,:12],-torch.ones_like(w[:,:,:1]),w[:,:,12:]],-1).reshape(8,3,5,5)
    def forward(self,x):
        return F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),self.kernel())


class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.a=block(c,c);self.b=block(c,c)
    def forward(self,x):return x+self.b(self.a(x))


class Carrier(nn.Module):
    receptive_field=29
    def __init__(self,width=160,checkpointing=True):
        super().__init__(); self.checkpointing=checkpointing
        bank_path=Path(__file__).resolve().parents[1]/'local_presence/SRM_Kernels.npy'
        assert hashlib.sha256(bank_path.read_bytes()).hexdigest()=='c53ad3a849fa64db1244152a3030c80a4d3b11fec0af81b4196d8e6dfb3db4b9'
        bank=np.load(bank_path)
        assert bank.shape==(30,1,5,5)
        self.register_buffer('srm',torch.tensor(bank,dtype=torch.float32).repeat(3,1,1,1))
        self.register_buffer('luma',torch.tensor([.299,.587,.114])[None,:,None,None])
        self.bayar=Bayar();self.stem=block(104,width)
        self.blocks=nn.ModuleList([Residual(width) for _ in range(4)])
        self.down=nn.Sequential(block(width,64,2),block(64,64,2));self.head=nn.Conv2d(64,1,1)
    def filters(self,x,obs):
        with torch.autocast(x.device.type,enabled=False):
            assert x.dtype==torch.float32
            s=F.conv2d(F.pad(x*255,(2,2,2,2),mode='reflect'),self.srm,groups=3).clamp(-3,3)
            b=self.bayar(x*255).clamp(-3,3)
            hp=((x-F.avg_pool2d(F.pad(x,(1,1,1,1),mode='reflect'),3,1))*255).clamp(-3,3)
            y=(x*self.luma).sum(1,keepdim=True)
            return torch.cat([s,b,hp,y,obs],1)
    def forward(self,x,obs):
        with torch.autocast(x.device.type,enabled=False):
            f=self.stem(self.filters(x,obs))
        for layer in self.blocks:
            f=checkpoint(layer,f,use_reentrant=False) if self.training and self.checkpointing else layer(f)
        f=self.down(f);return f,self.head(f).float()


def observability(x,jnd=None):
    """RF <=5, exact PixelSeal JND plus local texture, FP32 throughout."""
    with torch.autocast(x.device.type,enabled=False):
        y=(x*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
        k=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])[None,None]/8
        gx=F.conv2d(F.pad(y,(1,1,1,1),mode='reflect'),k)
        gy=F.conv2d(F.pad(y,(1,1,1,1),mode='reflect'),k.transpose(-1,-2))
        grad=torch.sqrt(gx.square()+gy.square()+1e-12)
        if jnd is not None:
            with torch.no_grad():first=jnd.heatmaps(x).mean(1,keepdim=True)
            return torch.cat([first,grad],1)
        lap=F.conv2d(F.pad(y,(1,1,1,1),mode='reflect'),x.new_tensor([[0,1,0],[1,-4,1],[0,1,0]])[None,None]).abs()
        return torch.cat([F.avg_pool2d(F.pad(grad,(1,1,1,1),mode='reflect'),3,1),F.avg_pool2d(F.pad(lap,(1,1,1,1),mode='reflect'),3,1)],1)


def ring_contrast(x,inner=3,outer=9):
    # Normalize by actual available ring pixels near borders, not padded zeros.
    v=x.float();ones=torch.ones_like(v[:,:1]);ko=2*outer+1;ki=2*inner+1
    total=F.avg_pool2d(v,ko,1,outer,count_include_pad=True)*ko**2-F.avg_pool2d(v,ki,1,inner,count_include_pad=True)*ki**2
    n=F.avg_pool2d(ones,ko,1,outer,count_include_pad=True)*ko**2-F.avg_pool2d(ones,ki,1,inner,count_include_pad=True)*ki**2
    return (v-total/n.clamp_min(1)).to(x.dtype)


class Refiner(nn.Module):
    body_receptive_field=7
    def __init__(self):
        super().__init__();self.body=nn.Sequential(block(133,64),block(64,64),block(64,64))
        self.wm=nn.Conv2d(64,1,1);self.clean=nn.Conv2d(64,1,1);self.edge=nn.Conv2d(64,1,1)
        for h in (self.wm,self.clean):nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
        self.raw_alpha_wm=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.raw_alpha_clean=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
    def forward(self,f,l,npmap,obs):
        n=F.dropout2d(npmap,p=.15,training=self.training)
        glob=n-n.flatten(2).median(2).values[:,:,None,None]
        h=self.body(torch.cat([f,ring_contrast(f),l,ring_contrast(n),glob,F.avg_pool2d(obs,4)],1))
        dw=F.softplus(self.wm(h).float());dc=F.softplus(self.clean(h).float())
        return {'delta_wm':dw,'delta_clean':dc,'edge_logit':self.edge(h).float(),
                'correction':F.softplus(self.raw_alpha_wm)*dw-F.softplus(self.raw_alpha_clean)*dc}


class TriForensicWM(nn.Module):
    def __init__(self,family,base_checkpoint=None,device='cuda',amp=True,checkpointing=True):
        super().__init__();self.family=family;self.amp=amp;self.checkpointing=checkpointing
        self.base=MessageLocalizer(family,device=device)
        if base_checkpoint is not None:
            saved=torch.load(base_checkpoint,map_location='cpu',weights_only=False)
            self.base.load_state_dict(saved['model'],strict=True)
        self.base.freeze_encoder(False)
        self.carrier=Carrier(checkpointing=checkpointing);self.refiner=Refiner()
        self.skips=nn.ModuleList([nn.Conv2d(64,128,1) for _ in range(4)])
        for layer in self.skips:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def ck(self,module,x):
        return checkpoint(module,x,use_reentrant=False) if self.training and self.checkpointing else module(x)
    def encode(self,image,size):
        enc=self.base.encoder
        with torch.autocast(image.device.type,enabled=False):
            x=image.float()
            if size!=x.shape[-1]:x=F.interpolate(x,(size,size),mode='bilinear',align_corners=False,antialias=True)
            x=x*2-1
            if self.family=='pixelseal':x=enc.net.convnext.downsample_layers[0](x)
            else:n=enc.net;x=n.maxpool(n.relu(n.bn1(n.conv1(x))))
        features=[]
        if self.family=='pixelseal':
            for i,(down,stage) in enumerate(zip(enc.net.convnext.downsample_layers,enc.net.convnext.stages)):
                if i:x=down(x)
                x=self.ck(stage,x);features.append(x)
            x=enc.net.pixel_decoder.output_upscaling(x);linear=enc.net.pixel_decoder.linear
            bits=F.conv2d(x,linear.weight[:,:,None,None],linear.bias)[:,1:]
        else:
            for key in ('layer1','layer2','layer3','layer4'):x=self.ck(getattr(n,key),x);features.append(x)
            bits=F.conv2d(x,n.fc.weight[:,:,None,None],n.fc.bias)
        return features,bits
    def global_forward(self,x,message,f,obs):
        features,z=self.encode(x,512);low,lowz=self.encode(x,self.base.encoder.native)
        lateral=[p(h)+resize(p(lo),h.shape[-1])+resize(s(f.detach()),h.shape[-1]) for p,s,h,lo in zip(self.base.proj,self.skips,features,low)]
        y=lateral[-1]
        for decoder,skip in zip(self.base.decoder,reversed(lateral[:-1])):y=self.ck(decoder,torch.cat([resize(y,skip.shape[-1]),skip],1))
        evidence=torch.cat([torch.cat([torch.tanh(m[:,:1]/4),m[:,1:]],1) for m in (resize(match_maps(z,message)),resize(match_maps(lowz,message)))],1)
        with torch.autocast(x.device.type,enabled=False):
            luma=(x*self.base.luma).sum(1,keepdim=True)
            hp=x-F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,1)
            jnd=obs[:,:1] if self.family=='pixelseal' else torch.zeros_like(luma)
            context=self.base.context[0](torch.cat([x,luma,hp,jnd],1))
        for layer in self.base.context[1:]:context=layer(context)
        return self.ck(self.base.head,torch.cat([resize(y),context,evidence],1)).float()
    def local_forward(self,x):
        with torch.autocast(x.device.type,enabled=self.amp and x.is_cuda,dtype=torch.bfloat16):
            obs=observability(x,self.base.encoder.jnd);f,l=self.carrier(x,obs)
        return f,l,obs
    def forward(self,x,message,npmap,pairs=None):
        with torch.autocast(x.device.type,enabled=self.amp and x.is_cuda,dtype=torch.bfloat16):
            f,l,obs=self.local_forward(x);base=self.global_forward(x,message,f,obs)
            out=self.refiner(f,l,npmap,obs)
            keep=(torch.rand(len(x),1,1,1,device=x.device)>=.25).float() if self.training else torch.ones(len(x),1,1,1,device=x.device)
            out.update(logit_base=base,logit_final=base*keep+out['correction'],l_loc=l,obs=F.avg_pool2d(obs,4),keep=keep)
            if pairs is not None:out['l_pair']=self.local_forward(pairs)[1]
        return out
    def parameter_groups(self):
        groups={}
        for name,p in self.named_parameters():
            if p.requires_grad:
                lr=2e-5 if name.startswith('base.') else 4e-4;wd=.01 if p.ndim>1 else 0.
                groups.setdefault((lr,wd),[]).append(p)
        return [{'params':ps,'lr':lr,'weight_decay':wd} for (lr,wd),ps in groups.items()]
