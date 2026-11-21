"""Trainable exact watermark extractor, native pyramid, low-scale message maps and RGB/JND context."""
import torch
from torch import nn
from torch.nn import functional as F
from .extractor import Extractor,match_maps,resize
from experiments.native_localizer.model import block

class MessageLocalizer(nn.Module):
    def __init__(self,family,device='cuda'):
        super().__init__();self.family=family;self.encoder=Extractor(family,device=device)
        self.proj=nn.ModuleList([nn.Conv2d(c,128,1) for c in self.encoder.channels])
        self.decoder=nn.ModuleList([nn.Sequential(block(256,128),block(128,128)) for _ in range(3)])
        # Original RGB, luminance, high-pass RGB and the original JND map (PS only).
        self.context=nn.Sequential(block(8,32),block(32,32,stride=2),block(32,32,stride=2),block(32,32))
        self.head=nn.Sequential(block(128+32+6,128),nn.Conv2d(128,1,1))
        self.register_buffer('luma',torch.tensor([.299,.587,.114])[None,:,None,None])
        self.frozen=False
    def freeze_encoder(self,freeze):
        self.frozen=freeze;self.encoder.net.requires_grad_(not freeze)
        if self.encoder.jnd is not None:self.encoder.jnd.requires_grad_(False)
    def forward(self,x,message):
        assert x.dtype==torch.float32 and x.shape[-2:]==(512,512)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            features,z=self.encoder(x,512)
            low_features,low_z=self.encoder(x,self.encoder.native)
        lateral=[p(f)+resize(p(lf),f.shape[-1]) for p,f,lf in zip(self.proj,features,low_features)];y=lateral[-1]
        for decoder,skip in zip(self.decoder,reversed(lateral[:-1])):
            y=decoder(torch.cat([resize(y,skip.shape[-1]),skip],1))
        y=resize(y)
        high=resize(match_maps(z,message));low=resize(match_maps(low_z,message))
        # Keep dot-logit scale bounded; cosine and soft-accuracy have known ranges.
        evidence=torch.cat([torch.cat([torch.tanh(m[:,0:1]/4),m[:,1:]],1) for m in (high,low)],1)
        luma=(x*self.luma).sum(1,keepdim=True)
        hp=x-F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,1)
        with torch.no_grad():
            jnd=self.encoder.jnd.heatmaps(x).mean(1,keepdim=True) if self.encoder.jnd is not None else torch.zeros_like(luma)
        context=self.context(torch.cat([x,luma,hp,jnd],1))
        return {'logits':self.head(torch.cat([y,context,evidence],1))[:,0]}
