import torch
from torch import nn
from torch.nn import functional as F
from wmloc.watermarks import load_watermark

class Extractor(nn.Module):
    def __init__(self,family,models='checkpoints',device='cuda'):
        super().__init__(); self.family=family
        wm=load_watermark(family,models_dir=models,device=device)
        self.net=wm.det if family=='pixelseal' else wm.net
        self.jnd=wm.model.attenuation if family=='pixelseal' else None
        if self.jnd is not None:self.jnd.requires_grad_(False)
        self.native=256 if family=='pixelseal' else wm.resolution
        self.channels=[96,192,384,768] if family=='pixelseal' else [256,512,1024,2048]
        self.nbits=wm.nbits
    def train(self,mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        return self
    def forward(self,image,size=None):
        if size is not None and image.shape[-1]!=size:
            image=F.interpolate(image,size=(size,size),mode='bilinear',align_corners=False,antialias=True)
        x=image*2-1; features=[]
        if self.family=='pixelseal':
            for down,stage in zip(self.net.convnext.downsample_layers,self.net.convnext.stages):
                x=stage(down(x)); features.append(x)
            x=self.net.pixel_decoder.output_upscaling(x)
            linear=self.net.pixel_decoder.linear
            bits=F.conv2d(x,linear.weight[:,:,None,None],linear.bias)[:,1:]
        else:
            n=self.net;x=n.maxpool(n.relu(n.bn1(n.conv1(x))))
            for key in ('layer1','layer2','layer3','layer4'):
                x=getattr(n,key)(x);features.append(x)
            bits=F.conv2d(x,n.fc.weight[:,:,None,None],n.fc.bias)
        return features,bits

def match_maps(bits,message):
    sign=message.float()*2-1; z=bits.float(); s=sign[:,:,None,None]
    dot=(z*s).mean(1,keepdim=True)
    cosine=(z*s).sum(1,keepdim=True)/(z.square().sum(1,keepdim=True).sqrt()*z.shape[1]**.5).clamp_min(1e-8)
    acc=(z*s).sigmoid().mean(1,keepdim=True)
    return torch.cat([dot,cosine,acc],1)

def resize(x,size=128):return F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)
