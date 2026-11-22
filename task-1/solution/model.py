"""Portable final PixelSeal model. No embedder downloads or original resume files."""
from pathlib import Path
import sys
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
sys.path.insert(0,str(HERE.parent))
from videoseal.modules.convnext import ConvNeXtV2
from videoseal.modules.pixel_decoder import PixelDecoder
from videoseal.modules.jnd import JND
from experiments.native_localizer.model import block
from experiments.triforensic_wm.model import TriForensicWM,Carrier,Refiner

class FinalPixelSeal(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        encoder=nn.Module();encoder.net=nn.Module()
        encoder.net.convnext=ConvNeXtV2(depths=[3,3,9,3],dims=[96,192,384,768])
        encoder.net.pixel_decoder=PixelDecoder(embed_dim=768,nbits=256,upscale_stages=[1],pixelwise=False)
        encoder.jnd=JND(in_channels=1,out_channels=1);encoder.native=256
        base=nn.Module();base.encoder=encoder
        base.proj=nn.ModuleList([nn.Conv2d(c,128,1) for c in (96,192,384,768)])
        base.decoder=nn.ModuleList([nn.Sequential(block(256,128),block(128,128)) for _ in range(3)])
        base.context=nn.Sequential(block(8,32),block(32,32,2),block(32,32,2),block(32,32))
        base.head=nn.Sequential(block(166,128),nn.Conv2d(128,1,1))
        base.register_buffer('luma',torch.tensor([.299,.587,.114])[None,:,None,None])
        tri=TriForensicWM.__new__(TriForensicWM);nn.Module.__init__(tri)
        tri.family='pixelseal';tri.amp=False;tri.checkpointing=False;tri.base=base
        tri.carrier=Carrier(checkpointing=False);tri.refiner=Refiner()
        tri.skips=nn.ModuleList([nn.Conv2d(64,128,1) for _ in range(4)])
        self.tri=tri
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['ema'],strict=True)
        self.eval().requires_grad_(False)
    def forward(self,images,bits):
        if images.dtype!=torch.float32 or images.shape[1:]!=(3,512,512):
            raise ValueError('Expected float32 RGB [B,3,512,512] in [0,1]')
        f,_,obs=self.tri.local_forward(images)
        return self.tri.global_forward(images,bits.float(),f,obs)[:,0].float()
