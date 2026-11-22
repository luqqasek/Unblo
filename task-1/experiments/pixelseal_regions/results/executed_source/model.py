"""Native pixel features and whole-rectangle evidence; no spatial/batch normalization."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def integral(x):
    return F.pad(x.float().cumsum(-2).cumsum(-1), (1, 0, 1, 0))


def region_sum(ii, boxes):
    x0,y0,x1,y1=boxes.long().unbind(-1)
    return (ii[:,y1,x1]-ii[:,y0,x1]-ii[:,y1,x0]+ii[:,y0,x0]).T


class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.a=nn.Conv2d(c,c,3,padding=1);self.b=nn.Conv2d(c,c,3,padding=1)
    def forward(self,x):return x+.1*self.b(F.gelu(self.a(x)))


class RegionPresence(nn.Module):
    stride=4
    def __init__(self):
        super().__init__()
        sys.path.insert(0,str(Path('checkpoints/videoseal').resolve()))
        from videoseal.modules.jnd import JND
        self.jnd=JND(in_channels=1,out_channels=1).requires_grad_(False)
        self.register_buffer('luma',torch.tensor([.299,.587,.114])[None,:,None,None])
        # Raw RGB retains middle/low frequencies; the learned residual filters are unrestricted.
        self.residual_filter=nn.Conv2d(3,16,5,padding=2,bias=False)
        with torch.no_grad():self.residual_filter.weight-=self.residual_filter.weight.mean((-2,-1),keepdim=True)
        self.stem=nn.Conv2d(26,32,3,padding=1)
        self.body=nn.Sequential(Residual(32),Residual(32),nn.Conv2d(32,32,3,2,1),nn.GELU(),Residual(32),nn.Conv2d(32,24,3,2,1),nn.GELU())
        # Interior mean/second moment/ring mean/contrast, observability moments and geometry.
        self.head=nn.Sequential(nn.Linear(24*4+6+6,96),nn.GELU(),nn.Linear(96,48),nn.GELU(),nn.Linear(48,2))
    def encode(self,rgb):
        with torch.autocast(rgb.device.type,enabled=False):
            x=rgb.float();y=(x*self.luma).sum(1,keepdim=True)
            chroma=torch.cat([x[:,0:1]-x[:,1:2],x[:,2:3]-x[:,1:2]],1)
            ycc=torch.cat([y,chroma],1)
            hp=(ycc-F.avg_pool2d(F.pad(ycc,(1,1,1,1),mode='reflect'),3,1))*255
            with torch.no_grad():obs=self.jnd.heatmaps(x)
            learned=self.residual_filter(ycc*255).clamp(-32,32)/8
            z=F.gelu(self.stem(torch.cat([x,ycc,hp.clamp(-16,16)/4,obs*16,learned],1)))
        z=self.body(z).float()
        o=F.avg_pool2d(torch.cat([obs*16,hp[:,:1].abs()/4, y],1),4)
        return z,o
    def tables(self,f,o):return integral(f),integral(f.square()),integral(o)
    def score(self,tables,boxes,hw):
        ii,sq,oo=tables;b=boxes.long();area=((b[:,2]-b[:,0])*(b[:,3]-b[:,1])).float().clamp_min(1)[:,None]
        outer=b+torch.tensor([-3,-3,3,3],device=b.device);outer[:,[0,2]]=outer[:,[0,2]].clamp(0,hw[1]);outer[:,[1,3]]=outer[:,[1,3]].clamp(0,hw[0]);oa=((outer[:,2]-outer[:,0])*(outer[:,3]-outer[:,1])).float()[:,None]
        mean=region_sum(ii,b)/area;second=region_sum(sq,b)/area
        ring=(region_sum(ii,outer)-region_sum(ii,b))/(oa-area).clamp_min(1)
        obs=region_sum(oo,b)/area;obsring=(region_sum(oo,outer)-region_sum(oo,b))/(oa-area).clamp_min(1)
        wh=(b[:,2:]-b[:,:2]).float();w,h=wh.unbind(1)
        geom=torch.stack([w/128,h/128,(w*h)/16384,torch.log(w/h)/3,torch.log1p(w)/5,torch.log1p(h)/5],1)
        features=torch.cat([mean,second,ring,mean-ring,obs,obsring,geom],1)
        return self.head(features)
    def forward(self,rgb,boxes,hws):
        f,o=self.encode(rgb)
        return [self.score(self.tables(fi,oi),b,hw) for fi,oi,b,hw in zip(f,o,boxes,hws)]
