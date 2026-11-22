"""Compact decoder exposing all available inference signals and privileged training targets."""
import torch
from torch import nn
from torch.nn import functional as F

def block(ci,co,stride=1):
    return nn.Sequential(nn.Conv2d(ci,co,3,stride,1),nn.GroupNorm(4,co),nn.SiLU())

class AllInformation(nn.Module):
    def __init__(self,channels,nbits,bit_channels,scalar_channels,full):
        super().__init__();self.full=full
        self.projections=nn.ModuleList([nn.Conv2d(c,16,1) for c in channels])
        self.raw=nn.Sequential(block(7,16),block(16,24,2),block(24,32,2),block(32,32))
        self.local_head=nn.Conv2d(32,1,1)
        self.scalar=nn.Sequential(nn.Conv2d(scalar_channels,16,1),nn.SiLU())
        self.bits=nn.Sequential(nn.Conv2d(bit_channels,16,1),nn.SiLU())
        self.message=nn.Sequential(nn.Linear(nbits,32),nn.SiLU(),nn.Linear(32,16))
        self.decoder=nn.Sequential(block(64+32+16+16+16,64),block(64,48),block(48,32))
        self.mask=nn.Conv2d(32,1,1)
        self.subpixel=nn.Sequential(nn.Conv2d(32,16,1),nn.PixelShuffle(4))
        self.geometry=nn.Conv2d(32,4,1)
        self.energy=nn.Conv2d(32,1,1)
        self.meta=nn.Linear(32,2)
    def raw_forward(self,rgb):
        x=rgb.float()/255
        y=(x*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        hp=(x-F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,1))*255
        return self.raw(torch.cat([x,y,hp.clamp(-8,8)],1))
    def forward(self,batch,wrong_message=False):
        raw=self.raw_forward(batch['rgb'])
        features=[F.interpolate(proj(x.float()),(128,128),mode='bilinear',align_corners=False) for proj,x in zip(self.projections,batch['features'])]
        scalar_values=batch['scalar'].float()
        if wrong_message:
            scalar_values=scalar_values.clone()
            scalar_values[:,0::3].neg_()
            scalar_values[:,1::3]=1-scalar_values[:,1::3]
            scalar_values[:,2::3].neg_()
        scalar=F.interpolate(self.scalar(scalar_values),(128,128),mode='bilinear',align_corners=False)
        if self.full:
            bit_values=batch['bits'].float();msg=batch['message'].float()*2-1
            if wrong_message:
                # Deterministic bitwise complement changes every supplied query bit.
                bit_values=-bit_values;msg=-msg
            bits=F.interpolate(self.bits(bit_values),(128,128),mode='bilinear',align_corners=False)
            message=self.message(msg)[:,:,None,None].expand(-1,-1,128,128)
        else:
            bits=torch.zeros_like(scalar);message=torch.zeros_like(scalar)
        z=self.decoder(torch.cat([*features,raw,scalar,bits,message],1))
        return {'logits':self.mask(z)[:,0], 'mask512':self.subpixel(z)[:,0],
                'geometry':self.geometry(z),'energy':self.energy(z)[:,0],
                'meta':self.meta(z.mean((2,3))),'local':self.local_head(raw)[:,0]}

def segmentation(z,y):
    y=y.float();z=z.float();p=z.sigmoid()
    bce=F.binary_cross_entropy_with_logits(z,y,reduction='none').mean((-2,-1))
    losses=[]
    for a,b in ((p,y),(1-p,1-y)):
        losses.append(1-(2*(a*b).sum((-2,-1))+1e-5)/(a.sum((-2,-1))+b.sum((-2,-1))+1e-5))
    return (.5*bce+.25*(losses[0]+losses[1])).mean()

def objective(model,batch,out):
    y=batch['mask128'].float();primary=segmentation(out['logits'],y)
    terms={'primary128':primary}
    if model.full:
        terms['pixel512']=.2*segmentation(out['mask512'],batch['mask512'])
        g=batch['geometry'].float()
        terms['edge']=.1*segmentation(out['geometry'][:,0],g[:,0])
        terms['distance']=.1*F.smooth_l1_loss(out['geometry'][:,1].tanh(),g[:,1])
        terms['components']=.05*F.binary_cross_entropy_with_logits(out['geometry'][:,2:],g[:,2:])
        terms['regime']=.1*F.binary_cross_entropy_with_logits(out['meta'][:,0],batch['full_watermark'].float())
        terms['coverage']=.1*F.mse_loss(out['meta'][:,1].sigmoid(),y.mean((1,2)))
        valid=batch['synthetic'].bool()
        if valid.any():
            # Residual ENERGY, not signed residual L1; unavailable for real photos.
            e=batch['energy'][valid].float()
            terms['energy']=.05*F.smooth_l1_loss(F.softplus(out['energy'][valid]),torch.log1p(e))
            clean_features=model.raw_forward(batch['clean'][valid])
            negative=model.local_head(clean_features)[:,0]
            terms['clean_pair']=.05*F.binary_cross_entropy_with_logits(negative,torch.zeros_like(negative))
            region=y[valid]>0
            if region.any():terms['pair_ranking']=.05*F.softplus(1-out['local'][valid][region]+negative[region]).mean()
        terms['local']=.1*segmentation(out['local'],y)
    return sum(terms.values()),terms
