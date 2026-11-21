"""Controlled RGB/SRM ablation; the existing segmentation head is imported unchanged."""
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'frozen_extractor'))
import cross_validate as cv


def srm_kernels():
    # Three fixed, zero-sum residuals, applied separately to each RGB channel.
    return torch.tensor([
        [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
        [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],
        [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
    ], dtype=torch.float32)[:,None] / torch.tensor([4.,12.,2.])[:,None,None,None]


@torch.no_grad()
def prepare_input(rgb):
    """Residuals at native 512 first; retain local residual energy when reducing to 256.

    Signed residuals use bilinear resize without antialias to avoid suppressing the
    signal twice. Same resize for RGB; no fitted preprocessing or synthetic data.
    """
    residual = F.conv2d(F.pad(rgb, (2,2,2,2), mode='reflect'),
                        srm_kernels().to(rgb.device).repeat(3,1,1,1), groups=3)
    return F.interpolate(torch.cat((rgb * 2 - 1, residual), 1), (256,256),
                         mode='bilinear', align_corners=False)


class RGBBranch(nn.Module):
    channels = [16,24,32,32]
    def __init__(self, srm):
        super().__init__()
        widths = [12 if srm else 3] + self.channels
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a,b,3,stride=1 if i==0 else 2,padding=1),
                          nn.GroupNorm(4,b),nn.ReLU())
            for i,(a,b) in enumerate(zip(widths,widths[1:]))])
        self.srm = srm

    def forward(self, x):
        x = x if self.srm else x[:,:3]
        features=[]
        for block in self.blocks:
            x=block(x)
            features.append(F.adaptive_avg_pool2d(x, (32,32)))
        return features


class Model(nn.Module):
    def __init__(self, variant, channels):
        super().__init__()
        self.variant = variant
        # Initialize the original head first: identical A/D head initialization.
        self.head = cv.model.Head(channels if variant in ('A','D') else RGBBranch.channels)
        if variant != 'A':
            self.branch = RGBBranch(variant != 'B')
        if variant == 'D':
            self.rgb_projections = nn.ModuleList([
                nn.Sequential(nn.Conv2d(c,32,1),nn.GroupNorm(8,32),nn.ReLU())
                for c in RGBBranch.channels])

    def forward(self, features, rgb):
        if self.variant == 'A':
            return self.head(features)
        branch = self.branch(rgb)
        if self.variant in ('B','C'):
            return self.head(branch)
        # Add projected features at each level; unchanged 128-channel fusion Head.
        merged = [F.interpolate(proj(f),(32,32),mode='bilinear',align_corners=False)
                  + rgb_proj(r) for proj,f,rgb_proj,r in
                  zip(self.head.projections,features,self.rgb_projections,branch)]
        return F.interpolate(self.head.fuse(torch.cat(merged,1)),(128,128),
                             mode='bilinear',align_corners=False)[:,0]
