"""GT-free rectangle bank; exact integral scoring targets and mask conversion."""
import numpy as np
import torch
from .model import integral,region_sum

SIZES=(3,4,5,6,8,10,12,16,20,24,32,40,48,64,80)

def candidates(h=128,w=128,step=2):
    parts=[]
    for bh in SIZES:
        for bw in SIZES:
            if bh>h or bw>w or max(bh/bw,bw/bh)>16 or not 24<=bh*bw<=820:continue
            xs=np.unique(np.r_[np.arange(0,w-bw+1,step),w-bw]);ys=np.unique(np.r_[np.arange(0,h-bh+1,step),h-bh]);xx,yy=np.meshgrid(xs,ys)
            parts.append(np.stack([xx.ravel(),yy.ravel(),xx.ravel()+bw,yy.ravel()+bh],1))
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros((0,4),np.int64)

def tight(mask):
    yy,xx=np.where(mask)
    return np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],np.int64) if len(xx) else None

def jitter(box,h,w,rng,count=32):
    if box is None:return np.zeros((0,4),np.int64)
    arr=np.tile(box,(count,1));arr[1:]+=rng.integers(-3,4,size=(count-1,4));arr[:,0]=np.clip(arr[:,0],0,w-1);arr[:,1]=np.clip(arr[:,1],0,h-1);arr[:,2]=np.clip(arr[:,2],arr[:,0]+1,w);arr[:,3]=np.clip(arr[:,3],arr[:,1]+1,h)
    return arr

def targets(mask,boxes):
    y=torch.as_tensor(mask,device=boxes.device).float();inter=region_sum(integral(y[None]),boxes)[:,0];area=((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])).float()
    return torch.stack([inter/area,inter/(area+y.sum()-inter).clamp_min(1)],1)

def draw(box,side=128):
    m=np.zeros((side,side),bool);x0,y0,x1,y1=np.asarray(box,dtype=int);m[y0:y1,x0:x1]=True;return m

def down(mask):
    h,w=mask.shape;assert h%4==w%4==0
    return (mask.reshape(h//4,4,w//4,4).sum((1,3))>=8).astype(np.uint8)
