"""Same-content equal-area supports: signal moves, visible edits and class balance do not."""
import numpy as np
import torch
from experiments.native_localizer.common import rgb_tensor
from wmloc.utils import downsample_mask

class PositionPairs:
    def __init__(self,data):
        self.data=data;self.eligible=[[],[]]
        for b in range(2):
            for idx in data.bybin[(True,b)]:
                mask=np.unpackbits(data.pool[idx]['mask']).reshape(512,512).astype(bool)
                y,x=np.nonzero(mask)
                if x.max()-x.min()+1<=240 and y.max()-y.min()+1<=240:self.eligible[b].append(idx)
        assert all(self.eligible)
    def pair(self,rng):
        b=int(rng.choice(2,p=[.7,.3]));pool_id=int(rng.choice(self.eligible[b]));entry=self.data.pool[pool_id]
        clean,marked,row=self.data.load(entry['index']);assert row['rectangle']==[0,0,512,512]
        mask=np.unpackbits(entry['mask']).reshape(512,512).astype(bool);yy,xx=np.nonzero(mask);x0,x1=int(xx.min()),int(xx.max()+1);y0,y1=int(yy.min()),int(yy.max()+1)
        m=mask[y0:y1,x0:x1];h,w=m.shape;anchors=[]
        for _ in range(100):
            x=int(rng.integers(1,(508-w)//4+1))*4;y=int(rng.integers(1,(508-h)//4+1))*4
            if not anchors or all(x+w<=a or a+w<=x or y+h<=c or c+h<=y for a,c in anchors):anchors.append((x,y))
            if len(anchors)==2:break
        if len(anchors)<2:anchors=[(4,4),(((508-w)//4)*4,((508-h)//4)*4)]
        C=clean.copy();W=marked.copy();regions=[]
        for x,y in anchors:
            s=np.zeros((512,512),bool);s[y:y+h,x:x+w]=m;regions.append(s)
        assert not (regions[0]&regions[1]).any()
        clean_minor=bool(rng.integers(2));supports=[~m if clean_minor else m for m in regions]
        xs=[np.where(s[...,None],W,C).astype(np.uint8) for s in supports];ys=[downsample_mask(s) for s in supports]
        assert ys[0].sum()==ys[1].sum() and not np.array_equal(ys[0],ys[1])
        meta=[dict(realization_id=row['realization_id'],source_id=row['source_id'],operation='equal_area_position_pair',clean_minority=clean_minor,object_id=entry['object_id'],source_mask_bbox=[x0,y0,x1,y1],anchors=anchors,initial_rectangle=row['rectangle'],transform='support-mask integer translations only; RGB pixel coordinates unchanged; identical clean photo',view_index=j,fg_fraction=float(y.mean())) for j,y in enumerate(ys)]
        return xs,ys,meta
    def batch(self,rng):
        xs=[];ys=[];meta=[];pairs=[]
        for g in range(24):
            a,b,m=self.pair(rng) if g<8 else self.data.sample(rng,True,False)
            if len(a)==2:pairs.append((len(xs),len(xs)+1))
            xs.extend(rgb_tensor(x) for x in a);ys.extend(torch.tensor(y.copy()).float() for y in b);meta.extend(m)
        return torch.stack(xs),torch.stack(ys),pairs,meta
