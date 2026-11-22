import numpy as np
import torch
from .tiny_data import TinyData
from .scan import coarse_boxes

def test_tiny_pixel_provenance():
    class Old:
        rows=[{}]*2
        def load(self,i):
            c=np.full((512,512,3),100,np.uint8);w=c.copy();w[30:62,40:104]=101
            return c,w,dict(message=[0]*256,rectangle=[40,30,104,62],primary_kind='partial',regime='local_crop',clean_minority=False,minority_fraction=2048/512**2)
    d=TinyData.__new__(TinyData);d.old=Old();d.rows=[dict(index=0,rectangle=[40,30,104,62],message=[0]*256)];d.patches=[np.full((32,64,3),101,np.uint8)]
    for n in [16,24,32]:
        x,y,b,pairs,delta,regions,valid=d.batch(np.random.default_rng(42),n,True)
        assert x.shape==(n,3,512,512) and y.shape==(n,128,128)
        for i in torch.where(valid)[0]:
            expected=(x[i,0]*255-100).round()
            assert torch.allclose(delta[i],expected,atol=1e-6)
            assert torch.all(delta[i][~regions[i]]==0)
        assert len(pairs)==n//4

def test_proposals_do_not_require_gt():
    p=np.zeros((128,128),np.float32);p[15,10]=1
    boxes=coarse_boxes(p);assert boxes==coarse_boxes(p)
    assert all(0<=x0<x1<=512 and 0<=y0<y1<=512 and max(x1-x0,y1-y0)<=2*min(x1-x0,y1-y0) for x0,y0,x1,y1 in boxes)

def test_microbatch_matches_segmentation_pair_gradients():
    from .tiny_train import backward_micro
    from experiments.native_localizer.losses import segmentation,pair_loss
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3))
        def forward(self,x,b):return {'logits':x*self.weight}
    torch.manual_seed(2);x=torch.randn(32,8,8);y=(torch.rand_like(x)>.8).float();b=torch.zeros(32,2);pairs=[(i,i+1) for i in [0,2,4,6,16,18,20,22]]
    m=Toy();z=m(x,b)['logits'];(segmentation(z,y)+.1*pair_loss(z,y,pairs)).backward();expected=m.weight.grad.clone();m.zero_grad()
    backward_micro(m,x,y,b,pairs,torch.zeros_like(x),torch.zeros_like(x,dtype=torch.bool),torch.zeros(32,dtype=torch.bool),False,0,.1)
    assert torch.allclose(m.weight.grad,expected,rtol=1e-6,atol=1e-7)
