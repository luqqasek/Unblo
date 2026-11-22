"""Critical invariants: no global carrier context, exact masks, finite losses."""
import numpy as np
import torch
from .model import Carrier,ChannelGroupNorm,Bayar,ring_contrast
from .losses import objective,sym_miou

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    m=Carrier(checkpointing=False).eval();x=(.5+torch.randn(1,3,64,64)*.003).requires_grad_();obs=torch.rand(1,2,64,64)
    _,z=m(x,obs);z[0,0,8,8].backward();coords=x.grad.abs().sum(1)[0].nonzero()
    assert coords.min(0).values.tolist()==[18,18] and coords.max(0).values.tolist()==[46,46]
    k=Bayar().kernel();assert torch.equal(k[:,:,2,2],torch.full((8,3),-1.))
    assert torch.allclose(k.sum((-2,-1)),torch.zeros(8,3),atol=1e-6)
    norm=ChannelGroupNorm(64);a=torch.randn(1,64,16,16);b=a.clone();b[:,:,0,0]+=100
    assert torch.equal(norm(a)[:,:,8,8],norm(b)[:,:,8,8])
    assert torch.allclose(ring_contrast(torch.ones(2,3,16,16)),torch.zeros(2,3,16,16),atol=1e-6)
    y=torch.cat([torch.zeros(1,1,16,16),torch.ones(1,1,16,16)]);v=torch.randn_like(y,requires_grad=True)
    out={k:v for k in ('logit_final','logit_base','l_loc','edge_logit','l_pair')};out['obs']=torch.rand(2,2,16,16);out['delta_clean']=v.softplus() if hasattr(v,'softplus') else torch.nn.functional.softplus(v)
    loss,terms=objective(out,y,y);loss.backward();assert torch.isfinite(loss) and torch.isfinite(v.grad).all()
    assert sym_miou((2*y-1)*30,y)<1e-5
    # Activation recomputation changes memory use, not the model or its gradient.
    m.train();m.zero_grad();m.checkpointing=False
    _,a=m(x.detach(),obs);a.square().mean().backward();g=m.stem[0].weight.grad.clone()
    m.zero_grad();m.checkpointing=True
    _,b=m(x.detach(),obs);b.square().mean().backward()
    assert torch.equal(a,b) and torch.allclose(g,m.stem[0].weight.grad,atol=1e-6,rtol=1e-5)
    from .trustmark_refine import neighbors,valid,local_proposals
    assert any(b[3]-b[1]==67 for b in neighbors((10,10,44,78),2))
    prob=np.zeros((128,128),np.float32);prob[20:25,15:20]=.25
    assert local_proposals(prob) and not local_proposals(np.zeros_like(prob))
    assert not valid((0,0,512,512))
    print({'carrier_parameters':sum(p.numel() for p in m.parameters()),'RF':29,'critical_checks':'passed'})
if __name__=='__main__':main()
