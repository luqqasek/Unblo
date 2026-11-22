import torch
from experiments.local_presence.verification import verified,correct_bits

def test_exact_raw_bit_threshold():
    score=float(torch.cat([torch.ones(90),torch.zeros(10)]).mean())
    assert score<.9
    assert correct_bits({'bit_accuracy':score})==90
    assert verified({'bit_accuracy':score})
    assert not verified({'bit_accuracy':float(torch.cat([torch.ones(89),torch.zeros(11)]).mean())})
from experiments.local_presence.model import Presence,Correction

def test_actual_receptive_field_and_constraints():
    torch.manual_seed(42);m=Presence();x=torch.rand(1,3,64,64,requires_grad=True)
    z,f=m(x);assert z.shape==(1,16,16)
    z[0,8,8].backward();y,xs=(x.grad[0].abs().sum(0)>1e-12).nonzero(as_tuple=True)
    assert int(y.max()-y.min()+1)==27 and int(xs.max()-xs.min()+1)==27
    assert float(m.bayar.kernel().sum((-1,-2)).abs().max())<1e-6

def test_zero_initial_correction_and_one_sided_gradient():
    for holes in [False,True]:
        m=Correction(holes);z=torch.ones(1,4,4);f=torch.ones(1,64,4,4)
        result=m(z,f);assert torch.equal(result,z)
        result.mean().backward();assert m.head.bias.grad.abs().item()>0
        with torch.no_grad():m.head.bias.fill_(2)
        if holes:assert torch.all(m(z,f)<=z)

def test_search_keeps_coordinate_bounds_and_area():
    from experiments.local_presence.tiled import starts,neighbors,valid
    assert len(starts(256,128))==3 and len(starts(160,80))==6
    assert starts(160,80)[-1]+160==512
    for box in [(0,0,34,67),(478,445,512,512),(243,270,277,337)]:
        for step in [8,4,2]:
            assert all(valid(b) for b in neighbors(box,step))
