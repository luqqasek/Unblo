import unittest
import importlib.util
from pathlib import Path
import torch
from prepare import spatial_evidence, cv
spec=importlib.util.spec_from_file_location('evidence_run',Path(__file__).with_name('run.py'))
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
EvidenceHead=module.EvidenceHead


class EvidenceTests(unittest.TestCase):
    def test_message_complement(self):
        z=torch.tensor([[[[2.,-1.]],[[1.,3.]]]])
        msg=torch.tensor([[1.,0.]])
        a=spatial_evidence(z,msg); b=spatial_evidence(z,1-msg)
        torch.testing.assert_close(a[:,0]+b[:,0],torch.ones_like(a[:,0]))
        torch.testing.assert_close(a[:,2],-b[:,2])
        torch.testing.assert_close(a[:,3]+b[:,3],torch.ones_like(a[:,3]))

    def test_zero_residual_matches_baseline(self):
        torch.manual_seed(42); base=cv.model.Head([8,16,32,64])
        torch.manual_seed(42); head=EvidenceHead([8,16,32,64],'residual')
        features=[torch.randn(2,c,n,n) for c,n in zip([8,16,32,64],[32,32,16,8])]
        torch.testing.assert_close(base(features),head(features+[torch.rand(2,4,8,8)]),atol=0,rtol=0)
        head(features+[torch.rand(2,4,8,8)]).mean().backward()
        self.assertGreater(float(head.correction[-1].weight.grad.abs().sum()),0)


if __name__=='__main__':
    torch.set_num_threads(4); unittest.main()
