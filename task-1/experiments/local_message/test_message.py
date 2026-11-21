"""Test masked supervision and the unchanged segmentation control."""
import importlib.util
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

spec=importlib.util.spec_from_file_location('local_message',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class MessageTests(unittest.TestCase):
    def test_background_has_zero_gradient(self):
        logits=torch.zeros(1,2,2,2,requires_grad=True)
        message=torch.tensor([[1.,0.]])
        coverage=torch.tensor([[[1.,.5],[0.,0.]]])
        loss=m.message_loss(logits,message,coverage)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[:,:,1,:]==0))
        self.assertLess(float(logits.grad[0,0,0,0]),0)
        self.assertGreater(float(logits.grad[0,1,0,0]),0)
        torch.testing.assert_close(logits.grad[:,:,0,1],.5*logits.grad[:,:,0,0])

    def test_per_image_normalization_and_empty_mask(self):
        z=torch.tensor([[[[0.,1.],[2.,3.]]],[[[-2.,1.],[2.,3.]]]],requires_grad=True)
        msg=torch.tensor([[1.],[0.]])
        mask=torch.tensor([[[1.,0.],[0.,0.]],[[1.,1.],[1.,1.]]])
        expected=.5*(F.binary_cross_entropy_with_logits(z[0,0,0,0],msg[0,0])+F.binary_cross_entropy_with_logits(z[1],msg[1,:,None,None].expand_as(z[1])))
        torch.testing.assert_close(m.message_loss(z,msg,mask),expected)
        empty=m.message_loss(z,msg,torch.zeros_like(mask))
        self.assertEqual(float(empty),0)
        self.assertTrue(torch.all(torch.autograd.grad(empty,z)[0]==0))

    def test_identical_segmentation_initialization(self):
        torch.manual_seed(42); base=m.cv.model.Head([8,16,32,64])
        torch.manual_seed(42); joint=m.JointHead([8,16,32,64],100)
        features=[torch.randn(2,c,n,n) for c,n in zip([8,16,32,64],[32,32,16,8])]
        for name,value in base.state_dict().items():
            torch.testing.assert_close(joint.state_dict()[name],value,rtol=0,atol=0)
        torch.testing.assert_close(base(features),joint(features),rtol=0,atol=0)
        bits=joint.joint(features)[1]
        m.message_loss(bits,torch.ones(2,100),torch.ones(2,32,32)).backward()
        self.assertGreater(float(joint.projections[0][0].weight.grad.abs().sum()),0)
        self.assertIsNone(joint.fuse[-1].weight.grad)


if __name__=='__main__':
    torch.set_num_threads(4); unittest.main()
