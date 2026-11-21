"""Check focal against torchvision and class balancing including absent classes."""
import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

spec=importlib.util.spec_from_file_location('loss_screen',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LossTests(unittest.TestCase):
    def test_focal_matches_reference(self):
        z=torch.tensor([[[-10.,-.7],[.4,10.]]],requires_grad=True)
        y=torch.tensor([[[0.,1.],[0.,1.]]])
        for gamma in (1,2):
            expected=.5*sigmoid_focal_loss(z,y,alpha=-1,gamma=gamma,reduction='mean')+.5*m.ab.dice_loss(z.sigmoid(),y)
            actual=m.loss(f'focal{gamma}_dice',z,y)
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(torch.autograd.grad(actual,z,retain_graph=True)[0],torch.autograd.grad(expected,z,retain_graph=True)[0])

    def test_class_mean_and_label_symmetry(self):
        z=torch.tensor([[[2.,-1.],[-1.,-1.]]])
        y=torch.tensor([[[1.,0.],[0.,0.]]])
        b=F.binary_cross_entropy_with_logits(z,y,reduction='none')
        expected=.5*(b[y==1].mean()+b[y==0].mean())
        torch.testing.assert_close(m.class_bce(z,y),expected)
        torch.testing.assert_close(m.class_bce(-z,1-y),expected)

    def test_absent_class_and_finite_gradients(self):
        for value in (0.,1.):
            z=torch.zeros(2,3,3,requires_grad=True); y=torch.full_like(z,value)
            torch.testing.assert_close(m.class_bce(z,y),F.binary_cross_entropy_with_logits(z,y))
            for variant in m.VARIANTS:
                grad=torch.autograd.grad(m.loss(variant,z,y),z)[0]
                self.assertTrue(torch.isfinite(grad).all())


if __name__=='__main__': unittest.main()
