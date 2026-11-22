import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from model_protection import HeadRowGuard,normalized_cls_loss
class Tests(unittest.TestCase):
 def test_adam_head_guard(self):
  torch.manual_seed(3);h=torch.nn.Linear(5,4);orig=h.weight.detach().clone();g=HeadRowGuard(h,torch.tensor([1,3]));opt=torch.optim.AdamW(h.parameters(),lr=.1,weight_decay=.1)
  for _ in range(4):
   opt.zero_grad();h(torch.randn(8,5)).square().sum().backward();g.mask_gradients();opt.step();g.restore()
  self.assertTrue(g.verified());self.assertFalse(torch.equal(h.weight[1],orig[1]));self.assertEqual(g.rows.tolist(),[0,2])
 def test_feature_gradient_and_scale(self):
  a=torch.randn(8,5,requires_grad=True);b=torch.randn(8,5,requires_grad=True);l=normalized_cls_loss(a,b);l.backward();self.assertIsNotNone(a.grad);self.assertIsNone(b.grad);self.assertLess(float(normalized_cls_loss(a,a*3)),1e-10)
if __name__=='__main__':unittest.main()
