"""Parameter-level head protection and CLS preservation; not a functional guarantee."""
import torch
import torch.nn.functional as F

class HeadRowGuard:
    def __init__(self, head, present_classes):
        classes=torch.arange(head.weight.shape[0],device=head.weight.device)
        self.rows=classes[~torch.isin(classes,present_classes.to(classes.device))]
        self.params=[p for p in (head.weight,head.bias) if p is not None]
        self.original=[p.detach()[self.rows].clone() for p in self.params]
    def mask_gradients(self):
        for p in self.params:
            if p.grad is not None:p.grad[self.rows]=0
    @torch.no_grad()
    def restore(self):
        for p,ref in zip(self.params,self.original):p[self.rows]=ref
    def verified(self):
        return all(torch.equal(p.detach()[self.rows],ref) for p,ref in zip(self.params,self.original))

def forward_with_cls(model,x):
    h=model.forward_features(x)
    if h.ndim!=3:raise ValueError('Expected ViT token features')
    return model.forward_head(h),h[:,0]

def normalized_cls_loss(student,teacher):
    return (F.normalize(student.float(),dim=-1)-F.normalize(teacher.detach().float(),dim=-1)).square().sum(-1).mean()
