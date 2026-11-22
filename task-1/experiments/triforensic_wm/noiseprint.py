import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class Noiseprint(nn.Module):
    def __init__(self):
        super().__init__();vendor=Path(__file__).resolve().parent/'vendor';spec=importlib.util.spec_from_file_location('triforensic_dncnn',vendor/'DnCNN.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.net=module.make_net(3,[3]*17,[64]*16+[1],[False]+[True]*15+[False],['relu']*16+['linear'],[1]*17,bn_momentum=.1,padding=1)
        state=torch.load(vendor/'noiseprint.th',map_location='cpu',weights_only=False);self.net.load_state_dict(state['network'],strict=True);self.eval().requires_grad_(False)
    def train(self,mode=True):return super().train(False)
    @torch.no_grad()
    def forward(self,x):
        with torch.autocast('cuda',enabled=False):
            # Original TruFor loader divides native uint8 pixels by256, not255.
            n=self.net(x.float()*(255/256));return F.avg_pool2d(n,4)
