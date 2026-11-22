"""Official strict-loaded IML-ViT, raw512 adaptation with FP32 image projection."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from experiments.native_localizer.common import sha256

PIN='07dd2be0f4ea27a5c97c9fa5ffbe236733833eac'
WEIGHT_HASH='7631fe852e139d904ad96fd766a37120d703c966cb80dfad70ffae4ff426cce4'

class IMLHoles(nn.Module):
    def __init__(self,root,amp=True,checkpointing=True):
        super().__init__();root=Path(root);source=root/'research-source/iml-vit';path=root/'research-weights/iml-vit/iml-vit_checkpoint.pth'
        assert sha256(path)==WEIGHT_HASH
        sys.path.insert(0,str(source))
        from iml_vit_model import iml_vit_model
        self.net=iml_vit_model() # Keep1024 parameter shapes for strict upstream loading.
        state=torch.load(path,map_location='cpu',weights_only=True)
        if 'model' in state:state=state['model']
        self.net.load_state_dict(state,strict=True)
        self.amp=amp;self.checkpointing=checkpointing
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        # One continuous full-model adaptation; disable stochastic paths, freeze BN running stats.
        for m in self.net.modules():
            if isinstance(m,nn.Dropout):m.p=0.
            if hasattr(m,'drop_prob'):m.drop_prob=0.
        self.train()
    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'net'):
            for m in self.net.modules():
                if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        return self
    def forward(self,x,padded=False,force_fp32=False):
        assert x.dtype==torch.float32 and x.shape[-2:]==(512,512)
        if padded:x=F.pad(x,(0,512,0,512),value=0.)
        # Exact decoded pixels and the first learned projection never enter BF16.
        with torch.autocast('cuda',enabled=False):
            x=(x-self.mean)/self.std;enc=self.net.encoder_net;x=enc.patch_embed(x);x=enc.add_position_embed(x)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.amp and not force_fp32):
            for block in enc.blocks:
                x=checkpoint(block,x,use_reentrant=False) if self.training and self.checkpointing else block(x)
            features={enc._out_features[0]:x.permute(0,3,1,2)}
            # Upstream keeps Python-list aliases to registered simfp modules;
            # DataParallel shallow-copies that list and would call GPU0 weights.
            # Resolve registered modules on each replica, preserving exact order/ops.
            fpn=self.net.featurePyramid_net
            levels=[getattr(fpn,f'simfp_{level}')(features['last_feat']) for level in (2,3,4,5)]
            if fpn.top_block is not None:
                top=features.get(fpn.top_block.in_feature)
                if top is None:top=levels[fpn._out_features.index(fpn.top_block.in_feature)]
                levels.extend(fpn.top_block(top))
            assert len(levels)==len(fpn._out_features)
            logits=self.net.predict_head(levels).float()
        if padded:
            # Match upstream1024 probability path, crop only valid native512 then pool.
            logits=F.interpolate(logits,size=(1024,1024),mode='bilinear',align_corners=False)
            probability=F.avg_pool2d(logits[...,:512,:512].sigmoid(),4)
            return torch.logit(probability.clamp(1e-6,1-1e-6))[:,0]
        assert logits.shape[-2:]==(128,128)
        return logits[:,0]
