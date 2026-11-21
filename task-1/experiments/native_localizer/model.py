"""ConvNeXt-v1 Base + strictly local RGB stem + progressive 128-grid decoder."""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import timm


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each position; never spatial statistics."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), ChannelNorm(cout), nn.GELU())


class LocalStem(nn.Module):
    # Six stride-1 3x3s: RF=13, then stride-2 twice: RF=19, refinement: RF=27.
    receptive_field = 27
    output_stride = 4

    def __init__(self):
        super().__init__()
        self.first = block(3, 32)
        self.layers = nn.ModuleList([block(32, 32) for _ in range(5)])
        self.down = nn.Sequential(block(32, 32, 2), block(32, 32, 2), block(32, 32))
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.first(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.down(x)
        return x, self.head(x)


class NativeLocalizer(nn.Module):
    model_name = "convnext_base.fb_in22k"

    def __init__(self, pretrained=False, backbone_path=None, checkpointing=True, amp=False):
        super().__init__()
        self.encoder = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=0, global_pool="", drop_path_rate=0,
        )
        self.encoder.head = nn.Identity()
        if backbone_path:
            from safetensors.torch import load_file
            state = load_file(str(backbone_path), device="cpu")
            # Classification-only parameters are not used in dense prediction.
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
            self.encoder.load_state_dict(state, strict=True)
        assert [f["num_chs"] for f in self.encoder.feature_info] == [128, 256, 512, 1024]
        self.encoder.set_grad_checkpointing(checkpointing)
        self.local = LocalStem()
        self.lateral = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in (128, 256, 512, 1024)])
        self.decode = nn.ModuleList([
            nn.Sequential(block(512, 256), block(256, 256)) for _ in range(3)
        ])
        self.fuse = nn.Sequential(block(288, 128), nn.Conv2d(128, 1, 1))
        cfg = self.encoder.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.checkpointing, self.amp = checkpointing, amp

    def _maybe_checkpoint(self, module, x):
        if self.training and self.checkpointing:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, image, return_features=False):
        if image.dtype != torch.float32 or image.shape[1:] != (3, 512, 512):
            raise ValueError("Expected original RGB float32 [B,3,512,512] in [0,1]")
        # Override any outer autocast for all weak-signal input operations.
        with torch.autocast(image.device.type, enabled=False):
            local, local_logits = self._maybe_checkpoint(self.local, image)
            normalized = (image - self.mean.float()) / self.std.float()
            first = self.encoder.stem(normalized)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.amp and image.is_cuda else nullcontext()
        with context:
            features = []
            x = first
            for stage in self.encoder.stages:
                x = stage(x)
                features.append(x)
            expected = [(128, 128, 128), (256, 64, 64), (512, 32, 32), (1024, 16, 16)]
            if [tuple(f.shape[1:]) for f in features] != expected:
                raise RuntimeError(f"Unexpected encoder shapes: {[f.shape for f in features]}")
            laterals = [p(f) for p, f in zip(self.lateral, features)]
            x = laterals[-1]
            for decode, lateral in zip(self.decode, reversed(laterals[:-1])):
                x = F.interpolate(x, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
                x = self._maybe_checkpoint(decode, torch.cat((x, lateral), 1))
            logits = self._maybe_checkpoint(self.fuse, torch.cat((x, local), 1))
        out = {"logits": logits[:, 0].float(), "local_logits": local_logits[:, 0].float()}
        if return_features:
            out.update(first=first, local=local, shapes=[tuple(f.shape) for f in features])
        return out

    def parameter_groups(self, backbone_lr=3e-5, new_lr=1e-4, weight_decay=.01):
        groups = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = backbone_lr if name.startswith("encoder.") else new_lr
            decay = weight_decay if parameter.ndim > 1 else 0.
            groups.setdefault((lr, decay), []).append(parameter)
        return [{"params": p, "lr": lr, "weight_decay": wd} for (lr, wd), p in groups.items()]
