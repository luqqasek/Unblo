"""New TrustMark local input and the exact deployed PixelSeal global/carrier branch."""
import torch
from torch import nn
from torch.nn import functional as F


class HighPassInput(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gain = 1.0
        self.conv = nn.Conv2d(6, original.out_channels, original.kernel_size,
                              original.stride, original.padding, bias=original.bias is not None)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :3].copy_(original.weight)
            if original.bias is not None:
                self.conv.bias.copy_(original.bias)
        axis = torch.arange(-2, 3, dtype=torch.float32)
        kernel = (-axis.square() / 2).exp()
        kernel = kernel / kernel.sum()
        self.register_buffer("gaussian", (kernel[:, None] * kernel[None, :])[None, None].repeat(3, 1, 1, 1))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            blur = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.gaussian, groups=3)
            return self.conv(torch.cat((x, (x - blur) * self.gain), 1))


class NativeBranch(nn.Module):
    def __init__(self, backbone=None, checkpointing=True, drop_path=.2):
        super().__init__()
        from experiments.native_localizer.model import NativeLocalizer
        from timm.layers import DropPath
        self.net = NativeLocalizer(backbone_path=backbone, checkpointing=checkpointing, amp=True)
        self.net.local.first[0] = HighPassInput(self.net.local.first[0])
        blocks = [b for stage in self.net.encoder.stages for b in stage.blocks]
        for b, rate in zip(blocks, torch.linspace(0, drop_path, len(blocks)).tolist()):
            b.drop_path = DropPath(rate) if rate else nn.Identity()

    def forward(self, images, bits):
        return self.net(images)["logits"]


class PixelSealBranch(nn.Module):
    def __init__(self, checkpoint, device="cpu", checkpointing=True):
        super().__init__()
        from experiments.triforensic_wm.model import TriForensicWM
        # Assets are loaded through the unchanged competition wrapper in ./checkpoints.
        tri = TriForensicWM("pixelseal", device=device, checkpointing=checkpointing)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("step") != 1561 or "ema" not in saved:
            raise ValueError("Require original #3188 EMA1561 resume checkpoint")
        tri.load_state_dict(saved["ema"], strict=True)
        tri.refiner.requires_grad_(False)
        tri.carrier.head.requires_grad_(False)
        # Deployed skips detach carrier features: preserve the pretrained carrier,
        # train the entire global branch. No ineffective gradients/Adam state.
        tri.carrier.requires_grad_(False)
        self.tri = tri

    def forward(self, images, bits):
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=images.is_cuda):
            with torch.no_grad():
                features, _, observations = self.tri.local_forward(images)
            return self.tri.global_forward(images, bits, features, observations)[:, 0].float()


def build_model(family, config, device):
    if family == "trustmark":
        model = NativeBranch(config["backbone"], config.get("checkpointing", True))
        model.net.local.first[0].gain = float(config.get("highpass_gain", 1.0))
    else:
        model = PixelSealBranch(config["pixelseal_checkpoint"], device, config.get("checkpointing", True))
    return model.to(device)
