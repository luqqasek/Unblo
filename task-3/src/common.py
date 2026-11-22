"""Shared helpers: model builder, data loading, server-style preprocessing and prediction."""
from pathlib import Path

import numpy as np
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "unlearning"
OUT = ROOT / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def build_model():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)


def load_model(path=DATA / "model.pt", device=DEVICE):
    m = build_model()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.to(device).eval()


def load_forget():
    with np.load(DATA / "forget_images.npz", allow_pickle=False) as d:
        return torch.from_numpy(d["images"].copy())  # uint8 [N,224,224,3]


def preprocess(u8, device=DEVICE):
    x = u8.to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


@torch.no_grad()
def predict(model, u8, bs=100, device=DEVICE, fp16=False):
    model.eval()
    out = []
    for i in range(0, len(u8), bs):
        x = preprocess(u8[i:i + bs], device)
        if fp16 or device == "cuda":   # the server scores under fp16 autocast
            with torch.autocast(device_type=device, dtype=torch.float16):
                out.append(model(x).float().cpu())
        else:
            out.append(model(x).float().cpu())
    return torch.cat(out)
