"""Architecture and file checks specified in Unlearning.pdf, pages 3-4."""

from pathlib import Path

import numpy as np
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "unlearning"
MAX_BYTES = 450_000_000  # Conservative decimal interpretation of the 450 MB limit.


def build_model():
    return timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4
    )


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this Python process.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is not available in this Python process.")
    return torch.device(name)


def load_checked_state(path, model):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing checkpoint: {path}")
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Checkpoint must have a .pt or .pth extension.")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Checkpoint exceeds {MAX_BYTES:,} bytes.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must contain a nonempty, bare state dict.")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Every state dict entry must be a named tensor.")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"{name}: expected float32, got {value.dtype}.")
        if value.is_complex() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name}: complex or non-finite values are not allowed.")
    model.load_state_dict(state, strict=True)
    return state


def check_images(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing forget set: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "images" not in archive:
            raise ValueError("Forget archive is missing the 'images' key.")
        images = archive["images"]
        expected = (3500, 224, 224, 3)
        if images.shape != expected or images.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 {expected}, got {images.dtype} {images.shape}."
            )
        return {"shape": list(images.shape), "dtype": str(images.dtype)}
