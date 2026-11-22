"""Exercise the local stack; optionally check the actual challenge artifacts."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys

import numpy as np
import torch

from _common import DATA, build_model, check_images, choose_device, load_checked_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--require-assets", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.manual_seed(0)
    device = choose_device(args.device)
    model = build_model().eval()
    state = model.state_dict()
    assert "blocks.0.ls1.gamma" in state, "Required LayerScale parameters are missing."
    artifacts = {name: (DATA / name).is_file() for name in (
        "model.pt", "forget_images.npz", "task_template.py", "submission.py"
    )}
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "timm", "numpy", "huggingface-hub"
        )},
        "device": str(device),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "challenge_files": artifacts,
        "weights_source": "random initialization (environment smoke check only)",
        "input_source": "synthetic zeros",
    }
    inputs = torch.zeros(1, 3, 224, 224, device=device)
    if args.require_assets:
        missing = [name for name, present in artifacts.items() if not present]
        if missing:
            raise ValueError(
                f"Missing: {', '.join(missing)}. Run .venv/bin/python scripts/download_assets.py"
            )
        load_checked_state(DATA / "model.pt", model)
        report["forget_images"] = check_images(DATA / "forget_images.npz")
        report["weights_source"] = str(DATA / "model.pt")
        spec = importlib.util.spec_from_file_location("official_task_template", DATA / "task_template.py")
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load the official preprocessing helper.")
        template = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(template)
        template.DEVICE = str(device)
        with np.load(DATA / "forget_images.npz", allow_pickle=False) as archive:
            sample = torch.from_numpy(archive["images"][:2].copy())
        inputs = template.preprocess(sample)
        assert inputs.dtype == torch.float32 and tuple(inputs.shape) == (2, 3, 224, 224)
        assert inputs.min().item() >= -1 and inputs.max().item() <= 1
        report["input_source"] = "first two forget images, official task_template.preprocess"
        report["preprocessing"] = "(pixels / 255 - 0.5) / 0.5; NHWC -> NCHW"
    model = model.to(device)
    with torch.inference_mode():
        logits = model(inputs)
    assert tuple(logits.shape) == (len(inputs), 100), logits.shape
    assert torch.isfinite(logits).all().item(), "Forward pass returned non-finite logits."
    report["forward_shape"] = list(logits.shape)
    report["predicted_classes"] = logits.argmax(dim=1).cpu().tolist()
    report["status"] = "PASS"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, AssertionError) as exc:
        print(f"Environment check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
