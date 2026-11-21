"""Check downloaded data and run one local CPU forward pass (no remote API)."""
import json
from pathlib import Path

import numpy as np
import torch

from official import task_template as template
from workflow import validate

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    template.DEVICE = "cpu"
    report = validate(ROOT / "outputs/baseline.pt")
    with np.load(ROOT / "data/forget_images.npz", allow_pickle=False) as data:
        images = data["images"]
        assert images.shape == (3500, 224, 224, 3), images.shape
        assert images.dtype == np.uint8, images.dtype
        report["data"] = {"keys": data.files, "shape": list(images.shape), "dtype": str(images.dtype)}
        sample = torch.from_numpy(images[:1].copy())
    model = template.load_model(ROOT / "outputs/baseline.pt")
    logits = template.predict(model, sample, batch_size=1)
    assert logits.shape == (1, 100)
    assert torch.isfinite(logits).all()
    report["smoke_inference"] = {"device": "cpu", "shape": list(logits.shape),
                               "predicted_class": logits.argmax(1).item(), "finite": True}
    report["limitations"] = "No labels or hidden gold/control/validation sets; local accuracy and contest score unavailable."
    (ROOT / "outputs/local_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
