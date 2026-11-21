"""Save and validate the submission format from the supplied task brief."""
from pathlib import Path
import torch


def validate_submission(submission, expected_names):
    if not isinstance(submission, dict) or set(submission) != {"image_names", "masks"}:
        raise ValueError("Expected image_names and masks keys")
    names, masks = submission["image_names"], submission["masks"]
    if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
        raise ValueError("Image names must be strings")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names")
    expected_names = list(expected_names)
    if len(expected_names) != len(set(expected_names)) or set(names) != set(expected_names):
        raise ValueError("Image names must match the complete expected test set")
    if not isinstance(masks, torch.Tensor) or tuple(masks.shape) != (len(names), 128, 128):
        raise ValueError("Expected masks tensor [N,128,128]")
    if not torch.all((masks == 0) | (masks == 1)).item():
        raise ValueError("Masks must contain only 0 or 1")


def save_submission(path, names, masks, expected_names):
    submission = {"image_names": list(names), "masks": masks}
    validate_submission(submission, expected_names)
    submission["masks"] = masks.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(submission, path)
    if path.stat().st_size > 20_000_000:
        path.unlink()
        raise ValueError("Submission exceeds 20 MB")
    return path
