"""Check a checkpoint's format locally; this does not measure unlearning quality."""

import argparse
import json
from pathlib import Path
import sys

import torch

from _common import build_model, load_checked_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model = build_model()
    state = load_checked_state(args.checkpoint, model)
    print(json.dumps({
        "status": "PASS", "checkpoint": str(args.checkpoint.resolve()),
        "bytes": args.checkpoint.stat().st_size, "tensors": len(state),
        "strict_load": True, "floating_weights": "float32", "finite": True,
        "quality_score": "Unavailable locally: gold, control, and validation sets are hidden.",
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Checkpoint validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
