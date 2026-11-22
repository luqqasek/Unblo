"""Technical checkpoint checks on GPU; no claim about hidden utility or score."""
import argparse
import hashlib
import json
import sys
import os
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from common import load_model, load_forget, predict

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("--report", type=Path, required=True)
a = ap.parse_args()
assert torch.cuda.is_available(), "Validate on the allocated GPU"
sd = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
assert all(not v.is_floating_point() or v.dtype == torch.float32 for v in sd.values())
assert all(torch.isfinite(v).all() for v in sd.values())
assert a.checkpoint.stat().st_size <= 450_000_000
m = load_model(a.checkpoint)
x = load_forget()[:256]
z = predict(m, x, bs=32, fp16=True)
assert z.shape == (256, 100) and torch.isfinite(z).all()
h = hashlib.sha256()
with a.checkpoint.open("rb") as f:
    for b in iter(lambda: f.read(8*1024*1024), b""):
        h.update(b)
report = {"checkpoint": str(a.checkpoint.resolve()), "sha256": h.hexdigest(), "bytes": a.checkpoint.stat().st_size,
          "strict_load": True, "finite_float32_weights": True, "fp16_finite_images": 256,
          "logit_range": [z.min().item(), z.max().item()], "torch_version": torch.__version__,
          "gpu": torch.cuda.get_device_name(), "hidden_validation_accuracy": None, "public_score": None}
a.report.parent.mkdir(parents=True, exist_ok=True)
a.report.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))

# An already-running batch can share its two idle GPUs with another task.
# After its final artifact is validated, retain the allocation only while actual
# srun steps remain live. The marker selects the artifact/job, not the wait status.
hold_path = ROOT / "outputs/allocation_hold.json"
if hold_path.exists():
    hold = json.loads(hold_path.read_text())
    job_id = os.environ.get("SLURM_JOB_ID")
    if hold.get("job_id") == job_id and hold.get("after_checkpoint") == a.checkpoint.name:
        del m, sd
        torch.cuda.empty_cache()
        while True:
            try:
                steps = subprocess.check_output(["squeue", "--steps", "-h", "-j", job_id, "-o", "%i"], text=True).split()
            except subprocess.CalledProcessError:
                time.sleep(15)
                continue
            live = [s for s in steps if s.startswith(job_id+".") and s.rsplit(".",1)[1].isdigit()]
            if not live:
                break
            print("Keeping shared allocation for live steps: " + ", ".join(live), flush=True)
            time.sleep(15)
