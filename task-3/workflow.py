"""Unchanged model baseline, strict validation and weights-only submission."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "outputs/baseline.pt"
URL = "http://34.147.229.55/submit/31-unlearning"


def validate(path):
    import torch
    from official.task_template import build_model
    torch.set_num_threads(4)
    if path.suffix.lower() not in (".pt", ".pth") or path.stat().st_size > 450_000_000:
        raise ValueError("Expected .pt/.pth state dict at most 450 MB")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected nonempty state dict")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Not a tensor: {name}")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"Not float32: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite values: {name}")
    model = build_model()
    model.load_state_dict(state, strict=True)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"tensors": len(state), "parameters": sum(p.numel() for p in model.parameters()),
            "bytes": path.stat().st_size, "sha256": digest, "strict_load": True}


def api_key():
    key = os.environ.get("UNLEARNING_API_KEY", "")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "UNLEARNING_API_KEY":
                key = value.strip().strip("\"'")
    if not key or key == "YOUR_API_KEY_HERE":
        raise ValueError("Set UNLEARNING_API_KEY in task-3/.env or environment")
    return key


def submit(path, summary):
    import requests

    key = api_key()
    folder = ROOT / "outputs"
    folder.mkdir(exist_ok=True)
    state = folder / "last_submission_attempt.json"
    now = datetime.now(timezone.utc)
    if state.exists():
        previous = json.loads(state.read_text(encoding="utf-8"))
        remaining = 1800 - (now - datetime.fromisoformat(previous["started_at"])).total_seconds()
        if remaining > 0:
            raise ValueError(f"Submission cooldown: {remaining:.0f} seconds remaining")
    record = {"started_at": now.isoformat(), "file": path.name, **summary, "status": "started"}
    receipt = folder / ("submission_" + now.strftime("%Y%m%dT%H%M%SZ") + ".json")

    def save():
        content = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        state.write_text(content, encoding="utf-8")
        receipt.write_text(content, encoding="utf-8")

    save()  # Also retain cooldown on errors and uncertain network outcomes.
    try:
        with path.open("rb") as handle:
            reply = requests.post(URL, headers={"X-API-Key": key},
                                  files={"file": (path.name, handle)}, timeout=(30, 600),
                                  allow_redirects=False)
        record["http_status"] = reply.status_code
        try:
            body = reply.json()
        except ValueError:
            body = {"message": reply.text[:1000]}
        record["response"] = body
        record["status"] = "success" if reply.status_code == 200 and body.get("status") == "success" else "rejected"
    except requests.RequestException as exc:
        record["status"] = "network_error"
        record["error"] = type(exc).__name__
    save()
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"Receipt: {receipt}")
    return 0 if record["status"] == "success" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate", "submit"))
    parser.add_argument("--file", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.command == "generate":
        import shutil
        source = ROOT / "data/model.pt"
        validate(source)
        args.file.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, args.file.open("xb") as dest:
            shutil.copyfileobj(src, dest)
    summary = validate(args.file)
    print(json.dumps(summary, indent=2))
    return submit(args.file, summary) if args.command == "submit" else 0


if __name__ == "__main__":
    raise SystemExit(main())
