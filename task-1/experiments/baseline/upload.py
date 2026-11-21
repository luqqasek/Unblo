"""Validate and upload the prior baseline using the team's WMLOC_API_KEY."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import requests
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from watermark_localization.submission import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=Path(__file__).parent / "results/submission.pt")
    args = parser.parse_args()
    metadata = json.loads((TASK / "data/test/metadata.json").read_text(encoding="utf-8"))
    validate_submission(torch.load(args.file, map_location="cpu", weights_only=True),
                        [m["image_name"] for m in metadata])
    if args.file.stat().st_size > 20_000_000:
        parser.error("Submission exceeds 20 MB")
    # Optional local .env; never copy credentials into tracked Python files.
    local = {}
    env_file = TASK / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                name, separator, value = line.partition("=")
                if separator:
                    local[name.strip()] = value.strip().strip("\"'")
    key = os.environ.get("WMLOC_API_KEY") or local.get("WMLOC_API_KEY")
    if not key:
        parser.error("Missing WMLOC_API_KEY: set it in the environment or task-1/.env")
    base = (os.environ.get("WMLOC_BASE_URL") or local.get("WMLOC_BASE_URL")
            or "http://34.147.229.55").rstrip("/")
    with args.file.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # One POST only: an ambiguous timeout must not automatically duplicate submissions.
    with args.file.open("rb") as stream:
        response = requests.post(
            f"{base}/submit/34-watermark-localisation-image",
            headers={"X-API-Key": key},
            files={"file": (args.file.name, stream, "application/octet-stream")},
            timeout=(10, 120), allow_redirects=False,
        )
    receipt = {"file": args.file.name, "sha256": digest,
               "http_status": response.status_code, "response": response.text.replace(key, "<REDACTED>")}
    (args.file.parent / "upload_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if not 200 <= response.status_code < 300:
        raise SystemExit("Upload rejected; inspect upload_receipt.json")


if __name__ == "__main__":
    main()
