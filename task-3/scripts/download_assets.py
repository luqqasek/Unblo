"""Download only this task's official files and the shared setup README."""

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import HfApi, hf_hub_download

SOURCES = {
    "SprintML/unlearning": (
        "unlearning", ("model.pt", "forget_images.npz", "task_template.py", "submission.py")
    ),
    "SprintML/hackathon": ("hackathon", ("README.md",)),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest_path = ROOT / "data" / "sources.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "sources": []}
    api = HfApi()
    for repo, (folder, filenames) in SOURCES.items():
        print(f"Resolving {repo} ...", flush=True)
        revision = api.dataset_info(repo, timeout=15).sha
        if not revision:
            raise ValueError(f"No immutable revision returned for {repo}.")
        source = {"repo": repo, "revision": revision, "files": []}
        for filename in filenames:
            print(f"Downloading {filename} ...", flush=True)
            path = Path(hf_hub_download(
                repo_id=repo, repo_type="dataset", filename=filename,
                revision=revision, local_dir=ROOT / "data" / folder,
            ))
            source["files"].append({
                "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        manifest["sources"].append(source)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Downloads complete. Next: .venv/bin/python scripts/check_environment.py --require-assets")
    print("Read data/hackathon/README.md and compare the official task_template.py before experiments.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        print(
            "Check network access and retry the same command. For a gated repository, "
            "authenticate with `.venv/bin/hf auth login` or set HF_TOKEN locally. Never commit tokens.",
            file=sys.stderr,
        )
        raise SystemExit(1)
