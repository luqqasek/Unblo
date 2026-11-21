"""Download pinned official files, checking size and published LFS SHA-256."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "beac43807fbc29a32c1037ea1d8ee2ed6548595a"
REPO = "https://huggingface.co/datasets/SprintML/unlearning"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    url = f"https://huggingface.co/api/datasets/SprintML/unlearning/tree/{REVISION}?recursive=true"
    entries = json.load(urllib.request.urlopen(url, timeout=60))
    manifest = []
    for entry in entries:
        if entry["type"] != "file":
            continue
        name = entry["path"]
        folder = "data" if name in ("model.pt", "forget_images.npz") else "official"
        path = ROOT / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        source = f"{REPO}/resolve/{REVISION}/{name}"
        expected = entry.get("lfs", {}).get("oid")
        if not (path.exists() and path.stat().st_size == entry["size"] and
                (not expected or sha256(path) == expected)):
            print(f"Downloading {name}: {entry['size'] / 1e6:.1f} MB", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(source, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
            if temporary.stat().st_size != entry["size"]:
                raise ValueError(f"Size mismatch: {name}")
            if expected and sha256(temporary) != expected:
                raise ValueError(f"SHA-256 mismatch: {name}")
            temporary.replace(path)
        manifest.append({"path": path.relative_to(ROOT).as_posix(), "url": source,
                         "revision": REVISION, "size": path.stat().st_size, "sha256": sha256(path)})
        print(f"Verified {name}", flush=True)
    existing = ROOT / "official/manifest.json"
    if existing.exists():
        manifest.extend(item for item in json.loads(existing.read_text(encoding="utf-8"))
                        if item["path"].startswith("official/hackathon/"))
    (ROOT / "official/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
