"""Download official COCO train archive with resumable bounded HTTP ranges.

Bulk download avoids 50k individual high-latency requests. No embedding/training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import zipfile

import requests

from .common import atomic_json, sha256

URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    archive = a.out/"train2017.zip"
    if not archive.exists():
        response = requests.head(URL, timeout=30); response.raise_for_status()
        total = int(response.headers["Content-Length"])
        size = 32*1024*1024
        chunks = a.out/"train2017.chunks"; chunks.mkdir(exist_ok=True)
        def fetch(i):
            start, stop = i*size, min((i+1)*size,total)-1
            path = chunks/f"{i:05d}.bin"
            if path.exists() and path.stat().st_size == stop-start+1:
                return path
            for attempt in range(5):
                try:
                    r = requests.get(URL, headers={"Range":f"bytes={start}-{stop}"}, timeout=(30,90))
                    r.raise_for_status()
                    if r.status_code != 206 or len(r.content) != stop-start+1:
                        raise RuntimeError("Server returned an incomplete range")
                    temp = path.with_suffix(".part"); temp.write_bytes(r.content); temp.replace(path)
                    return path
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(4,attempt+1))
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            files = []
            for i, part in enumerate(pool.map(fetch, range((total+size-1)//size))):
                files.append(part)
                if (i+1) % 16 == 0:
                    print(f"COCO archive: {min((i+1)*size,total)}/{total} bytes", flush=True)
        temp = archive.with_suffix(".part")
        with open(temp,"wb") as out:
            for part in files:
                with open(part,"rb") as stream:
                    for data in iter(lambda:stream.read(8*1024*1024), b""):
                        out.write(data)
        if temp.stat().st_size != total:
            raise RuntimeError("Archive size mismatch")
        temp.replace(archive)
    atomic_json(a.out/"download.json", {"url": URL, "bytes": archive.stat().st_size, "sha256": sha256(archive)})
    with zipfile.ZipFile(archive) as z:
        # Archive only contains train2017/<image>.jpg. Enforce confinement explicitly.
        for item in z.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "train2017":
                raise RuntimeError("Unexpected archive path")
            target = a.out/path
            if not item.is_dir() and (not target.exists() or target.stat().st_size != item.file_size):
                z.extract(item, a.out)
    print("COCO train2017 archive extracted", flush=True)


if __name__ == "__main__":
    main()
