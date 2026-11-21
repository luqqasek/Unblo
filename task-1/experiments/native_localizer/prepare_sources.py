"""COCO source-level splits BEFORE embedding. No competition test images are read.

Run on a login node for networking (bounded downloads); not training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from .common import ROOT, atomic_json, sha256

ANNOTATIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"


def perceptual_hash(image):
    # DCT pHash; supplementary screen, not a guarantee against transformed duplicates.
    from scipy.fft import dctn
    x = np.asarray(image.convert("L").resize((32, 32)), np.float32)
    values = dctn(x, norm="ortho")[:8, :8].ravel()[1:]
    return np.packbits(values > np.median(values)).tobytes().hex()


def hash_distance(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--train-sources", type=int, default=50000)
    p.add_argument("--dev-sources", type=int, default=2000)
    p.add_argument("--locked-sources", type=int, default=2000)
    p.add_argument("--download-limit", type=int, default=5000,
                   help="Per partition; source partitions are still fixed over the full source set")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--images-directory", type=Path, help="Optional extracted official train2017 image directory")
    p.add_argument("--competition-train", type=Path, default=ROOT/"data/train/images")
    p.add_argument("--competition-fingerprints", type=Path,
                   help="Precomputed SHA256/pHash of all 510 train images; useful before remote image transfer")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ann = a.annotations or a.out/"instances_train2017.json"
    if not ann.exists():
        archive = a.out/"annotations_trainval2017.zip"
        if not archive.exists():
            urllib.request.urlretrieve(ANNOTATIONS_URL, archive.with_suffix(".part"))
            archive.with_suffix(".part").replace(archive)
        with zipfile.ZipFile(archive) as z:
            ann.write_bytes(z.read("annotations/instances_train2017.json"))
    coco = json.loads(ann.read_text())
    objects = {}
    for obj in coco["annotations"]:
        if not obj.get("iscrowd", 0) and obj["area"] >= 64:
            objects.setdefault(obj["image_id"], []).append(obj)
    # Stable source ordering independent of filesystem and of downloads.
    images = sorted(coco["images"], key=lambda r: hashlib.sha256(f"42:coco2017:{r['id']}".encode()).hexdigest())
    need = a.train_sources+a.dev_sources+a.locked_sources
    if len(images) < need:
        raise ValueError("Not enough distinct COCO source IDs")
    # Near-duplicate source siblings share their split. Exclude cross-split siblings
    # at download time rather than moving them after any generated data exists.
    competition = (json.loads(a.competition_fingerprints.read_text()) if a.competition_fingerprints else
                   [(f.stem, sha256(f), perceptual_hash(Image.open(f)))
                    for f in sorted(a.competition_train.glob("*.png"))])
    if len(competition) != 510:
        raise RuntimeError(f"Expected all 510 competition TRAIN images for duplicate screening, found {len(competition)}")
    if len({r[0] for r in competition}) != 510:
        raise ValueError("Duplicate competition fingerprint identities")
    manifests = {"train": [], "dev": [], "locked": []}
    for i, record in enumerate(images[:need]):
        split = "locked" if i < a.locked_sources else "dev" if i < a.locked_sources+a.dev_sources else "train"
        manifests[split].append({
            "source_id": f"coco2017:{record['id']}", "coco_id": record["id"],
            "partition": split, "url": record["coco_url"].replace(
                "http://images.cocodataset.org/", "https://s3.amazonaws.com/images.cocodataset.org/"
            ).replace("https://images.cocodataset.org/", "https://s3.amazonaws.com/images.cocodataset.org/"),
            "width": record["width"], "height": record["height"],
            "license_id": record["license"], "objects": objects.get(record["id"], []),
            "path": str(((a.images_directory or a.out/"images")/record["file_name"]).resolve()),
        })
    splitfile = a.out/"source_partitions.json"
    assignment = {k: [r["source_id"] for r in v] for k, v in manifests.items()}
    if splitfile.exists() and json.loads(splitfile.read_text()) != assignment:
        raise RuntimeError("Source partitions already exist and differ; use a new output directory")
    atomic_json(splitfile, assignment)
    atomic_json(a.out/"source_provenance.json", {
        "dataset": "COCO train2017", "annotations_url": ANNOTATIONS_URL,
        "annotations_sha256": sha256(ann), "licenses": coco["licenses"],
        "competition_train_sha256": {n: h for n, h, _ in competition},
        "duplicate_screen": "exact bytes plus DCT pHash Hamming <=4; not exhaustive semantic deduplication",
    })
    def download(r):
        path = Path(r["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not path.exists():
                urllib.request.urlretrieve(r["url"], path.with_suffix(".part"))
                path.with_suffix(".part").replace(path)
            with Image.open(path) as image:
                r = dict(r, sha256=sha256(path), phash=perceptual_hash(image))
            hits = [name for name, h, ph in competition
                    if h == r["sha256"] or hash_distance(ph, r["phash"]) <= 4]
            return dict(r, excluded=bool(hits), competition_duplicate_candidates=hits,
                        screened_competition_n=len(competition))
        except Exception as e:
            return dict(r, excluded=True, error=str(e))
    # Screen dev/locked first; their images will not be used as generation donors.
    seen = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for split in ("locked", "dev", "train"):
            rows = []
            for r in pool.map(download, manifests[split][:a.download_limit]):
                hits = [other["source_id"] for other in seen
                        if other["partition"] != split and
                        (other["sha256"] == r.get("sha256") or
                         hash_distance(other["phash"], r.get("phash", other["phash"])) <= 4)]
                if hits:
                    r.update(excluded=True, cross_partition_duplicate_candidates=hits)
                if not r["excluded"]:
                    seen.append(r)
                rows.append(r)
                if len(rows) % 100 == 0:
                    print(json.dumps({"partition": split, "downloaded": len(rows),
                                      "accepted": sum(not x["excluded"] for x in rows)}), flush=True)
            # Atomic replacement allows extending the available source count safely.
            out = a.out/f"{split}.jsonl"
            temp = out.with_suffix(".tmp")
            temp.write_text("".join(json.dumps(r)+"\n" for r in rows))
            temp.replace(out)


if __name__ == "__main__":
    main()
