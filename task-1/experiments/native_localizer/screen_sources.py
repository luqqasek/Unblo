"""Recheck downloaded COCO sources using locally computed competition fingerprints."""
import argparse
import json
from pathlib import Path

from .common import jsonl
from .prepare_sources import hash_distance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--fingerprints", type=Path, required=True)
    a = p.parse_args()
    fingerprints = json.loads(a.fingerprints.read_text())
    if len(fingerprints) != 510 or len({r[0] for r in fingerprints}) != 510:
        raise ValueError("Require all unique 510 training fingerprints")
    rows = jsonl(a.sources)
    for row in rows:
        if "phash" not in row or "sha256" not in row:
            continue
        hits = [name for name, h, ph in fingerprints
                if h == row["sha256"] or hash_distance(ph, row["phash"]) <= 4]
        row["competition_duplicate_candidates"] = hits
        row["screened_competition_n"] = 510
        row["excluded"] = bool(hits or row.get("cross_partition_duplicate_candidates") or row.get("error"))
    temporary = a.sources.with_suffix(".screened.tmp")
    temporary.write_text("".join(json.dumps(r)+"\n" for r in rows))
    temporary.replace(a.sources)
    print(json.dumps({"source_manifest": str(a.sources), "rows": len(rows),
                      "accepted": sum(not r["excluded"] for r in rows), "screened_against": 510}), flush=True)


if __name__ == "__main__":
    main()
