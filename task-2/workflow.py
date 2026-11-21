"""Offline baseline and CSV-only submission. Never queries model endpoints."""
import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENDPOINTS = {f"ep-{i:02d}" for i in range(1, 34)}
DEFAULT = ROOT / "outputs/random_seed_2026.csv"
URL = "http://34.147.229.55/submit/33-model-equality"


def validate(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if rows and rows[0] == ["endpoint", "family", "model"]:
        rows = rows[1:]
    if len(rows) != 33 or any(len(row) != 3 for row in rows):
        raise ValueError("Expected exactly 33 rows with three columns")
    if {row[0] for row in rows} != ENDPOINTS:
        raise ValueError("Expected each endpoint ep-01 through ep-33 exactly once")
    families = {}
    for endpoint, family, model in rows:
        if any(not label.strip() or any(c in label for c in ',\r\n') for label in (family, model)):
            raise ValueError("Invalid family/model label")
        if model in families and families[model] != family:
            raise ValueError("A model cannot belong to multiple families")
        families[model] = family
    return {"endpoints": len(rows), "families": len(set(r[1] for r in rows)), "models": len(families),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def api_key():
    key = os.environ.get("MODEL_EQUALITY_API_KEY", "")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "MODEL_EQUALITY_API_KEY":
                key = value.strip().strip("\"'")
    if not key or key == "YOUR_API_KEY_HERE":
        raise ValueError("Set MODEL_EQUALITY_API_KEY in task-2/.env or environment")
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
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.command == "generate":
        rng = random.Random(args.seed)
        args.file.parent.mkdir(parents=True, exist_ok=True)
        with args.file.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("endpoint", "family", "model"))
            for endpoint in sorted(ENDPOINTS):
                family = chr(ord("A") + rng.randrange(3))
                writer.writerow((endpoint, family, f"{family}{rng.randrange(3) + 1}"))
    summary = validate(args.file)
    print(json.dumps(summary, indent=2))
    return submit(args.file, summary) if args.command == "submit" else 0


if __name__ == "__main__":
    raise SystemExit(main())
