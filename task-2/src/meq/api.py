"""Client for the hackathon server: budgets and resumable, append-only querying.

Every response from a scored endpoint costs one of 100 queries that never come back, so this module is built to
never lose or double-spend one: answers are appended and flushed as they arrive, finished (endpoint, id, repeat)
triples are skipped on restart, and the full server reply is kept next to the extracted text.
"""
import csv
import hashlib
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from . import config


def session() -> requests.Session:
    s = requests.Session()
    s.headers["X-API-Key"] = config.api_key()
    return s


def budget(s: requests.Session | None = None) -> dict:
    """{"ep-01": {"limit": 100, "used": 3, "left": 97}, ...} for every endpoint the server lists."""
    reply = (s or session()).get(f"{config.BASE_URL}/models", timeout=60)
    if reply.status_code != 200:
        raise SystemExit(f"Server said HTTP {reply.status_code}: {reply.text[:300]}")
    out = {}
    for m in reply.json()["data"]:
        b = m.get("budget") or {}
        limit, used = b.get("limit"), b.get("used", 0)
        out[m["id"]] = {"limit": limit, "used": used, "left": math.inf if limit is None else limit - used}
    return out


def read_prompts(path: Path) -> list[tuple[str, str]]:
    """[(id, prompt), ...] from a .jsonl ({"id", "prompt"}), a .csv (id,prompt) or a .txt (one prompt per line).

    A prompt without an id gets a hash of its text, so editing the file never shifts ids and the resume logic
    cannot mistake one prompt for another."""
    def auto(text: str) -> str:
        return "h" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

    with path.open(newline="", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
            prompts = [(str(r.get("id") or auto(r["prompt"])), r["prompt"]) for r in rows]
        elif path.suffix == ".csv":
            prompts = [(row.get("id") or auto(row["prompt"]), row["prompt"]) for row in csv.DictReader(f)]
        else:
            prompts = [(auto(t), t) for t in (line.rstrip("\n") for line in f) if t.strip()]
    ids = [pid for pid, _ in prompts]
    repeated = sorted({pid for pid in ids if ids.count(pid) > 1})
    if repeated:
        raise SystemExit(f"{path}: repeated prompt ids {repeated}")
    return prompts


def load_answers(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def query(prompts: list[tuple[str, str]], endpoints: list[str], output: Path, repeats: int = 1,
          max_tokens: int = config.MAX_TOKENS, dry_run: bool = False, parallel: int = config.PARALLEL,
          order: str = "round_robin") -> int:
    """Sends every prompt to every endpoint `repeats` times and appends each answer to `output` as one JSON line:
        {"endpoint", "id", "repeat", "prompt", "response", "ts", "latency_s", "raw"}
    Prints the plan first; with dry_run it stops there and spends nothing. Returns the number of planned requests.

    parallel: requests in flight at once (the server allows 4). Use 1 when answers must not share a server batch.
    order: "round_robin" sends each prompt to every endpoint in turn, so requests in flight go to different
    endpoints; "by_endpoint" finishes one endpoint before the next, so requests in flight hit the same endpoint."""
    s = session()
    left = {ep: b["left"] for ep, b in budget(s).items()}
    unknown = [ep for ep in endpoints if ep not in left]
    if unknown:
        raise SystemExit(f"Unknown endpoints {unknown}; the server lists: {', '.join(left)}")

    answered = load_answers(output)
    done = {(r["endpoint"], r["id"], r["repeat"]) for r in answered}
    logged = {r["id"]: r["prompt"] for r in answered}
    clash = [pid for pid, text in prompts if logged.get(pid, text) != text]
    if clash:
        raise SystemExit(f"Prompt ids {clash} already stand for a different prompt in {output}. Use new ids.")

    # Round-robin over endpoints, so a run that stops early has covered all of them evenly.
    jobs, planned, over = [], dict.fromkeys(endpoints, 0), 0
    if order == "by_endpoint":
        wanted = [(ep, pid, repeat, prompt) for ep in endpoints for repeat in range(repeats) for pid, prompt in prompts]
    else:
        wanted = [(ep, pid, repeat, prompt) for repeat in range(repeats) for pid, prompt in prompts for ep in endpoints]
    for ep, pid, repeat, prompt in wanted:
        if (ep, pid, repeat) in done:
            continue
        if planned[ep] < left[ep]:
            planned[ep] += 1
            jobs.append((ep, pid, repeat, prompt))
        else:
            over += 1

    total = len(prompts) * len(endpoints) * repeats
    print(f"{len(prompts)} prompts x {len(endpoints)} endpoints x {repeats} repeats = {total} answers: "
          f"{total - len(jobs) - over} already in {output.name}, {len(jobs)} to query"
          + (f", {over} left out (over budget)" if over else ""))
    for ep in endpoints:
        if planned[ep] and ep.startswith("ep-"):
            print(f"  {ep}: spends {planned[ep]}, leaves {left[ep] - planned[ep]} of its budget")
    if dry_run or not jobs:
        return len(jobs)

    output.parent.mkdir(parents=True, exist_ok=True)
    lock, exhausted, stop, count = threading.Lock(), set(), threading.Event(), [0]
    out = output.open("a", encoding="utf-8")

    def one(job) -> None:
        ep, pid, repeat, prompt = job
        while ep not in exhausted and not stop.is_set():
            started = time.time()
            try:
                r = s.post(f"{config.BASE_URL}/chat/completions", timeout=180,
                           json={"model": ep, "max_tokens": max_tokens,
                                 "messages": [{"role": "user", "content": prompt}]})
            except requests.RequestException as exc:
                print(f"Server unreachable ({type(exc).__name__}); retrying in 30 s.")
                stop.wait(30)
                continue
            try:
                body = r.json()
            except ValueError:
                body = {"unparsed": r.text}
            if r.status_code == 200:
                # Paid for: save it whatever shape it has.
                try:
                    text = body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    text = None
                line = {"endpoint": ep, "id": pid, "repeat": repeat, "prompt": prompt, "response": text,
                        "ts": round(started, 3), "latency_s": round(time.time() - started, 3), "raw": body}
                with lock:
                    out.write(json.dumps(line, ensure_ascii=False) + "\n")
                    out.flush()
                    count[0] += 1
                    if count[0] % 20 == 0 or count[0] == len(jobs):
                        print(f"{count[0]}/{len(jobs)} answers saved")
                if text is None:
                    print(f"{ep} id {pid}: reply had no text; raw reply saved.", file=sys.stderr)
                return
            detail = body.get("detail", str(body)[:300]) if isinstance(body, dict) else str(body)[:300]
            if r.status_code == 429 and "Retry-After" in r.headers:      # busy: wait for a free slot
                try:
                    stop.wait(float(r.headers["Retry-After"]))
                except ValueError:
                    stop.wait(5)
            elif r.status_code == 429:                                    # this endpoint's budget is used up
                with lock:
                    if ep not in exhausted:
                        exhausted.add(ep)
                        print(detail)
            elif r.status_code in (502, 503, 504):
                print(f"{detail} Retrying in 15 s.")
                stop.wait(15)
            else:                                                         # rejected: not charged, not retried
                print(f"{ep} id {pid}: HTTP {r.status_code}: {detail}", file=sys.stderr)
                return

    pool = ThreadPoolExecutor(max_workers=max(1, min(parallel, config.PARALLEL)))
    interrupted = False
    try:
        list(pool.map(one, jobs))
    except KeyboardInterrupt:
        stop.set()
        interrupted = True
        print("\nStopping: requests already sent are saved, nothing new is sent.")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        out.close()
    print(f"{count[0]} answers appended to {output}", flush=True)
    if interrupted:
        raise SystemExit(130)          # a stage made of several query runs must not carry on with the next one
    return len(jobs)
