"""meq: command line for the Model Equality workspace.

  meq budget                      queries left on every endpoint
  meq query FILE                  send prompts to the reference endpoints (unlimited, never scored)
  meq query FILE --scored         plan a run on ep-01..ep-33 and print what it would spend; add --yes to spend it
  meq score FILE                  score a practice answer (ref-01..ref-09) against the published truth
  meq check FILE                  check a submission file's format
  meq submit FILE                 check and show a submission; add --yes to upload it (one per 30 minutes)
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import requests

from . import api, config, scoring


def cmd_budget(args) -> int:
    for ep, b in api.budget().items():
        limit = "no limit" if b["limit"] is None else f"{b['left']} left of {b['limit']}"
        print(f"  {ep}: {limit} ({b['used']} used)")
    return 0


def cmd_query(args) -> int:
    prefix, default_out = ("ep-", config.SCORED_ANSWERS) if args.scored else ("ref-", config.REFERENCE_ANSWERS)
    listed = [ep for ep in api.budget() if ep.startswith(prefix)]
    endpoints = args.endpoints.split(",") if args.endpoints else listed
    outside = [ep for ep in endpoints if ep not in listed]
    if outside:
        hint = "" if args.scored else " (scored endpoints need --scored)"
        sys.exit(f"Not {prefix}* endpoints the server lists: {outside}{hint}")
    dry_run = args.scored and not args.yes
    planned = api.query(api.read_prompts(args.file), endpoints, args.out or default_out, repeats=args.repeats,
                        max_tokens=args.max_tokens, dry_run=dry_run)
    if dry_run and planned:
        print("Dry run: nothing was sent. This budget never refills; add --yes to spend it.")
    return 0


def cmd_score(args) -> int:
    truth = scoring.read_labels(args.truth, args.prefix)
    if not truth:
        sys.exit(f"No endpoints starting with {args.prefix!r} in {args.truth}")
    result = scoring.score(truth, scoring.read_labels(args.file, args.prefix))
    tp, fp, fn, tn = result["model_counts"]
    print(f"Practice model MCC: {result['model_mcc'] or 0.0:.3f}  ({tp + tn} pairs right, {fp} wrongly called the "
          f"same, {fn} wrongly called different)")
    if result["family_mcc"] is not None:      # undefined on the published practice set: it is a single family
        print(f"Family MCC: {result['family_mcc']:.3f}  ->  combined score {result['score']:.3f}")
    for a, kind_a, b, kind_b, guess in result["wrong"]:
        print(f"  {a} ({kind_a}) and {b} ({kind_b}): you said {'same' if guess else 'different'}")
    return 0


def cmd_check(args) -> int:
    labels, problems = scoring.validate(args.file)
    print(scoring.describe(labels))
    for p in problems:
        print(f"  PROBLEM: {p}")
    print("Format OK." if not problems else f"{len(problems)} problems: the server would reject this file.")
    return 1 if problems else 0


def last_submission() -> float:
    """Time of the most recent upload attempt, 0 if none."""
    if not config.SUBMISSION_LOG.exists():
        return 0.0
    lines = [json.loads(line) for line in config.SUBMISSION_LOG.read_text(encoding="utf-8").splitlines() if line]
    return max((entry["ts"] for entry in lines), default=0.0)


def cmd_submit(args) -> int:
    if cmd_check(args):
        return 1
    wait = config.COOLDOWN_S - (time.time() - last_submission())
    if wait > 0 and not args.force:
        print(f"Cool-down: last upload was {int((config.COOLDOWN_S - wait) / 60)} min ago. "
              f"Wait {int(wait / 60) + 1} more min (--force overrides; a rejected upload restarts the 30 min).")
        return 1
    if not args.yes:
        print("Dry run: nothing was uploaded. Add --yes to submit (starts a 30 minute cool-down).")
        return 0

    key = config.api_key()
    config.SUBMISSIONS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    kept = config.SUBMISSIONS / f"{stamp}_{args.file.name}"
    shutil.copyfile(args.file, kept)
    entry = {"ts": time.time(), "file": kept.name, "sha1": hashlib.sha1(kept.read_bytes()).hexdigest()}
    print(f"Submitting {args.file.name}. Scoring can take a few minutes...")
    try:
        with kept.open("rb") as f:
            reply = requests.post(config.SUBMIT_URL, headers={"X-API-Key": key}, files={"file": (args.file.name, f)},
                                  timeout=(600, 600))
        try:
            body = reply.json()
        except ValueError:
            body = {"unparsed": reply.text[:300]}
        entry.update(http=reply.status_code, reply=body)
    except requests.RequestException as exc:
        entry.update(http=None, reply={"error": str(exc)})
    with config.SUBMISSION_LOG.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")

    body = entry["reply"]
    if entry["http"] == 200 and body.get("status") == "success":
        print(f"Public score: {body.get('score')}  (a rough check only; the final ranking uses other pairs)")
        if "previous_best" in body:
            print(f"Previous best: {body['previous_best']}  (improved: {body.get('improved')})")
        print(body.get("message", ""))
        return 0
    print(f"Not scored (HTTP {entry['http']}): {body}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="meq", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("budget").set_defaults(run=cmd_budget)

    q = sub.add_parser("query")
    q.add_argument("file", type=Path, help="prompts: .jsonl ({id, prompt}), .csv (id,prompt) or .txt (one per line)")
    q.add_argument("--scored", action="store_true", help="target ep-* instead of ref-* (dry run unless --yes)")
    q.add_argument("--yes", action="store_true", help="really spend scored budget")
    q.add_argument("--endpoints", help="comma-separated subset, e.g. ref-01,ref-02")
    q.add_argument("--repeats", type=int, default=1, help="answers per prompt and endpoint")
    q.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
    q.add_argument("--out", type=Path, help="answers file (default: data/reference_answers.jsonl or data/answers.jsonl)")
    q.set_defaults(run=cmd_query)

    s = sub.add_parser("score")
    s.add_argument("file", type=Path)
    s.add_argument("--truth", type=Path, default=config.PRACTICE_TRUTH)
    s.add_argument("--prefix", default="ref-", help="endpoint name prefix to score; use tb- with testbed/out/truth.csv")
    s.set_defaults(run=cmd_score)

    c = sub.add_parser("check")
    c.add_argument("file", type=Path)
    c.set_defaults(run=cmd_check)

    u = sub.add_parser("submit")
    u.add_argument("file", type=Path)
    u.add_argument("--yes", action="store_true", help="really upload")
    u.add_argument("--force", action="store_true", help="ignore the local 30 minute cool-down check")
    u.set_defaults(run=cmd_submit)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
