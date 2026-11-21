"""Submit your result to the hackathon server and print the score.

Before running, edit the two settings below:
    API_KEY          your team's API key (keep the double quotes)
    SUBMISSION_FILE  path to your .csv file, e.g. "submission.csv"
Then run:
    python submission.py

The file has one line per endpoint: endpoint,family,model (header line optional, any labels, no commas). Endpoints with
the same model label must share the family label. The old matrix format is no longer accepted.
Wait 30 minutes between submissions.

"""
import sys
from pathlib import Path

import requests

API_KEY = "YOUR_API_KEY_HERE"            # replace with your team's API key (keep the double quotes)
SUBMISSION_FILE = "/PATH/FILE.csv"       # replace with the path to your results file

TASK_ID = "33-model-equality"            # do not change
SERVER_URL = "http://34.147.229.55"

ALLOWED_SUFFIXES = (".csv",)

def main() -> int:
    file = Path(SUBMISSION_FILE)

    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY.strip():
        print("Set API_KEY at the top of submission.py to your team's API key.", file=sys.stderr)
        return 1
    if not file.is_file():
        print(f"File not found: {file}. Set SUBMISSION_FILE at the top of submission.py.", file=sys.stderr)
        return 1
    if file.suffix.lower() not in ALLOWED_SUFFIXES:
        print(f"{TASK_ID} expects a {' or '.join(ALLOWED_SUFFIXES)} file, got {file.name}", file=sys.stderr)
        return 1

    print(f"Submitting {file.name} to {TASK_ID}. Scoring can take a few minutes...")
    with file.open("rb") as f:
        try:
            reply = requests.post(
                f"{SERVER_URL.rstrip('/')}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files={"file": (file.name, f)},
                timeout=(600, 600),  # the server waits up to 500 s for the score
            )
        except requests.RequestException as exc:
            print(f"Could not reach the server: {exc}", file=sys.stderr)
            return 1

    try:
        body = reply.json()
    except ValueError:
        print(f"HTTP {reply.status_code}: {reply.text[:300]}", file=sys.stderr)
        return 1

    if reply.status_code != 200:
        print(f"Rejected (HTTP {reply.status_code}): {body.get('detail', body)}", file=sys.stderr)
        return 1
    status = body.get("status")
    if status == "success":
        print(f"Score (public split): {body.get('score')}")
        if "previous_best" in body:
            print(f"Previous best: {body['previous_best']}  (improved: {body.get('improved')})")
        print(body.get("message", ""))
        return 0
    print(f"{status}: {body.get('message')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
