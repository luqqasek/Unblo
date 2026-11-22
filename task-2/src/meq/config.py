"""Paths, server constants and API-key loading."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "starter"
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"

REFERENCE_ANSWERS = DATA / "reference_answers.jsonl"   # ref-01..ref-09, unlimited
SCORED_ANSWERS = DATA / "answers.jsonl"                # ep-01..ep-33, 100 responses each, ever
PRACTICE_TRUTH = STARTER / "practice_answers.csv"
SUBMISSION_LOG = SUBMISSIONS / "log.jsonl"

SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL}/{TASK_ID}/v1"
SUBMIT_URL = f"{SERVER_URL}/submit/{TASK_ID}"

SCORED = [f"ep-{i:02d}" for i in range(1, 34)]
MAX_TOKENS = 256          # server cap per response
PARALLEL = 4              # server cap on concurrent requests per team
COOLDOWN_S = 30 * 60      # one submission per 30 minutes, failed ones included


def api_key() -> str:
    """MEQ_API_KEY from the environment, else from .env in the workspace root."""
    key = os.environ.get("MEQ_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "MEQ_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        raise SystemExit(f"No API key. Put it in {env} as MEQ_API_KEY=<your team key>")
    return key
