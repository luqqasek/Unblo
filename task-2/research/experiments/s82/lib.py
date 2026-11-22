"""Shared helpers for session hackathon-82: loading answers and per-prompt-group two-sample z-scores."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from meq import features, moments, probes  # noqa: E402

KIND = {'ref-01': 'plain', 'ref-02': 'fp8', 'ref-03': 'sysprompt', 'ref-04': 'lora.7', 'ref-05': 'persona',
        'ref-06': 'watermark', 'ref-07': 'lora1.5', 'ref-08': 'temp1.0', 'ref-09': 'lora.3'}
TRUTH = {'ref-01': 'S', 'ref-02': 'T1', 'ref-03': 'S', 'ref-04': 'T3', 'ref-05': 'S', 'ref-06': 'S', 'ref-07': 'T4',
         'ref-08': 'S', 'ref-09': 'T2'}


def load(paths, prefix):
    """{endpoint: {prompt id: [responses in arrival order]}}"""
    data = defaultdict(lambda: defaultdict(list))
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.open(encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            if r['endpoint'].startswith(prefix):
                data[r['endpoint']][r['id']].append(r.get('response') or '')
    return data


def group(pid):
    if pid in probes.PACKS:
        return 'pack'
    if pid.startswith('open-') or pid.startswith('recon-self'):
        return 'open'
    if pid[0] == 'm' and pid[1].isdigit():
        return 'micro'
    if pid[0] == 'o' and pid[1].isdigit():
        return 'override'
    if pid.startswith('t-'):
        return 'template'
    if pid.startswith('d') and pid[1].isdigit():
        return 'domain'
    if pid.startswith('recon-extract') or pid.startswith('a1'):
        return 'extract'
    return 'single'


def reps(answers, pids=None, limit=None):
    return {pid: [features.represent(pid, x) for x in (v if limit is None else v[:limit])]
            for pid, v in answers.items() if pids is None or pid in pids}


def z(a, b, pids=None, limit=None):
    out = moments.mmd(reps(a, pids, limit), reps(b, pids, limit))
    return out['z'], out['stat'], out['prompts']
