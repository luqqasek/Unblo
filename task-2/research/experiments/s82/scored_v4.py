"""Apply rule v4 to the scored endpoints with the local evidence collected so far; write labels for a submission.

  .venv/bin/python research/experiments/s82/scored_v4.py NAME        writes submissions/NAME.labels.json and NAME.csv (no upload)
"""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import v3, v4
S = Path(__file__).parent
fam = json.loads((S / 'families.json').read_text())
# disguise tags established from the text (see uploads.jsonl notes and digits/prefix reports)
TAGS = {'ep-08': {'prompt'}, 'ep-20': {'prompt'}, 'ep-30': {'hot'}, 'ep-06': {'repeat'},
        'ep-16': {'prompt'}, 'ep-29': {'prompt'}, 'ep-31': {'hot'}, 'ep-27': {'skew'},
        'ep-02': {'prompt'}, 'ep-23': {'prompt'}, 'ep-33': {'hot'}, 'ep-04': {'repeat'}}
LORA = {'Q': {'general': [S / 'wb-ep-q-full.jsonl', S / 'wb-ep-q.jsonl']}, 'L': {'general': [S / 'wb-ep-l-full.jsonl', S / 'wb-ep-l.jsonl']},
        'N': {a: [S / f'wb-ep-n-min-{a}.jsonl'] for a in ('code', 'math', 'latex', 'persona')}}
QUANT = {'Q': S / 'qllr-q.json', 'L': S / 'qllr-l.json', 'N': S / 'qllr-n.json'}
NOT_PLAIN = {'ep-26'}            # upload 4: merging ep-26 into the base lowered the public score
labels = {}
for f in ('N', 'Q', 'L'):
    eps = sorted(e for e in fam if fam[e] == f)
    tags = {e: set(TAGS.get(e, ())) for e in eps}
    paths = {}
    for a, cands in LORA[f].items():
        for p in cands:
            if Path(p).exists() and Path(p).stat().st_size > 0:
                paths[a] = p; break
    lora = v3.lora_gains(paths)[0] if paths else None
    quant = json.loads(QUANT[f].read_text()) if QUANT[f].exists() else None
    y, why = v4.label(eps, tags, lora, quant, exclude_plain=NOT_PLAIN)
    print(f"== family {f}: adapters {sorted(paths)}; quantized-copy table {'yes' if quant else 'no'}")
    for e in eps:
        print(f"   {e} {y[e]:22s} {why[e]}")
        labels[e] = [f, y[e]]
name = sys.argv[1] if len(sys.argv) > 1 else 's82-v4'
lab = S / 'submissions' / f"{name}.labels.json"
lab.write_text(json.dumps(labels, indent=0))
r = subprocess.run([sys.executable, str(S / 'make_submission.py'), str(lab), str(S / 'submissions' / f"{name}.csv")], capture_output=True, text=True)
print(r.stdout.strip())
