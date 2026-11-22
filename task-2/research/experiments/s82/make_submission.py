"""Write a submission CSV in the newer brief's label format: family = a letter, model = letter + number.

  .venv/bin/python research/experiments/s82/make_submission.py LABELS.json OUT.csv
LABELS.json: {endpoint: [family key, model key]} with any hashable keys; they are renamed to A, B, ... and A1, A2, ...
"""
import csv, json, string, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from meq import config, scoring

labels = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
eps = sorted(labels)
assert eps == config.SCORED, f"need exactly ep-01..ep-33, got {len(eps)}"
fam_letter, model_name, count = {}, {}, {}
rows = []
for e in eps:
    f, m = labels[e]
    if f not in fam_letter:
        fam_letter[f] = string.ascii_uppercase[len(fam_letter)]
    L = fam_letter[f]
    if (f, m) not in model_name:
        count[L] = count.get(L, 0) + 1
        model_name[(f, m)] = f"{L}{count[L]}"
    rows.append([e, L, model_name[(f, m)]])
with out.open('w', newline='', encoding='utf-8') as h:
    w = csv.writer(h)
    w.writerow(['endpoint', 'family', 'model'])
    w.writerows(rows)
got, problems = scoring.validate(out)
print(f"{out}: {len(rows)} endpoints, {len(fam_letter)} families, {len(model_name)} models; problems: {problems or 'none'}")
for L in sorted(set(r[1] for r in rows)):
    groups = {}
    for e, f, m in rows:
        if f == L:
            groups.setdefault(m, []).append(e[3:])
    print(f"  family {L}: " + '  '.join(f"{m}[{' '.join(v)}]" for m, v in groups.items()))
