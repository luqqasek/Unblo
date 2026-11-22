"""Local benchmark for rule v4 vs earlier rules: 0.625 test bed + 0.375 practice, task formula with known families."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lib import ROOT, KIND, TRUTH, load
import typed2 as T, v3, v4

S = Path(__file__).parent
verbose = '-v' in sys.argv
PLAN = {'f2-who': 3, 'recon-extract-1': 1, 'k4-names': 2, 'k6-picks': 2, 'm8-capital': 2, 'm2-count': 2, 'open-market': 4, 's1-digits': 4}
tb_truth = {r['endpoint']: (r['family'], r['model'], r['kind']) for r in csv.DictReader(open(ROOT / 'testbed/out/truth.csv'))}
tb = defaultdict(lambda: defaultdict(list)); KEEP = set()
for line in open(S / 'tb-plan-answers.jsonl'):
    r = json.loads(line)
    if r['response'] not in tb[r['endpoint']][r['id']]:
        tb[r['endpoint']][r['id']].append(r['response']); KEEP.add((r['endpoint'], r['id'], r['repeat']))
seq = load([S / 'seq-testbed-answers.jsonl'], 'tb-')
for e in tb:
    tb[e]['s1-digits'] = [r for pid in sorted(seq.get(e, {})) if pid.startswith('dg-') for r in seq[e][pid]][:4]
pr_all = load([ROOT / 'data/reference_answers.jsonl', S / 'practice.jsonl'], 'ref-')
for e in pr_all:
    pr_all[e]['open-market'] = pr_all[e].get('open-market', []) + pr_all[e].pop('recon-self-seq', [])
pr = {e: {pid: v for pid, v in d.items()} for e, d in pr_all.items()}          # practice: all collected answers (as for the local scoring)
pr_truth = {e: ('P', t, KIND[e]) for e, t in TRUTH.items()}
LORA = {'Q15': {'general': S / 'wb-tb-plan-q15.jsonl'}, 'Q30': {a: S / f'wb-tb-plan-q30-{a}.jsonl' for a in ('persona', 'code', 'math', 'latex')}}
QUANT = {'Q15': S / 'qllr-tb-q15.json', 'Q30': S / 'qllr-tb-q30.json', 'OL1': S / 'qllr-tb-ol1.json', 'Q05': S / 'qllr-tb-q05.json', 'P': S / 'qllr-practice-all.json'}


def run(data, truth, rule):
    tot = [0, 0, 0, 0]; missing = []
    for famname in sorted({truth[e][0] for e in data}):
        fam = {e: data[e] for e in data if truth[e][0] == famname}
        tags = T.types(T.features(fam))
        why = {}
        if rule == 'v2':
            y = {e: ('BASE' if tags[e] else e) for e in fam}
        else:
            paths = {a: p for a, p in LORA.get(famname, {}).items() if Path(p).exists()}
            lora = v3.lora_gains(paths, KEEP if famname != 'P' else None)[0] if paths else None
            quant = json.loads(Path(QUANT[famname]).read_text()) if Path(QUANT[famname]).exists() else None
            if quant is None:
                missing.append(famname)
            y, why = v4.label(sorted(fam), tags, lora, quant)
        c = T.counts({e: truth[e][1] for e in fam}, y)
        tot = [a + b for a, b in zip(tot, c)]
        if verbose and rule != 'v2':
            print(f"== {famname} ({rule}) MCC {T.mcc(c):.3f} {c}")
            for e in sorted(fam):
                print(f"   {e} {y[e]:22s} {why.get(e, '')[:62]:62s} | {truth[e][1]:16s} {truth[e][2]}")
    return 0.1 + 0.9 * max(0.0, T.mcc(tot)), tuple(tot), missing


for rule in ('v2', 'v4'):
    st, ct, m1 = run(tb, tb_truth, rule)
    sp, cp, m2 = run(pr, pr_truth, rule)
    print(f"rule {rule}: test bed {st:.3f} {ct}  practice {sp:.3f} {cp}  BENCHMARK {0.625 * st + 0.375 * sp:.3f}" + (f"   (local runs still missing for {m1 + m2})" if m1 + m2 else ''))
