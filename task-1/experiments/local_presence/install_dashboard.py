"""Install the measured-result adapter and verify both language builds."""
from pathlib import Path
import subprocess,sys,json

root=Path(__file__).resolve().parents[2]
p=root/'dashboard/build.py';s=p.read_text(encoding='utf-8');marker='    # Local presence measured adapter\n'
if marker not in s:
    anchor='    titles={';assert s.count(anchor)==1
    s=s.replace(anchor,marker+'    from local_presence_adapter import add_rows as add_local_presence_rows\n    add_local_presence_rows(EXPERIMENTS, DIST, row)\n'+anchor)
p.write_text(s,encoding='utf-8')
(root/'dashboard/local_presence_adapter.py').write_bytes((root/'experiments/local_presence/dashboard_adapter.py').read_bytes())
subprocess.run([sys.executable,str(p)],check=True)
d=json.loads((root/'dashboard/dist/data.json').read_text(encoding='utf-8'))
expected=len(json.loads((root/'experiments/local_presence/results/measurements.json').read_text()))
rows=[r for r in d['rows'] if r['experiment']=='local_presence'];assert len(rows)==expected
for name in ['index.html','index-en.html']:
    text=(root/'dashboard/dist'/name).read_text(encoding='utf-8');assert 'local_presence' in text
assert (root/'dashboard/dist/reports/local_presence/REPORT.html').exists()
receipts=[r for r in d['submissions'] if r['id']==2409]
if receipts:assert receipts[0]['score']==.6369
latest=root/'experiments/local_presence/results/submission/upload_receipt.json'
if latest.exists():
    response=json.loads(latest.read_text())['response'];response=json.loads(response) if isinstance(response,str) else response
    if response.get('status')=='success':
        assert next(r for r in d['submissions'] if r['id']==response['submission_id'])['score']==response['score']
        for name in ['index.html','index-en.html']:assert str(response['submission_id']) in (root/'dashboard/dist'/name).read_text(encoding='utf-8')
        instructions=root/'AGENTS.md'
        if instructions.exists() and response['submission_id']==2675:
            text=instructions.read_text(encoding='utf-8');text=text.replace('#2409 (0.6369','#2675 (0.6440');instructions.write_text(text,encoding='utf-8')
print(json.dumps({'local_presence_rows':len(rows),'both_languages_verified':True,'submission_2409_present':bool(receipts)}))
