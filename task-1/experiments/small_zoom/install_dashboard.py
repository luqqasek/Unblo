from pathlib import Path
import subprocess,sys,json
root=Path(r'L:\3 - Projekty pythonowe\European-AI-Hackathon\task-1')
p=root/'dashboard/build.py';s=p.read_text(encoding='utf-8')
for name,title in [('region_data','Region data measured adapter'),('small_zoom','Small zoom measured adapter')]:
    marker='    # '+title+'\n'
    if marker not in s:
        anchor='    titles={';assert s.count(anchor)==1
        s=s.replace(anchor,marker+f'    from {name}_adapter import add_rows as add_{name}_rows\n    add_{name}_rows(EXPERIMENTS, DIST, row)\n'+anchor)
    (root/f'dashboard/{name}_adapter.py').write_bytes((root/f'experiments/{name}/dashboard_adapter.py').read_bytes())
p.write_text(s,encoding='utf-8');subprocess.run([sys.executable,str(p)],check=True)
d=json.loads((root/'dashboard/dist/data.json').read_text(encoding='utf-8'))
expected=len(json.loads((root/'experiments/small_zoom/results/measurements.json').read_text()))
rows=[r for r in d['rows'] if r['experiment']=='small_zoom'];assert len(rows)==expected
assert next(r for r in d['submissions'] if r['id']==2163)['score']==.6145
new=root/'experiments/small_zoom/results/submission/server_receipt.json'
if new.exists():
    receipt=json.loads(new.read_text());response=receipt['response'];response=json.loads(response) if isinstance(response,str) else response
    if response.get('status')=='success':assert next(r for r in d['submissions'] if r['id']==response['submission_id'])['score']==response['score']
for name in ['index.html','index-en.html']:
    text=(root/'dashboard/dist'/name).read_text(encoding='utf-8');assert 'small_zoom' in text and '2163' in text
assert (root/'dashboard/dist/reports/small_zoom/REPORT.html').exists()
print(json.dumps({'small_zoom_rows':len(rows),'region_data_rows':sum(r['experiment']=='region_data' for r in d['rows']),'both_languages_verified':True,'submission_2163_unchanged':True}))
