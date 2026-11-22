from pathlib import Path
import subprocess,sys,json
root=Path(r'L:\3 - Projekty pythonowe\European-AI-Hackathon\task-1')
p=root/'dashboard/build.py';s=p.read_text(encoding='utf-8');marker='    # Region data measured adapter\n'
if marker not in s:
 anchor='    titles={';assert s.count(anchor)==1
 s=s.replace(anchor,marker+'    from region_data_adapter import add_rows as add_region_data_rows\n    add_region_data_rows(EXPERIMENTS, DIST, row)\n'+anchor);p.write_text(s,encoding='utf-8')
(root/'dashboard/region_data_adapter.py').write_bytes((root/'experiments/region_data/dashboard_adapter.py').read_bytes())
subprocess.run([sys.executable,str(p)],check=True)
d=json.loads((root/'dashboard/dist/data.json').read_text(encoding='utf-8'));rows=[r for r in d['rows'] if r['experiment']=='region_data'];assert len(rows)==12
assert next(r for r in d['submissions'] if r['id']==2163)['score']==.6145
for name in ['index.html','index-en.html']:
 text=(root/'dashboard/dist'/name).read_text(encoding='utf-8');assert 'region_data' in text and '2163' in text
assert (root/'dashboard/dist/reports/region_data/REPORT.html').exists()
print(json.dumps({'region_data_rows':len(rows),'both_languages_verified':True,'submission_2163_unchanged':True}))
