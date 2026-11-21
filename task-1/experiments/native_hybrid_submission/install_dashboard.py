from pathlib import Path
import subprocess
import sys
import json
root=Path(r'L:\3 - Projekty pythonowe\European-AI-Hackathon\task-1')
p=root/'dashboard/build.py'
s=p.read_text(encoding='utf-8')
marker='    # Native hybrid measured adapter\n'
if marker not in s:
    anchor='    titles={'
    assert s.count(anchor)==1
    insertion=marker+"    from native_hybrid_adapter import add_rows as add_native_hybrid_rows\n    add_native_hybrid_rows(EXPERIMENTS, DIST, row)\n"
    s=s.replace(anchor,insertion+anchor)
    p.write_text(s,encoding='utf-8')
(root/'dashboard/native_hybrid_adapter.py').write_bytes((root/'experiments/native_hybrid_submission/dashboard_adapter.py').read_bytes())
subprocess.run([sys.executable,str(p)],check=True)
data=json.loads((root/'dashboard/dist/data.json').read_text(encoding='utf-8'))
rows=[r for r in data['rows'] if r['experiment']=='native_hybrid_submission']
assert len(rows)==6
receipt=next(r for r in data['submissions'] if r['id']==2008)
assert receipt['score']==0.5993
for name in ['index.html','index-en.html']:
    text=(root/'dashboard/dist'/name).read_text(encoding='utf-8')
    assert '2008' in text and '0.5993' in text and 'native_hybrid_submission' in text
assert (root/'dashboard/dist/reports/native_hybrid_submission/REPORT.html').exists()
result={'rows':len(rows),'submission_id':2008,'score':receipt['score'],'both_languages_verified':True,'report_verified':True}
(root/'experiments/native_hybrid_submission/results/dashboard_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
