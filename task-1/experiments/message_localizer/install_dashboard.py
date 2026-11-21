from pathlib import Path
import subprocess, sys, json
root=Path(r'L:\3 - Projekty pythonowe\European-AI-Hackathon\task-1')
p=root/'dashboard/build.py'
s=p.read_text(encoding='utf-8')
marker='    # Message localizer measured adapter\n'
if marker not in s:
    anchor='    titles={'
    assert s.count(anchor)==1
    s=s.replace(anchor,marker+'    from message_localizer_adapter import add_rows as add_message_localizer_rows\n    add_message_localizer_rows(EXPERIMENTS, DIST, row)\n'+anchor)
    p.write_text(s,encoding='utf-8')
(root/'dashboard/message_localizer_adapter.py').write_bytes((root/'experiments/message_localizer/dashboard_adapter.py').read_bytes())
subprocess.run([sys.executable,str(p)],check=True)
data=json.loads((root/'dashboard/dist/data.json').read_text(encoding='utf-8'))
rows=[r for r in data['rows'] if r['experiment']=='message_localizer']
assert len(rows)==8
assert next(r for r in data['submissions'] if r['id']==2008)['score']==0.5993
for name in ['index.html','index-en.html']:
    text=(root/'dashboard/dist'/name).read_text(encoding='utf-8')
    assert 'message_localizer' in text and '2008' in text and '0.5993' in text
assert (root/'dashboard/dist/reports/message_localizer/REPORT.html').exists()
receipt_path=root/'experiments/message_localizer/results/submission/upload_receipt.json'
submission=None
if receipt_path.exists():
    response=json.loads(receipt_path.read_text(encoding='utf-8'))['response']
    if isinstance(response,str):response=json.loads(response)
    submission=response['submission_id']
    assert next(r for r in data['submissions'] if r['id']==submission)['score']==response['score']
    for name in ['index.html','index-en.html']:
        assert str(submission) in (root/'dashboard/dist'/name).read_text(encoding='utf-8')
result={'new_submission_verified':submission,'rows':len(rows),'both_languages_verified':True,'report_verified':True,'unchanged_submission':2008,'unchanged_public_score':0.5993}
(root/'experiments/message_localizer/results/dashboard_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
