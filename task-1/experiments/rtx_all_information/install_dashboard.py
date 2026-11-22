"""Add isolated adapter; preserve other experiments and build both language variants."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
build=ROOT/'dashboard/build.py';source=build.read_text(encoding='utf-8')
marker='    # RTX all-information measured adapter\n'
if marker not in source:
    anchor='    titles={'
    assert source.count(anchor)==1
    source=source.replace(anchor,marker+'    from rtx_all_information_adapter import add_rows as add_rtx_all_information_rows\n    add_rtx_all_information_rows(EXPERIMENTS, DIST, row)\n'+anchor)
    build.write_text(source,encoding='utf-8')
(ROOT/'dashboard/rtx_all_information_adapter.py').write_bytes((HERE/'dashboard_adapter.py').read_bytes())
subprocess.run([sys.executable,str(build)],check=True)
dist=ROOT/'dashboard/dist';data=json.loads((dist/'data.json').read_text(encoding='utf-8'))
expected=json.loads((HERE/'results/measurements.json').read_text(encoding='utf-8'))
rows=[r for r in data['rows'] if r['experiment']=='rtx_all_information']
assert len(rows)==len(expected) and len(expected) in (4,6),(len(rows),len(expected))
for name in ('index.html','index-en.html'):
    page=(dist/name).read_text(encoding='utf-8');assert 'rtx_all_information' in page
    for m in expected:assert str(m['miou']) in page,m
assert (dist/'reports/rtx_all_information/REPORT.html').exists()
result={'rows':len(rows),'both_languages_verified':True,'report_verified':True,'measurement_count':len(expected)}
(HERE/'results/dashboard_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
