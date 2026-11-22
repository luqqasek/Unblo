from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[2];p=root/'dashboard/build.py';s=p.read_text();marker='    # TriForensic measured adapter\n'
if marker not in s:
    anchor='    titles={';assert s.count(anchor)==1
    s=s.replace(anchor,marker+'    from triforensic_adapter import add_rows as add_triforensic_rows\n    add_triforensic_rows(EXPERIMENTS,DIST,row)\n'+anchor)
p.write_text(s);(root/'dashboard/triforensic_adapter.py').write_bytes((root/'experiments/triforensic_wm/dashboard_adapter.py').read_bytes())
subprocess.run([sys.executable,str(p)],check=True)
d=json.loads((root/'dashboard/dist/data.json').read_text());expected=len(json.loads((root/'experiments/triforensic_wm/results/measurements.json').read_text()));actual=[r for r in d['rows'] if r['experiment']=='triforensic_wm'];assert len(actual)==expected
for name in ('index.html','index-en.html'):assert 'triforensic_wm' in (root/'dashboard/dist'/name).read_text()
assert (root/'dashboard/dist/reports/triforensic_wm/REPORT.html').exists()
print({'triforensic_rows':len(actual),'both_languages_verified':True})
