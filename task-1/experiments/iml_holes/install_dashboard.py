from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[2];p=root/'dashboard/build.py';s=p.read_text();marker='    # Shared IML-ViT measured adapter\n'
if marker not in s:
    anchor='    titles={';assert s.count(anchor)==1;s=s.replace(anchor,marker+'    from iml_holes_adapter import add_rows as add_iml_holes\n    add_iml_holes(EXPERIMENTS,DIST,row)\n'+anchor)
p.write_text(s);(root/'dashboard/iml_holes_adapter.py').write_bytes((root/'experiments/iml_holes/dashboard_adapter.py').read_bytes());subprocess.run([sys.executable,str(p)],check=True)
d=json.loads((root/'dashboard/dist/data.json').read_text());expected=len(json.loads((root/'experiments/iml_holes/results/measurements.json').read_text()));assert len([r for r in d['rows'] if r['experiment']=='iml_holes'])==expected
for name in ('index.html','index-en.html'):assert 'iml_holes' in (root/'dashboard/dist'/name).read_text()
assert (root/'dashboard/dist/reports/iml_holes/REPORT.html').exists();print({'iml_rows':expected,'both_languages_verified':True})
