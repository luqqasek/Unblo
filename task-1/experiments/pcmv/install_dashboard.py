from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[2];p=root/'dashboard/build.py';s=p.read_text();marker='    # PCMV measured adapter\n'
if marker not in s:
    anchor='    titles={';assert s.count(anchor)==1;s=s.replace(anchor,marker+'    from pcmv_adapter import add_rows as add_pcmv\n    add_pcmv(EXPERIMENTS,DIST,row)\n'+anchor)
p.write_text(s);(root/'dashboard/pcmv_adapter.py').write_bytes((root/'experiments/pcmv/dashboard_adapter.py').read_bytes());subprocess.run([sys.executable,str(p)],check=True)
d=json.loads((root/'dashboard/dist/data.json').read_text());expected=len(json.loads((root/'experiments/pcmv/results/measurements.json').read_text()));assert len([r for r in d['rows'] if r['experiment']=='pcmv'])==expected
for name in ('index.html','index-en.html'):assert 'pcmv' in (root/'dashboard/dist'/name).read_text()
assert (root/'dashboard/dist/reports/pcmv/REPORT.html').exists();print({'pcmv_measurement_rows':expected,'both_languages_verified':True})
