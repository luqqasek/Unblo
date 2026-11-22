from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[2];p=root/'dashboard/build.py';s=p.read_text();marker='    # Next ideas measured adapter\n'
if marker not in s:
    anchor='    titles={';assert s.count(anchor)==1;s=s.replace(anchor,marker+'    from next_ideas_adapter import add_rows as add_next_ideas\n    add_next_ideas(EXPERIMENTS,DIST,row)\n'+anchor)
p.write_text(s);(root/'dashboard/next_ideas_adapter.py').write_bytes((root/'experiments/next_ideas/dashboard_adapter.py').read_bytes());subprocess.run([sys.executable,str(p)],check=True)
d=json.loads((root/'dashboard/dist/data.json').read_text());expected=len(json.loads((root/'experiments/next_ideas/results/measurements.json').read_text()));assert len([r for r in d['rows'] if r['experiment']=='next_ideas'])==expected
for name in ('index.html','index-en.html'):assert 'next_ideas' in (root/'dashboard/dist'/name).read_text()
assert (root/'dashboard/dist/reports/next_ideas/REPORT.html').exists();print({'next_ideas_rows':expected,'both_languages_verified':True})
