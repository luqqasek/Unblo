from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[2];p=root/'dashboard/build.py';s=p.read_text();marker='    # Native PixelSeal regions adapter\n'
if marker not in s:
    anchor='    titles={';assert s.count(anchor)==1;s=s.replace(anchor,marker+'    from pixelseal_regions_adapter import add_rows as add_regions\n    add_regions(EXPERIMENTS,DIST,row)\n'+anchor)
p.write_text(s);(root/'dashboard/pixelseal_regions_adapter.py').write_bytes((root/'experiments/pixelseal_regions/dashboard_adapter.py').read_bytes());subprocess.run([sys.executable,str(p)],check=True)
d=json.loads((root/'dashboard/dist/data.json').read_text());expected=len(json.loads((root/'experiments/pixelseal_regions/results/measurements.json').read_text()));assert len([r for r in d['rows'] if r['experiment']=='pixelseal_regions'])==expected
for name in ('index.html','index-en.html'):assert 'pixelseal_regions' in (root/'dashboard/dist'/name).read_text()
print({'region_rows':expected,'both_languages':True})
