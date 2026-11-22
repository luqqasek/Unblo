"""Export reproducible small artifacts, never the large image/feature mmap cache."""
from pathlib import Path
import zipfile,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
assert (HERE/'results/dashboard_verification.json').exists()
outputs=[]
for source,dest in [(HERE,Path.home()/'rtx_all_information_results.zip'),(ROOT/'dashboard/dist',Path.home()/'rtx_all_information_dashboard.zip')]:
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(source.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if source==HERE and p.suffix not in {'.py','.md','.json','.csv','.png','.npz','.pt','.log'}:continue
            z.write(p,p.relative_to(source))
    outputs.append({'path':str(dest),'bytes':dest.stat().st_size})
print(json.dumps(outputs))
