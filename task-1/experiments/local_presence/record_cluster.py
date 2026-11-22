"""Record checkpoint provenance and remove only regenerable caches from this screen."""
import json,hashlib,importlib.metadata
from pathlib import Path

root=Path('/p/project1/training2666/pszenny1/native-localizer');runs=root/'local-presence-runs';record={'root':str(root),'checkpoints':[],'removed_caches':[],'data':{},'packages':{}}
for p in sorted(runs.glob('*/last.pt')):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    record['checkpoints'].append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
for family in ['pixelseal','trustmark']:
    record['data'][family]={}
    for part in ['train','dev']:
        p=root/f'local-presence-data/{family}/{part}/manifest.json';rows=json.loads(p.read_text());record['data'][family][part]={'encoded_realizations':len(rows),'unique_sources':len({r['source_id'] for r in rows}),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for name in ['torch','timm','numpy','trustmark','scipy','scikit-learn']:
    record['packages'][name]=importlib.metadata.version(name)
for source in ['pixelseal-presence','ps-adapt-control','ps-adapt-real','tm-adapt-control','tm-adapt-real']:
    p=runs/source/'spatial-cache.pt'
    if not p.exists():continue
    downstream='ps-spatial-iou' if source=='pixelseal-presence' else source+'-spatial'
    assert (runs/downstream/'completed.json').exists() and (p.parent/'last.pt').exists()
    record['removed_caches'].append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size});p.unlink()
record['freed_bytes']=sum(r['bytes'] for r in record['removed_caches'])
(runs/'cluster_provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'checkpoints':len(record['checkpoints']),'freed_bytes':record['freed_bytes'],'data':record['data']}))
