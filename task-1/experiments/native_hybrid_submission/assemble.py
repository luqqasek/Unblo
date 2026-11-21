"""Assemble fixed per-family predictions by exact names and validate safe reload."""
import json
from pathlib import Path
import hashlib
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from watermark_localization.submission import save_submission,validate_submission
OUT=Path(__file__).resolve().parent/'results'
metadata=json.loads((ROOT/'data/test/metadata.json').read_text())
names=[r['image_name'] for r in metadata]
assert len(names)==len(set(names))==1020
lookup={n:i for i,n in enumerate(names)}
masks=torch.empty((1020,128,128),dtype=torch.uint8);seen=set();sources={}
for family in ('pixelseal','trustmark','stegastamp'):
    path=OUT/f'{family}-subset.pt'
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    expected=[r['image_name'] for r in metadata if r['method']==family]
    validate_submission(artifact,expected)
    assert len(expected)==340 and not seen.intersection(artifact['image_names'])
    masks[[lookup[n] for n in artifact['image_names']]]=artifact['masks'].to(torch.uint8)
    seen.update(artifact['image_names'])
    sources[family]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n':340,'predicted_area':float(artifact['masks'].float().mean())}
assert seen==set(names)
path=OUT/'submission-best-per-family.pt'
if path.exists():raise FileExistsError(path)
save_submission(path,names,masks,names)
loaded=torch.load(path,weights_only=True);validate_submission(loaded,names)
assert torch.equal(loaded['masks'],masks)
report={'n':1020,'shape':list(masks.shape),'binary':True,'unique_names':True,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sources':sources,'uploaded':False}
(OUT/'assembly_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
