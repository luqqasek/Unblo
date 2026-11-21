"""Validate combined artifact and export explicitly non-submittable family subsets."""
import argparse
import json
from pathlib import Path
import torch
from watermark_localization.submission import validate_submission
from .common import atomic_json,sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument('--submission',type=Path,required=True); p.add_argument('--data',type=Path,required=True); a=p.parse_args()
    meta=json.loads((a.data/'test/metadata.json').read_text())
    names=[r['image_name'] for r in meta]
    obj=torch.load(a.submission,map_location='cpu',weights_only=True)
    validate_submission(obj,names)
    assert len(names)==1020 and a.submission.stat().st_size<=20_000_000
    lookup={name:i for i,name in enumerate(obj['image_names'])}
    report={}
    for family in ('pixelseal','trustmark','stegastamp'):
        selected=[r['image_name'] for r in meta if r['method']==family]
        masks=obj['masks'][[lookup[n] for n in selected]]
        assert len(selected)==340
        path=a.submission.with_name(f'{a.submission.stem}-{family}-subset.pt')
        if path.exists(): raise FileExistsError(path)
        torch.save({'image_names':selected,'masks':masks},path)
        report[family]={'n':340,'file':path.name,'sha256':sha256(path),'predicted_area':float(masks.float().mean()),'empty_masks':int((masks.flatten(1).sum(1)==0).sum()),'full_masks':int((masks.flatten(1).sum(1)==128*128).sum())}
    atomic_json(a.submission.with_suffix('.package.json'),{'families':report,'sha256':sha256(a.submission),'bytes':a.submission.stat().st_size,'valid':True,'subset_files_are_not_complete_submissions':True})
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
