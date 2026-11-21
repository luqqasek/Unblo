"""Combine selected PS/TM full refits and unchanged StegaStamp by unique names."""
from pathlib import Path
import json
import torch
from watermark_localization.submission import save_submission,validate_submission
from experiments.native_localizer.common import atomic_json,sha256

def main():
    root=Path(__file__).resolve().parents[2]
    out=Path(__file__).resolve().parent/'results/submission'
    recipe=json.loads((out/'frozen_recipe.json').read_text());assert recipe['recipe_frozen']
    metadata=json.loads((root/'data/test/metadata.json').read_text());names=[r['image_name'] for r in metadata]
    assert len(names)==len(set(names))==1020
    lookup={n:i for i,n in enumerate(names)};seen=set();masks=torch.empty((1020,128,128),dtype=torch.uint8);sources={}
    for family in ('pixelseal','trustmark','stegastamp'):
        path=out/f'{family}-subset.pt' if family!='stegastamp' else root/recipe['families'][family]['subset']
        if family=='stegastamp':assert sha256(path)==recipe['families'][family]['subset_sha256']
        obj=torch.load(path,map_location='cpu',weights_only=True)
        expected=[r['image_name'] for r in metadata if r['method']==family]
        validate_submission(obj,expected);assert len(expected)==340 and not seen.intersection(obj['image_names'])
        masks[[lookup[n] for n in obj['image_names']]]=obj['masks'].to(torch.uint8);seen.update(obj['image_names'])
        sources[family]={'n':340,'sha256':sha256(path),'predicted_area':float(obj['masks'].float().mean())}
    assert seen==set(names)
    dest=out/'submission-message-localizer.pt'
    if dest.exists():raise FileExistsError(dest)
    save_submission(dest,names,masks,names);loaded=torch.load(dest,weights_only=True);validate_submission(loaded,names)
    assert torch.equal(loaded['masks'],masks) and dest.stat().st_size<=20_000_000
    result={'n':1020,'shape':list(masks.shape),'dtype':str(masks.dtype),'unique_names':True,'binary':True,'bytes':dest.stat().st_size,'sha256':sha256(dest),'sources':sources}
    atomic_json(out/'assembly_validation.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
