"""Combine frozen new PS/TM subsets with unchanged StegaStamp; never overwrite."""
import json
from pathlib import Path
import torch
from watermark_localization.submission import save_submission,validate_submission
from experiments.native_localizer.common import atomic_json,sha256

def main():
    task=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent/'results/submission';out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((task/'data/test/metadata.json').read_text());names=[m['image_name'] for m in metadata];assert len(names)==len(set(names))==1020
    lookup={n:i for i,n in enumerate(names)};allmasks=torch.empty((1020,128,128),dtype=torch.uint8);seen=set();sources={}
    for family in ['pixelseal','trustmark','stegastamp']:
        path=out/f'{family}-subset.pt' if family!='stegastamp' else task/'experiments/native_hybrid_submission/results/stegastamp-subset.pt'
        obj=torch.load(path,map_location='cpu',weights_only=True);expected=[r['image_name'] for r in metadata if r['method']==family];validate_submission(obj,expected)
        assert len(expected)==340 and not seen&set(obj['image_names']);seen.update(obj['image_names']);allmasks[[lookup[n] for n in obj['image_names']]]=obj['masks'].to(torch.uint8)
        sources[family]={'path':str(path),'sha256':sha256(path),'n':340}
    assert seen==set(names);dest=out/'submission-small-zoom.pt';assert not dest.exists()
    # StegaStamp explicitly unchanged from the previous accepted production artifact.
    previous=torch.load(task/'experiments/message_localizer/results/submission/submission-message-localizer.pt',map_location='cpu',weights_only=True)
    previous_lookup={n:i for i,n in enumerate(previous['image_names'])}
    for m in metadata:
        if m['method']=='stegastamp':assert torch.equal(allmasks[lookup[m['image_name']]],previous['masks'][previous_lookup[m['image_name']]].to(torch.uint8))
    recipe={'recipe_frozen':True,'selection':'validation only; no test labels; no leaderboard hyperparameter search','pixelseal':'region-control600 + full150 replay refit, threshold0.5','trustmark':'region-objects600 + full150 replay refit + NRGMark512/1536/2560/4096 proposals, exact crop decoder bit accuracy>=0.9, union','stegastamp':'unchanged accepted #2163 subset','sources':sources}
    atomic_json(out/'frozen_recipe.json',recipe);save_submission(dest,names,allmasks,names);obj=torch.load(dest,weights_only=True);validate_submission(obj,names)
    assert dest.stat().st_size<=20_000_000 and torch.equal(obj['masks'],allmasks)
    atomic_json(out/'assembly_validation.json',{'n':1020,'shape':list(allmasks.shape),'dtype':str(allmasks.dtype),'unique_names':True,'binary':True,'bytes':dest.stat().st_size,'sha256':sha256(dest),'stegastamp_unchanged':True,'sources':sources})
    print('Validated',dest,dest.stat().st_size,sha256(dest))
if __name__=='__main__':main()
