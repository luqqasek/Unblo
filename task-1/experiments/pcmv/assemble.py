"""Replace only PixelSeal in accepted #3054; preserve other family masks exactly."""
import json
from pathlib import Path
import torch
from experiments.native_localizer.common import atomic_json,sha256
from watermark_localization.submission import save_submission,validate_submission


def main():
    here=Path(__file__).parent;task=here.parents[1];out=here/'results/base-candidate';dest=out/'submission';dest.mkdir(parents=True,exist_ok=True)
    check=json.loads((out/'evaluation.json').read_text());decision=json.loads((out/'decision.json').read_text());assert decision['selected'];inference=json.loads((out/'test-inference.json').read_text());subset=out/'pixelseal-subset.pt';assert sha256(subset)==inference['artifact_sha256']
    previous=here.parent/'next_ideas/results/extended/submission/submission-next-ideas-450.pt';assert sha256(previous)=='e8abdf5626b4dd5af067cd8266adeb95a01f1fa19ecf03c9e5df82090e887714'
    metadata=json.loads((task/'data/test/metadata.json').read_text());names=[r['image_name'] for r in metadata];family={r['image_name']:r['method'] for r in metadata};old=torch.load(previous,map_location='cpu',weights_only=True);validate_submission(old,names);oldm=dict(zip(old['image_names'],old['masks']));new=torch.load(subset,map_location='cpu',weights_only=True);validate_submission(new,new['image_names']);replacement=dict(zip(new['image_names'],new['masks']));assert len(replacement)==340 and all(family[n]=='pixelseal' for n in replacement)
    masks=torch.stack([replacement.get(n,oldm[n]).to(torch.uint8) for n in names]);changed={f:sum(not torch.equal(m,oldm[n].to(torch.uint8)) for n,m in zip(names,masks) if family[n]==f) for f in ('pixelseal','trustmark','stegastamp')};assert changed['trustmark']==changed['stegastamp']==0 and changed['pixelseal']>0
    file=dest/'submission-ps-global1561.pt';assert not file.exists();save_submission(file,names,masks,names);assert file.stat().st_size<=20_000_000;validate_submission(torch.load(file,weights_only=True),names)
    report={'base_submission':3054,'base_sha256':sha256(previous),'sha256':sha256(file),'bytes':file.stat().st_size,'shape':list(masks.shape),'dtype':str(masks.dtype),'unique_names':len(set(names))==1020,'binary':True,'changed_images':changed,'checkpoint_sha256':inference['checkpoint_sha256'],'test_GT_used':False,'real_training_images':113,'all170_refit':False,'selection':decision};atomic_json(dest/'assembly_validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
