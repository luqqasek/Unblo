"""One candidate, replacing SS only; preserve accepted PS/TM bit-for-bit."""
import json
from pathlib import Path
import torch
from experiments.native_localizer.common import atomic_json,sha256
from watermark_localization.submission import save_submission,validate_submission


def main():
    here=Path(__file__).parent;task=here.parents[1];out=here/'results/submission';out.mkdir(exist_ok=True);base=here.parent/'local_presence/results/submission/submission-local-presence.pt';rows=json.loads((task/'data/test/metadata.json').read_text());names=[r['image_name'] for r in rows];family={r['image_name']:r['method'] for r in rows};assert len(names)==len(set(names))==1020
    previous=torch.load(base,map_location='cpu',weights_only=True);validate_submission(previous,names);old=dict(zip(previous['image_names'],previous['masks']));masks=torch.stack([old[n].to(torch.uint8) for n in names]);lookup={n:i for i,n in enumerate(names)};seen=set();receipts=[]
    for part in range(4):
        path=here/f'results/production/stegastamp/part{part}.pt';obj=torch.load(path,map_location='cpu',weights_only=True);validate_submission(obj,obj['image_names']);receipt=json.loads(path.with_suffix('.json').read_text());assert receipt['artifact_sha256']==sha256(path);receipts.append(receipt)
        for n,m in zip(obj['image_names'],obj['masks']):assert n not in seen and family[n]=='stegastamp';seen.add(n);masks[lookup[n]]=m.to(torch.uint8)
    assert len(seen)==340 and len({r['recipe_sha256'] for r in receipts})==1 and len({r['checkpoint_sha256'] for r in receipts})==1
    changed={f:sum(not torch.equal(masks[lookup[n]],old[n].to(torch.uint8)) for n in names if family[n]==f) for f in ('pixelseal','trustmark','stegastamp')};assert changed['pixelseal']==changed['trustmark']==0
    dest=out/'submission-next-ideas.pt';assert not dest.exists();save_submission(dest,names,masks,names);validate_submission(torch.load(dest,weights_only=True),names);assert dest.stat().st_size<=20_000_000
    report={'names':1020,'shape':list(masks.shape),'dtype':str(masks.dtype),'unique_names':True,'binary':True,'bytes':dest.stat().st_size,'sha256':sha256(dest),'base_sha256':sha256(base),'changed_images':changed,'sources':receipts,'test_GT_used':False};atomic_json(out/'assembly_validation.json',report);print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
