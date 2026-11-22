"""Replace explicitly selected families in the accepted #2675 submission."""
import argparse,json
from pathlib import Path
import torch
from experiments.native_localizer.common import atomic_json,sha256
from watermark_localization.submission import save_submission,validate_submission


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--parts',type=Path,required=True)
    p.add_argument('--families',nargs='+',choices=['pixelseal','trustmark'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--metadata',type=Path,default=Path('data/test/metadata.json'));a=p.parse_args()
    rows=json.loads(a.metadata.read_text());names=[r['image_name'] for r in rows];family={r['image_name']:r['method'] for r in rows};assert len(names)==len(set(names))==1020
    old=torch.load(a.base,map_location='cpu',weights_only=True);validate_submission(old,names)
    old_lookup=dict(zip(old['image_names'],old['masks']));masks=torch.stack([old_lookup[n].to(torch.uint8) for n in names]);lookup={n:i for i,n in enumerate(names)};sources=[]
    for f in a.families:
        seen=set();expected={n for n in names if family[n]==f}
        for part in range(4):
            path=a.parts/f/f'part{part}.pt';obj=torch.load(path,map_location='cpu',weights_only=True);validate_submission(obj,obj['image_names'])
            receipt=json.loads(path.with_suffix('.json').read_text());assert receipt['recipe']['all_real_refit'] and receipt['sha256']==sha256(path)
            for n,m in zip(obj['image_names'],obj['masks']):
                assert n in expected and n not in seen;seen.add(n);masks[lookup[n]]=m.to(torch.uint8)
            sources.append({'family':f,'part':part,'sha256':sha256(path),'recipe':receipt['recipe']})
        assert seen==expected and len(seen)==340
    for n in names:
        if family[n] not in a.families:assert torch.equal(masks[lookup[n]],old_lookup[n].to(torch.uint8))
    a.out.mkdir(parents=True,exist_ok=True);dest=a.out/'submission-triforensic.pt';assert not dest.exists(),'Preserve prior candidate'
    save_submission(dest,names,masks,names);validate_submission(torch.load(dest,weights_only=True),names);assert dest.stat().st_size<=20_000_000
    changed={f:sum(not torch.equal(masks[lookup[n]],old_lookup[n].to(torch.uint8)) for n in names if family[n]==f) for f in ('pixelseal','trustmark','stegastamp')}
    report={'names':1020,'shape':list(masks.shape),'dtype':str(masks.dtype),'unique_names':True,'binary':True,'bytes':dest.stat().st_size,'sha256':sha256(dest),'base_sha256':sha256(a.base),'sources':sources,'changed_images':changed,'test_GT_used':False,'preserved_families':[f for f in ('pixelseal','trustmark','stegastamp') if f not in a.families]}
    atomic_json(a.out/'assembly_validation.json',report);print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
