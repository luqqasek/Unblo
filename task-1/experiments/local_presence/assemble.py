"""One immutable candidate: new PS, GT-free TM scan, byte-identical SS masks."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.native_localizer.common import atomic_json,sha256
from experiments.small_zoom.proposals import boxmask
from watermark_localization.submission import save_submission,validate_submission
from .verification import verified,correct_bits

def main():
    here=Path(__file__).resolve().parent;task=here.parents[1];out=here/'results/submission';out.mkdir(exist_ok=True)
    metadata=json.loads((task/'data/test/metadata.json').read_text());names=[r['image_name'] for r in metadata];family={r['image_name']:r['method'] for r in metadata};assert len(names)==len(set(names))==1020
    oldpath=here.parent/'small_zoom/results/submission/submission-small-zoom.pt';old=torch.load(oldpath,map_location='cpu',weights_only=True);validate_submission(old,names);lookup={n:i for i,n in enumerate(old['image_names'])};masks=torch.stack([old['masks'][lookup[n]].to(torch.uint8) for n in names]);index={n:i for i,n in enumerate(names)}
    ps=out/'pixelseal-subset.pt';obj=torch.load(ps,map_location='cpu',weights_only=True);expected=[n for n in names if family[n]=='pixelseal'];validate_submission(obj,expected)
    for n,m in zip(obj['image_names'],obj['masks']):masks[index[n]]=m.to(torch.uint8)
    seen=set();accepted=[];wrong=0
    for part in range(4):
        folder=here/f'results/production/tm-scan-{part}';cfg=json.loads((folder/'config.json').read_text());rows=json.loads((folder/'proposals.json').read_text());assert {r['image_name'] for r in rows}==set(cfg['selected'])
        for row in rows:
            n=row['image_name'];assert n not in seen and family[n]=='trustmark' and not old['masks'][lookup[n]].any();seen.add(n);best=row['correct_top'][0];wrong+=int(verified({'bit_accuracy':row['wrong_max_accuracy']}))
            if verified(best):
                masks[index[n]]|=torch.from_numpy(boxmask(best['box']).astype(np.uint8));accepted.append({'image_name':n,**best,'correct_bits':correct_bits(best)})
    assert len(seen)==38
    for n in names:
        if family[n]=='stegastamp' or (family[n]=='trustmark' and n not in seen):assert torch.equal(masks[index[n]],old['masks'][lookup[n]].to(torch.uint8))
    dest=out/'submission-local-presence.pt';assert not dest.exists();save_submission(dest,names,masks,names);validate_submission(torch.load(dest,weights_only=True),names);assert dest.stat().st_size<=20_000_000
    changed={f:sum(not torch.equal(masks[index[n]],old['masks'][lookup[n]].to(torch.uint8)) for n in names if family[n]==f) for f in ['pixelseal','trustmark','stegastamp']}
    atomic_json(out/'assembly_validation.json',{'names':1020,'shape':list(masks.shape),'dtype':str(masks.dtype),'unique_names':True,'binary':True,'bytes':dest.stat().st_size,'sha256':sha256(dest),'base_submission_sha256':sha256(oldpath),'pixelseal_subset_sha256':sha256(ps),'changed_images':changed,'tm_scanned_empty_masks':len(seen),'tm_accepted':accepted,'wrong_query_acceptances':wrong,'stegastamp_unchanged':True,'nonempty_trustmark_masks_unchanged':True,'test_labels_used':False})
    print(json.dumps({'path':str(dest),'bytes':dest.stat().st_size,'sha256':sha256(dest),'changed_images':changed,'tm_accepted_count':len(accepted),'wrong_query_acceptances':wrong},indent=2))
if __name__=='__main__':main()
