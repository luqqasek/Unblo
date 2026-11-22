"""Validate #4129, rebuild its masks, or rerun its PixelSeal checkpoint on official images."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'src')]
from experiments.h100_hour.h2_decode import expected_miou_decode
from watermark_localization.submission import validate_submission,save_submission

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(path,manifest,name):
    if sha(path)!=manifest['files'][name]['sha256']:raise ValueError('Checksum mismatch: '+str(path))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['validate','replay','infer']);p.add_argument('--assets',type=Path,default=HERE/'assets');p.add_argument('--data',type=Path);p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu');p.add_argument('--out',type=Path,default=Path('submission.pt'));a=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text());accepted=HERE/'submission-4129.pt';verify(accepted,manifest,'submission-4129.pt');reference=torch.load(accepted,map_location='cpu',weights_only=True);validate_submission(reference,json.loads((HERE/'image_names.json').read_text()));assert len(reference['image_names'])==1020 and reference['masks'].dtype==torch.uint8
    if a.mode=='validate':print('VALID: accepted #4129, 1020 binary uint8 masks, SHA256 verified.');return
    replay=a.assets/'replay-inputs-4129.npz';verify(replay,manifest,replay.name)
    with np.load(replay) as z:
        names=z['image_names'].tolist();ps_names=z['pixelseal_names'].tolist();base=z['pixelseal_ensemble_prob'].copy();prob=z['pixelseal_specialist_prob'].copy();masks=z['frozen_masks'].copy()
    if a.mode=='infer':
        if a.data is None:p.error('--data must contain the official test/metadata.json and test/images/')
        from model import FinalPixelSeal
        from wmloc.data import LocalisationDataset
        checkpoint=a.assets/'pixelseal-featuremix-4129.pt';verify(checkpoint,manifest,checkpoint.name)
        model=FinalPixelSeal(checkpoint).to(a.device);torch.set_num_threads(2);ds=LocalisationDataset(a.data,'test');by_name={r['image_name']:i for i,r in enumerate(ds.meta)}
        with torch.inference_mode():
            for i,name in enumerate(ps_names):
                item=ds[by_name[name]];image=torch.from_numpy(item['image'].transpose(2,0,1).copy())[None].to(a.device).float()/255;bits=torch.tensor(item['message'],device=a.device)[None].float();prob[i]=model(image,bits).sigmoid()[0].cpu().numpy()
                if (i+1)%20==0:print(f'PixelSeal {i+1}/340',flush=True)
    predicted=expected_miou_decode(.7*base+.3*prob);lookup={n:i for i,n in enumerate(names)}
    masks[[lookup[n] for n in ps_names]]=predicted
    save_submission(a.out,names,torch.from_numpy(masks),names)
    reloaded=torch.load(a.out,map_location='cpu',weights_only=True);ref_lookup={n:i for i,n in enumerate(reference['image_names'])};same=torch.equal(reloaded['masks'],reference['masks'][[ref_lookup[n]for n in names]])
    if a.mode=='replay' and not same:raise AssertionError('Replay differs from accepted masks')
    print(json.dumps(dict(output=str(a.out),images=1020,identical_masks_to_4129=same,sha256=sha(a.out)),indent=2))
if __name__=='__main__':main()
