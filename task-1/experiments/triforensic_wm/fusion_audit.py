"""Offline component audit. GT only summarizes outputs after inference."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import Engine,device_batch
from .model import TriForensicWM
from .evaluate import summarize
from experiments.native_localizer.common import atomic_json
from experiments.native_localizer.evaluate import image_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--expected-step',type=int,required=True);p.add_argument('--run',default='unified-v1');a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/f'triforensic-runs/{a.family}/{a.run}';state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);assert state['step']==a.expected_step
    variant='control' if a.family=='pixelseal' else 'objects';model=TriForensicWM(a.family,a.root/f'region-runs/{a.family}/{variant}-adapt/fold1/model.pt').cuda();model.load_state_dict(state['ema']);del state;model.eval().requires_grad_(False);parallel=torch.nn.DataParallel(model);engine=Engine(a.root,a.family)
    metrics={k:[] for k in ('logit_base','logit_final','l_loc','correction')};rows=[]
    with torch.no_grad():
        for start in range(0,len(engine.valid),8):
            ids=engine.valid[start:start+8];batch=device_batch(engine.stack([engine.real_full(i) for i in ids]));pred=parallel(batch['x'],batch['bits'],batch['npmap'])
            values={k:v[:,0].float().cpu().numpy() for k,v in pred.items() if k in (*metrics,'delta_wm','delta_clean')}
            for j,i in enumerate(ids):
                it=engine.real[i];y=it['mask128'].astype(bool);r={'image_name':it['image_name'],'true_area':float(y.mean()),'regions':{}}
                for key in metrics:
                    m=image_metrics(torch.from_numpy(values[key][j]).sigmoid().numpy(),y);m['image_name']=it['image_name'];metrics[key].append(m)
                for pol,mask in (('watermarked',y),('clean',~y)):
                    r['regions'][pol]={k:{'mean':float(v[j][mask].mean()),'q05':float(np.quantile(v[j][mask],.05)),'q95':float(np.quantile(v[j][mask],.95)),'max':float(v[j][mask].max())} for k,v in values.items()} if mask.any() else None
                rows.append(r)
    result={'step':a.expected_step,'alpha_wm':float(torch.nn.functional.softplus(model.refiner.raw_alpha_wm)),'alpha_clean':float(torch.nn.functional.softplus(model.refiner.raw_alpha_clean)),
            'component_metrics':{k:summarize(v) for k,v in metrics.items()},'per_image':rows,'protocol':'Offline decomposition at fixed0.5. Correction-only is diagnostic, not a selected inference pipeline. No GT in model input; no tuning.'}
    atomic_json(out/f'fusion-audit-{a.expected_step}.json',result);print(json.dumps({k:v for k,v in result.items() if k!='per_image'}),flush=True)
if __name__=='__main__':main()
