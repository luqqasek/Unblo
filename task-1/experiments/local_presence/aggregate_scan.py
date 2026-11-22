"""CPU aggregation only; GT cannot change frozen scan candidates or verifier."""
import json
from pathlib import Path
import numpy as np
from .train import summary
from experiments.small_zoom.proposals import boxmask
from experiments.native_localizer.evaluate import image_metrics,gallery
from wmloc.data import LocalisationDataset
from .verification import verified

def main():
    here=Path(__file__).resolve().parent;exp=here.parent;out=here/'results/tm-scan';out.mkdir(exist_ok=True)
    original={r['image_name']:r for r in json.loads((exp/'small_zoom/results/finalscale/trustmark/proposals.json').read_text())};base={};scan={};done=[];eligible=set()
    for part in [0,1]:
        folder=here/f'results/tm-scan-{part}';done.append(json.loads((folder/'completed.json').read_text()));config=json.loads((folder/'config.json').read_text());eligible.update(config['all_eligible'])
        for r in json.loads((folder/'proposals.json').read_text()):
            assert r['image_name'] not in scan;scan[r['image_name']]=r
    assert set(scan)==eligible and len(scan)==sum(r['n'] for r in done)
    for fold in [1,2,3]:
        with np.load(exp/f'region_data/results/runs/trustmark/objects-adapt/fold{fold}/maps.npz') as z:base.update({str(n):p.squeeze() for n,p in zip(z['names'],z['probabilities'])})
    ds=LocalisationDataset(str(exp.parent/'data'),'train',methods=['trustmark']);br=[];rr=[];changed=[];names=[];probs=[];gt=[]
    for i in range(len(ds)):
        it=ds[i];n=it['image_name'];y=it['mask128'];p=base[n].copy();old=original[n]['correct_top']
        if old and old[0]['bit_accuracy']>=.9:p[boxmask(old[0]['box'])]=1
        b=image_metrics(p,y);b['image_name']=n;br.append(b);before=p.copy()
        if n in scan:
            assert not (p>=.5).any();candidate=scan[n]['correct_top']
            if candidate and verified(candidate[0]):p[boxmask(candidate[0]['box'])]=1
        r=image_metrics(p,y);r['image_name']=n;rr.append(r);names.append(n);probs.append(p);gt.append(y)
        if r['miou']!=b['miou']:
            changed.append({'image_name':n,'before':b,'after':r,'candidate':scan[n]['correct_top'][0]})
            gallery(here/f'results/trustmark_scan_example_{n}.png',it['image'],y,p,before)
    delta=np.array([r['miou']-b['miou'] for r,b in zip(rr,br)]);rng=np.random.default_rng(42);boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)])
    d={'baseline':summary(br),'final':summary(rr),'delta_miou':float(delta.mean()),'improved':int((delta>1e-8).sum()),'worse':int((delta< -1e-8).sum()),'wrong_accepts':sum(r['wrong_accepts'] for r in done),'seconds':max(r['seconds'] for r in done),'worker_seconds':sum(r['seconds'] for r in done),'peak_bytes':max(r['peak_bytes'] for r in done),'screened_images':len(scan),'paired_bootstrap_ci95':np.quantile(boots,[.025,.975]).tolist(),'protocol':'Reused170-image OOF with fixed GT-free empty-mask scan; adaptive development, not independent test'}
    for name,obj in [('completed',d),('per_image',rr),('baseline_per_image',br),('changed',changed)]: (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
    np.savez_compressed(out/'maps.npz',names=names,probabilities=probs,targets=gt)
    print(json.dumps(d,indent=2));print('CHANGED',[(r['image_name'],r['before']['miou'],r['after']['miou'],r['after']['fg_iou']) for r in changed])
if __name__=='__main__':main()
