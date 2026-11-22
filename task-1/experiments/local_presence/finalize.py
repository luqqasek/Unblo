"""CPU-only complete bucket reporting and paired-image diagnostics."""
import json
from pathlib import Path
import numpy as np
from .train import summary
from experiments.native_localizer.evaluate import gallery
from wmloc.data import LocalisationDataset

def main():
    here=Path(__file__).resolve().parent;root=here/'results';ds=LocalisationDataset(str(here.parents[1]/'data'),'train',methods=['pixelseal']);lookup={r['image_name']:i for i,r in enumerate(ds.meta)}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'completed.json').exists():continue
        d=json.loads((folder/'completed.json').read_text())
        if 'baseline' not in d:continue
        if (folder/'correction-0.json').exists():
            step=600 if (folder/'correction-600.json').exists() else 300
            before=json.loads((folder/'correction-0.json').read_text())['per_image'];after=json.loads((folder/f'correction-{step}.json').read_text())['per_image']
        else:
            bp=folder/'baseline_per_image.json'
            if not bp.exists():bp=root/'tm-scan/baseline_per_image.json'
            before=json.loads(bp.read_text());after=json.loads((folder/'per_image.json').read_text())
        b={r['image_name']:r for r in before};a={r['image_name']:r for r in after};assert set(a)==set(b)
        d['baseline']=summary(before);d['final']=summary(after);deltas=np.array([a[n]['miou']-b[n]['miou'] for n in sorted(a)])
        d['improved']=int((deltas>1e-8).sum());d['worse']=int((deltas< -1e-8).sum());rng=np.random.default_rng(42);boot=np.array([rng.choice(deltas,len(deltas),replace=True).mean() for _ in range(10000)])
        d['paired_bootstrap_ci95']=np.quantile(boot,[.025,.975]).tolist();(folder/'completed.json').write_text(json.dumps(d,indent=2)+'\n')
        if folder.name=='ps-spatial-iou':
            with np.load(folder/'maps-0.npz') as z:baseline={str(n):p for n,p in zip(z['names'],z['probabilities'])}
            with np.load(folder/'maps-600.npz') as z:
                entries=[(str(n),p,y) for n,p,y in zip(z['names'],z['probabilities'],z['targets'])]
            entries.sort(key=lambda r:a[r[0]]['miou']-b[r[0]]['miou'])
            for label,(n,p,y) in zip(['regression','gain'],[entries[0],entries[-1]]):
                gallery(root/f'pixelseal_example_{label}_{n}.png',ds[lookup[n]]['image'],y,p,baseline[n])
            (folder/'gallery_selection.json').write_text(json.dumps([{'kind':k,'image_name':r[0],'baseline':b[r[0]],'final':a[r[0]]} for k,r in zip(['regression','gain'],[entries[0],entries[-1]])],indent=2)+'\n')
    print('Finalized full buckets and image bootstrap; no model selection or threshold changes.')
if __name__=='__main__':main()
