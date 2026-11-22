"""Development-only examples: strongest gains, regressions, tiny regions and holes."""
import json
from pathlib import Path
import numpy as np
from wmloc.data import LocalisationDataset
from experiments.native_localizer.evaluate import gallery,image_metrics
from experiments.native_localizer.common import atomic_json


def main():
    root=Path(__file__).parent;out=root/'results/base-candidate';new=np.load(out/'all-real-maps.npz');old=np.load(root.parent/'local_presence/results/ps-data-small/maps-150.npz');ni={str(n):i for i,n in enumerate(new['names'])};oi={str(n):i for i,n in enumerate(old['names'])};rows=[]
    for n,i in oi.items():
        j=ni[n];assert np.array_equal(old['targets'][i],new['targets'][j]);a=image_metrics(old['probabilities'][i],old['targets'][i]);b=image_metrics(new['probabilities'][j],new['targets'][j]);rows.append({'name':n,'before':a,'after':b,'delta':b['miou']-a['miou']})
    ordered=sorted(rows,key=lambda r:r['delta']);chosen=ordered[-3:]+ordered[:2]
    for fn in [lambda r:0<r['after']['true_area']<=.01,lambda r:.01<r['after']['true_area']<=.05,lambda r:.99<=r['after']['true_area']<1]:
        rr=[r for r in rows if fn(r)];chosen.extend(rr[:1])
    chosen=list({r['name']:r for r in chosen}.values());ds=LocalisationDataset('data','train',methods=['pixelseal']);di={r['image_name']:i for i,r in enumerate(ds.meta)}
    for k,row in enumerate(chosen):
        n=row['name'];it=ds[di[n]];file=f'example_ps_{k}_{n}.png';gallery(out/file,it['image'],it['mask128'],new['probabilities'][ni[n]],old['probabilities'][oi[n]]);row['file']=file
    atomic_json(out/'gallery.json',chosen)
    lines=['# PixelSeal validation gallery','','All examples belong to the same57-image development fold. Selection intentionally includes gains, regressions, tiny watermark failures and clean holes; it is not a random sample.']
    for r in chosen:
        a,b=r['before'],r['after'];lines+=['',f"## {r['name']}",'',f"mIoU {a['miou']:.4f} → {b['miou']:.4f}; FG IoU {a['fg_iou']:.4f} → {b['fg_iou']:.4f}; clean IoU {a['clean_iou']:.4f} → {b['clean_iou']:.4f}; true watermark area {b['true_area']:.2%}.",'',f"![RGB GT probability prediction errors and baseline](results/base-candidate/{r['file']})"]
    (root/'GALLERY.md').write_text('\n'.join(lines)+'\n');print({'validation_examples':len(chosen)})


if __name__=='__main__':main()
