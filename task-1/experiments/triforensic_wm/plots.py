"""Learning curves, paired image bootstrap and galleries from saved predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from experiments.native_localizer.evaluate import gallery
from experiments.native_localizer.common import atomic_json
from wmloc.data import LocalisationDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--data',type=Path,default=Path('data'));a=p.parse_args()
    base=json.loads((a.run/'real-baseline.json').read_text());end=json.loads((a.run/f'real-{a.step}.json').read_text());lookup={r['image_name']:r for r in base['per_image']}
    delta=np.array([r['miou']-lookup[r['image_name']]['miou'] for r in end['per_image']]);rng=np.random.default_rng(42);boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
    atomic_json(a.run/f'paired-change-{a.step}.json',{'n_unique_images':len(delta),'mean_delta':float(delta.mean()),'image_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),'improved':int((delta>1e-8).sum()),'worse':int((delta<-1e-8).sum()),'same':int((abs(delta)<=1e-8).sum()),'limitations':'Reused development fold, one seed, unadjusted exploratory interval. One prediction per image; no repeated seeds treated independently.'})
    rows=[json.loads(s) for s in (a.run/'train.jsonl').read_text().splitlines()];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key in ('seg_iou','paired','local','edge'):axes[0].plot([r['step'] for r in rows],[r['terms'][key] for r in rows],label=key)
    axes[0].set(xlabel='Optimizer update',ylabel='Weighted training loss',title='Training objectives');axes[0].legend(fontsize=8)
    checkpoints=sorted([json.loads(p.read_text()) for p in a.run.glob('real-*.json') if p.stem!='real-baseline'],key=lambda d:d['step'])
    axes[1].axhline(base['summary']['all']['miou'],color='black',linestyle='--',label='Original matched baseline')
    axes[1].plot([r['step'] for r in checkpoints],[r['summary']['all']['miou'] for r in checkpoints],'o-',label='EMA validation mIoU')
    axes[1].set(xlabel='Optimizer update',ylabel='Official mIoU',title=f'{a.family}:57 fixed validation images');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(a.run/'learning-curves.png',dpi=150);plt.close(fig)
    mp=a.run/f'maps-{a.step}.npz';bp=a.run/'maps-baseline.npz'
    if not mp.exists() or not bp.exists():return
    maps=np.load(mp);before=np.load(bp);bd={str(n):p for n,p in zip(before['names'],before['probabilities'])};ds=LocalisationDataset(a.data,'train',methods=[a.family]);di={r['image_name']:i for i,r in enumerate(ds.meta)};mi={str(n):i for i,n in enumerate(maps['names'])};out=a.run/f'gallery-{a.step}';out.mkdir(exist_ok=True)
    # Include all tiny cases plus largest gains/regressions and clean-hole failures.
    selected={r['image_name'] for r in end['per_image'] if 0<r['true_area']<=.01 or 0<1-r['true_area']<=.01}
    selected.update(end['per_image'][i]['image_name'] for i in np.argsort(delta)[:3]);selected.update(end['per_image'][i]['image_name'] for i in np.argsort(delta)[-3:]);er={r['image_name']:r for r in end['per_image']}
    for name in sorted(selected):
        idx=mi[name];item=ds[di[name]];target=out/f'example-{name}.png';gallery(target,item['image'],maps['targets'][idx],maps['probabilities'][idx],bd[name])
        im=Image.open(target);canvas=Image.new('RGB',(im.width,im.height+35),'white');canvas.paste(im,(0,0));draw=ImageDraw.Draw(canvas);r=er[name]
        draw.text((5,im.height+6),f"step{a.step} | {name} | mIoU {r['miou']:.4f} vs {lookup[name]['miou']:.4f} | FG {r['fg_iou']:.4f} clean {r['clean_iou']:.4f} | GT {100*r['true_area']:.2f}%",fill='black');canvas.save(target)
    print({'gallery_examples':len(selected),'mean_delta':float(delta.mean()),'CI':np.quantile(boot,[.025,.975]).tolist()})
if __name__=='__main__':main()
