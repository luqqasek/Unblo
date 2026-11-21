"""Sequential loss screening with conditional three-fold CV."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('head_ablation', HERE.parent/'head_ablation/run.py')
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)
cv = ab.cv
OUT = HERE/'results'
ab.OUT = OUT
VARIANTS = ('bce_only', 'bce75_dice25', 'balanced_bce_mix', 'focal1_dice', 'focal2_dice')
FORMULAS = {
    'bce_only': 'BCE',
    'bce75_dice25': '0.75 BCE + 0.25 foreground Dice',
    'balanced_bce_mix': '0.5 pixel-mean BCE + 0.5 per-image class-mean BCE',
    'focal1_dice': '0.5 focal(gamma=1, no alpha weighting) + 0.5 foreground Dice',
    'focal2_dice': '0.5 focal(gamma=2, no alpha weighting) + 0.5 foreground Dice',
}


def class_bce(logits, target):
    pixels = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    means, present = [], []
    for mask in (target, 1-target):
        count = mask.sum((1,2))
        means.append((pixels*mask).sum((1,2))/count.clamp_min(1))
        present.append((count>0).to(pixels.dtype))
    return ((means[0]+means[1])/(present[0]+present[1])).mean()


def loss(variant, logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if variant=='bce_only':
        return bce
    if variant=='balanced_bce_mix':
        return .5*bce + .5*class_bce(logits,target)
    p = logits.sigmoid()
    dice = ab.dice_loss(p,target)
    if variant=='bce75_dice25':
        return .75*bce+.25*dice
    gamma = {'focal1_dice':1, 'focal2_dice':2}[variant]
    pt = p*target+(1-p)*(1-target)
    focal = ((1-pt).pow(gamma)*F.binary_cross_entropy_with_logits(logits,target,reduction='none')).mean()
    return .5*focal+.5*dice


def diagnose(row):
    path = OUT/row['variant']/f"{row['method']}_fold{row['fold']}.npz"
    with np.load(path) as z:
        p=z['probability']>=row['threshold']; g=z['target'].astype(bool)
        fp=(p & ~g).sum((1,2)); fn=(~p & g).sum((1,2))
        npos=g.sum((1,2)); nneg=(~g).sum((1,2))
        row['diagnostics']=dict(fp_pixels=int(fp.sum()), fn_pixels=int(fn.sum()),
            fpr=float(fp.sum()/max(1,nneg.sum())), fnr=float(fn.sum()/max(1,npos.sum())),
            predicted_fraction=float(p.mean()), target_fraction=float(g.mean()))
        area=g.mean((1,2)); values=np.asarray(row['selected']['per_image'])
        row['diagnostics']['area_bins']={}
        for name,mask in [('under10',area<.1),('10to90',(area>=.1)&(area<=.9)),('over90',area>.9)]:
            row['diagnostics']['area_bins'][name]=dict(n=int(mask.sum()),miou=float(values[mask].mean()) if mask.any() else None)
        np.testing.assert_allclose(cv.model.metrics(z['probability'],z['target'],row['threshold'])['per_image'],values,atol=1e-12)
    return row


def publish(summary):
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
    subprocess.run([sys.executable,str(cv.model.ROOT/'dashboard/build.py')],check=True)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    ds=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    splits=json.loads((cv.OUT/'folds.json').read_text())
    baseline=json.loads((cv.OUT/'metrics.json').read_text())
    summary=dict(gate='Screen fold delta >= 0.005 and improvements in at least two methods',
                 formulas=FORMULAS, variants={}, baseline=baseline, status='running')
    for variant in VARIANTS:
        fn=lambda logits,target: loss(variant,logits,target)
        rows=[diagnose(ab.fit(variant,m,0,ds,splits,fn)) for m in cv.METHODS]
        delta=[r['selected']['miou']-baseline['methods'][r['method']]['folds'][0]['selected']['miou'] for r in rows]
        passed=bool(np.mean(delta)>=.005 and sum(d>0 for d in delta)>=2)
        summary['variants'][variant]=dict(screen=rows.copy(),screen_delta=float(np.mean(delta)),passed=passed,cv=[])
        print('GATE',variant,passed,float(np.mean(delta)),flush=True)
        publish(summary)
        if passed:
            for fold in (1,2):
                rows.extend(diagnose(ab.fit(variant,m,fold,ds,splits,fn)) for m in cv.METHODS)
            summary['variants'][variant]['cv']=rows
            publish(summary)
    summary['status']='complete'
    publish(summary)


if __name__=='__main__':
    main()
