"""Compare the fixed PIPE endpoint against initial and matched old replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.native_localizer.common import atomic_json


def compare(a,b):
    lookup={r['image_name']:r for r in a['per_image']}
    assert set(lookup)=={r['image_name'] for r in b['per_image']}
    delta=np.array([r['miou']-lookup[r['image_name']]['miou'] for r in b['per_image']])
    rng=np.random.default_rng(42);ci=np.quantile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975])
    small=lambda r:np.mean([v['fg_iou'] for v in r['per_image'] if 0<v['true_area']<=.05])
    guard=small(b)>=small(a)-.01 and b['summary']['wm_over50']['clean_iou']>=a['summary']['wm_over50']['clean_iou']-.01
    return {'delta':float(delta.mean()),'ci95':ci.tolist(),'minority_guard':bool(guard),'passed':bool(delta.mean()>=.005 and ci[0]>0 and guard),'improved':int((delta>0).sum()),'worse':int((delta<0).sum())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'data-expansion/runs'
    load=lambda arm,step:json.loads((out/arm/f'evaluation-{step}.json').read_text())
    base=load('old',0);old=load('old',200);pipe=load('pipe',200)
    a1=compare(base,pipe);a2=compare(old,pipe)
    result={'baseline':base['summary'],'old_replay':old['summary'],'pipe':pipe['summary'],'pipe_vs_initial':a1,'pipe_vs_old':a2,'selected':bool(a1['passed'] and a2['passed']),'protocol':'fixed200 endpoint, reused57-image fold, matched replay exposure; no public score selection'}
    atomic_json(out/'evaluation.json',result);print('PIPE_FINAL',json.dumps(result),flush=True)


if __name__=='__main__':main()
