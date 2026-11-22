"""Latest OOF diagnostics; no fitting, oracle results explicitly labeled."""
import json
from pathlib import Path
import numpy as np
from experiments.small_region_diagnosis.common import metadata, select, bucket, mean, save, sha
from experiments.small_region_diagnosis.oof import exact_oracle

def main():
    here=Path(__file__).resolve().parent;root=here.parent/'region_data/results/runs'
    meta=metadata();out={}
    for family,variant in [('pixelseal','control-adapt'),('trustmark','objects-adapt')]:
        rows=[];probs={};targets={};hashes={}
        for fold in (1,2,3):
            p=root/family/variant/f'fold{fold}'
            split=json.loads((p/'split.json').read_text());assert not set(split['train'])&set(split['validation'])
            saved={r['image_name']:r for r in json.loads((p/'per_image.json').read_text())}
            with np.load(p/'maps.npz') as z:
                assert set(z['names'])==set(split['validation'])
                for n,prob,y in zip(z['names'],z['probabilities'],z['targets']):
                    n=str(n);prob=prob.squeeze();y=y.squeeze().astype(bool)
                    assert prob.dtype==np.float32 and n not in probs
                    probs[n]=prob;targets[n]=y
                    rows.append(dict(saved[n],bucket=bucket(float(y.mean())),full_watermark=meta[n]['full_watermark'],
                        max_prob_gt=float(prob[y].max()),max_prob_bg=float(prob[~y].max())))
            hashes[str(p/'maps.npz')]=sha(p/'maps.npz')
        assert len(rows)==170
        groups={}
        for b in ['all','le1','1to5','5to20','gt20','clean_holes']:
            rr=select(rows,b);g={'n':len(rr),**{k:mean(rr,k) for k in ['miou','fg_iou','clean_iou','roc_auc','fg_ap','empty','max_prob_gt','max_prob_bg']}}
            if b in ['le1','1to5']:
                g['oracle_bucket_fg']=exact_oracle([probs[r['image_name']] for r in rr],[targets[r['image_name']] for r in rr])
            groups[b]=g
        out[family]={'variant':variant,'groups':groups,'hashes':hashes}
        print(family,json.dumps(groups),flush=True)
    save(here/'results/latest_oof_diagnosis.json',out)
if __name__=='__main__':main()
