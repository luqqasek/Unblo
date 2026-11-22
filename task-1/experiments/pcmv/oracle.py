"""GT-only proposal coverage audit; no inference policy or feature parameters changed."""
import argparse,json
from pathlib import Path
import numpy as np
from .proposals import exact_oracle
from experiments.native_localizer.common import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();tasks=[t for t in json.loads((a.out/'plan.json').read_text())['tasks'] if t['role']=='validation'];assert len(tasks)==57;rows=[]
    for t in tasks:
        with np.load(a.out/'features'/(t['id']+'.npz')) as d:
            m=np.unpackbits(d['masks'],axis=-1).astype(bool);gt=d['gt'];b=d['base']>=.5;r=exact_oracle(b,m,gt);r.update(image_name=t['id'],area=float(gt.mean()),proposals=len(m));minority=gt if gt.mean()<=.5 else ~gt;union=(m|minority).sum((1,2));r['best_minority_overlap']=float(np.max((m&minority).sum((1,2))/union));rows.append(r)
    filters={'all':lambda r:True,'fg_le1':lambda r:0<r['area']<=.01,'fg_1to5':lambda r:.01<r['area']<=.05,'clean_le1':lambda r:0<1-r['area']<=.01,'clean_1to5':lambda r:.01<1-r['area']<=.05};summary={}
    for name,fn in filters.items():
        rr=[r for r in rows if fn(r)];summary[name]={'n':len(rr),**{k:float(np.mean([r[k] for r in rr])) if rr else None for k in ('base','one_toggle','two_disjoint_toggles','best_minority_overlap')},'overlap_at_least_half':sum(r['best_minority_overlap']>=.5 for r in rr)}
    result={'diagnostic_only':True,'not_achievable_score':True,'no_recipe_changes_from_audit':True,'summary':summary,'per_image':rows};atomic_json(a.out/'oracle.json',result);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
