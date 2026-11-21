"""Refit selected scaling recipes; choose seed ensemble only on saved real OOF."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SCALING=HERE.parent/'synthetic_scaling'
sys.path.insert(0,str(SCALING))
import train as legacy
OUT=HERE/'results'
SEEDS=(42,7,123)
VARIANTS={'pixelseal':'n4k_s2500','trustmark':'n4k_s5000'}

def main():
    OUT.mkdir(exist_ok=True)
    legacy.setup('cuda:0')
    started=time.monotonic()
    plan={}
    # Freeze ensemble choice on the saved validation predictions, before test access.
    for family,variant in VARIANTS.items():
        directory=SCALING/'results/runs'/family
        seed_scores={s:[] for s in SEEDS}; ensemble_scores=[]; details={}
        for fold in (1,2,3):
            zs=[np.load(directory/f'{variant}_fold{fold}_seed{s}.npz') for s in SEEDS]
            assert all(np.array_equal(zs[0]['names'],z['names']) and np.array_equal(zs[0]['target'],z['target']) for z in zs)
            for seed,z in zip(SEEDS,zs):
                m=legacy.cv.model.metrics(z['probability'],z['target'],float(z['threshold']))
                seed_scores[seed].extend(m['per_image'])
            p=np.mean([z['probability'] for z in zs],axis=0)
            t=float(np.mean([float(z['threshold']) for z in zs]))
            ensemble_scores.extend(legacy.cv.model.metrics(p,zs[0]['target'],t)['per_image'])
        mean_single=float(np.mean([np.mean(v) for v in seed_scores.values()]))
        ensemble=float(np.mean(ensemble_scores))
        selected=list(SEEDS) if ensemble>=mean_single else [42]
        for seed in selected:
            reports=[json.loads((directory/f'{variant}_fold{f}_seed{seed}.json').read_text()) for f in (1,2,3)]
            source=directory/f'{variant}_seed{seed}_pretrain.pt'
            details[str(seed)]={'epochs':int(np.median([r['epoch'] for r in reports])),
                'threshold':float(np.median([r['threshold'] for r in reports])),
                'pretrain_sha256':legacy.sha(source),
                'fold_reports_sha256':[legacy.sha(directory/f'{variant}_fold{f}_seed{seed}.json') for f in (1,2,3)]}
        plan[family]={'variant':variant,'seeds':selected,'per_seed':details,
                      'mean_single_seed_oof_miou':mean_single,'ensemble_oof_miou':ensemble,
                      'selected_oof_miou':ensemble if len(selected)>1 else float(np.mean(seed_scores[42])),
                      'ensemble_rule':'mean probability >= mean member threshold; use 3 seeds if saved OOF improves mean single, else fixed seed42',
                      'schedule':'per-seed median inner-selected epochs and thresholds across 3 folds; full real refit from synthetic checkpoint'}
    plan_path=OUT/'frozen_recipe.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:legacy.save(plan_path,plan)
    print('FROZEN_PLAN',json.dumps(plan),flush=True)
    # All refits finish before reading test features.
    for family,recipe in plan.items():
        data=legacy.load_real(family,'cuda:0'); ids=np.arange(170)
        assert len(set(data.names))==170
        for seed in recipe['seeds']:
            cfg=recipe['per_seed'][str(seed)]; path=OUT/f'{family}_seed{seed}_full.pt'
            if path.exists():continue
            source=SCALING/f'results/runs/{family}/{recipe["variant"]}_seed{seed}_pretrain.pt'
            state=torch.load(source,map_location='cpu',weights_only=True)['state_dict']
            head,opt,rng=legacy.initialize(data,seed,state)
            history=[]
            for e in range(cfg['epochs']):
                history.append(legacy.epoch(head,opt,rng,data,ids))
                print('REFIT',family,seed,e+1,cfg['epochs'],flush=True)
            payload={'state_dict':legacy.state_cpu(head),'channels':data.channels,'seed':seed,'family':family,
                     'threshold':cfg['threshold'],'epoch':cfg['epochs'],'train_names':data.names,'losses':history,'pretrain_sha256':cfg['pretrain_sha256']}
            torch.save(payload,path)
            del head,opt
        del data
        torch.cuda.empty_cache()
    metadata=json.loads((ROOT/'data/test/metadata.json').read_text())
    summary={}
    for family,recipe in plan.items():
        cache_path=HERE.parent/f'frozen_extractor/results/full_train/test_features/{family}_features.pt'
        cache=torch.load(cache_path,map_location='cpu',weights_only=True)
        expected=[r['image_name'] for r in metadata if r['method']==family]
        assert len(cache['names'])==len(set(cache['names']))==340 and set(cache['names'])==set(expected)
        data=legacy.Data([f.to('cuda:0') for f in cache['features']],torch.zeros(340,128,128,device='cuda:0'),'cuda:0',cache['names'])
        probs=[]; checkpoints={}
        for seed in recipe['seeds']:
            path=OUT/f'{family}_seed{seed}_full.pt'; ck=torch.load(path,map_location='cpu',weights_only=True)
            head,_,_=legacy.initialize(data,seed,ck['state_dict'])
            probs.append(legacy.predict(head,data,np.arange(340)).cpu().numpy()); checkpoints[str(seed)]=legacy.sha(path)
            del head
        probability=np.mean(probs,axis=0); threshold=float(np.mean([recipe['per_seed'][str(s)]['threshold'] for s in recipe['seeds']]))
        masks=torch.from_numpy((probability>=threshold).astype(np.uint8))
        path=OUT/f'{family}-subset.pt'
        if path.exists():raise FileExistsError(path)
        torch.save({'image_names':cache['names'],'masks':masks},path)
        np.savez_compressed(OUT/f'{family}-test.npz',names=cache['names'],probability=probability,threshold=threshold)
        summary[family]={'n':340,'threshold':threshold,'variant':recipe['variant'],'seeds':recipe['seeds'],'checkpoints':checkpoints,
                         'test_cache_sha256':legacy.sha(cache_path),'subset_sha256':legacy.sha(path),'predicted_area':float(masks.float().mean())}
        print('PREDICTIONS_READY',family,flush=True)
        del data,cache;torch.cuda.empty_cache()
    legacy.save(OUT/'prediction_summary.json',{'methods':summary,'seconds':time.monotonic()-started,'recipe_sha256':legacy.sha(plan_path),'test_selection':False})
if __name__=='__main__':main()
