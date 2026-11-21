"""Compare message correction with the newer stable StegaStamp selection protocol."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from prepare import HERE,OUT,cv,prepare,prepare_bits

spec=importlib.util.spec_from_file_location('evidence_run',HERE/'run.py')
experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)
spec=importlib.util.spec_from_file_location('stable_selection',HERE.parent/'stegastamp_stable_selection/run.py')
stable=importlib.util.module_from_spec(spec);spec.loader.exec_module(stable)


def main():
    torch.set_num_threads(4)
    evidence=prepare_bits('stegastamp',prepare('stegastamp'))
    cache=torch.load(cv.model.OUT/'stegastamp_features.pt',weights_only=True)
    names=cache['names']; features=cache['features']+[evidence['perbit']]
    ds=cv.model.LocalisationDataset(cv.model.ROOT/'data','train')
    gt=np.stack([ds.m128[n] for n in names]); target=torch.from_numpy(gt).float()
    meta={m['image_name']:m for m in ds.meta}; flags=np.array([meta[n]['full_watermark'] for n in names])
    splits=json.loads((cv.OUT/'folds.json').read_text())['stegastamp']
    baseline=json.loads((stable.OUT/'metrics.json').read_text())['runs']
    baseline={r['fold']:r for r in baseline if r['seed']==42}
    dest=OUT/'stable'; dest.mkdir(exist_ok=True)
    summary=dict(status='running',seed=42,runs=[],gate='Fold1 delta>=.005 and real vs wrong message >=.005, then remaining folds')
    def init():
        torch.manual_seed(42)
        h=experiment.EvidenceHead([f.shape[1] for f in cache['features']],'bit_residual',100)
        return h,torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001),np.random.default_rng(42)
    for fold,split in enumerate(splits,1):
        path=dest/f'fold{fold}.json'
        if path.exists():
            r=json.loads(path.read_text())
        else:
            fit=np.array([names.index(n) for n in split['train']]); val=np.array([names.index(n) for n in split['validation']])
            parts=stable.inner_folds(fit,gt,flags)
            models=[init() for _ in parts]; trains=[np.setdiff1d(fit,p) for p in parts]
            best=-1;stale=0;history=[]
            for epoch in range(1,51):
                probs=[]
                for (h,opt,rng),train,valid in zip(models,trains,parts):
                    assert not set(train)&set(valid) and not set(train)&set(val)
                    cv.train_epoch(h,opt,rng,features,target,train)
                    probs.append(cv.predict(h,features,valid))
                pooled=np.concatenate(probs); inner_gt=gt[np.concatenate(parts)]
                score=cv.model.metrics(pooled,inner_gt,.5)['miou'];history.append(score)
                if score>best: best=score;stale=0;chosen=epoch;best_prob=pooled.copy()
                else:stale+=1
                if epoch%5==0:print('stable',fold,'epoch',epoch,score,flush=True)
                if stale>=8:break
            threshold=max(cv.model.CONFIG['threshold_candidates'],key=lambda t:cv.model.metrics(best_prob,inner_gt,t)['miou'])
            h,opt,rng=init()
            for _ in range(chosen):cv.train_epoch(h,opt,rng,features,target,fit)
            prob=cv.predict(h,features,val)
            wrong=cv.predict(h,cache['features']+[evidence['wrong_perbit']],val)
            result=cv.model.metrics(prob,gt[val],threshold)
            counter=cv.model.metrics(wrong,gt[val],threshold)
            r=dict(fold=fold,seed=42,epoch=chosen,threshold=threshold,selected=result,counterfactual=counter,
                   baseline=baseline[fold]['miou'],delta=result['miou']-baseline[fold]['miou'],history=history,
                   train_names=split['train'],val_names=split['validation'],inner_val_names=[[names[i] for i in p] for p in parts])
            torch.save(dict(state_dict=h.state_dict(),channels=[f.shape[1] for f in cache['features']],evidence_channels=100,variant='bit_residual',epoch=chosen,threshold=threshold),path.with_suffix('.pt'))
            np.savez_compressed(path.with_suffix('.npz'),probability=prob,counterfactual=wrong,target=gt[val],names=split['validation'])
            path.write_text(json.dumps(r,indent=2),encoding='utf-8')
        summary['runs'].append(r)
        if fold==1:
            summary['passed']=r['delta']>=.005 and r['selected']['miou']-r['counterfactual']['miou']>=.005
        (OUT/'stable_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('STABLE RESULT',fold,r['selected']['miou'],'baseline',r['baseline'],'delta',r['delta'],'passed',summary['passed'],flush=True)
        subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
        subprocess.run([sys.executable,str(cv.model.ROOT/'dashboard/build.py')],check=True)
        if not summary['passed']:break
    summary['status']='complete';(OUT/'stable_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
    subprocess.run([sys.executable,str(cv.model.ROOT/'dashboard/build.py')],check=True)


if __name__=='__main__':main()
