"""Single fit per predeclared treatment/control; inner-only action calibration."""
import json,time,pickle,os,subprocess,fcntl
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from experiments.native_localizer.common import atomic_json
from experiments.native_localizer.evaluate import image_metrics
from experiments.triforensic_wm.evaluate import summarize
from .proposals import compose,metrics,exact_oracle


def load(out,skip_audit=False):
    plan=json.loads((out/'plan.json').read_text());items=[]
    for task in plan['tasks']:
        if skip_audit and task['role']=='audit_control':continue
        path=out/'features'/(task['id']+'.npz');d=dict(np.load(path));meta=json.loads(path.with_suffix('.json').read_text());meta['base_description']=meta.pop('base');d.update(meta);d['masks']=np.unpackbits(d['masks'],axis=-1).astype(bool)
        if task['kind']=='real':assert meta.get('signed_bits_fixed') is True,('Real-bit sign repair incomplete',task['id'])
        from .native_probe import KEYS
        with np.load(out/'native512'/(task['id']+'.npz')) as native:
            for k in ('features','wrong'):d[k]=np.column_stack([d[k],native[k]])
        d['feature_keys']+=KEYS;items.append(d)
    assert len({tuple(d['feature_keys']) for d in items})==1
    return items


def ci(delta):
    delta=np.asarray(delta);rng=np.random.default_rng(42);return np.quantile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975]).tolist()


def predict(items,models,indices,wrong=False):
    return [np.column_stack([m.predict(d['wrong' if wrong else 'features'][:,indices]) for m in models]) for d in items]


def outcomes(items,predictions,threshold):
    rows=[];actions=[];maps=[]
    for d,gains in zip(items,predictions):
        mask,act=compose(d['base']>=.5,d['masks'],gains,threshold,max_toggles=1);p=d['base'].copy()
        for a in act:p[d['masks'][a['proposal']]]=1. if a['action']=='add' else 0.
        assert np.array_equal(p>=.5,mask)
        r=image_metrics(p,d['gt']);r['image_name']=d['id'];yy,xx=np.where(d['gt']);fill=len(yy)/((yy.max()-yy.min()+1)*(xx.max()-xx.min()+1)) if len(yy) else 0.;r['shape']='rectangular' if fill>=.98 else 'irregular';rows.append(r);actions.append(act);maps.append(p)
    return rows,actions,maps


def control_audit(items,predictions,threshold):
    scores=[];rates={'clean':[],'marked':[]};low_obs=[]
    for d,g in zip(items,predictions):
        b=d['base']>=.5;m,actions=compose(b,d['masks'],g,threshold,max_toggles=1);scores.append(metrics(m,d['gt'])-metrics(b,d['gt']));positive=bool(d['marked']);bad=any((a['action']=='carve') if positive else (a['action']=='add') for a in actions);rates['marked' if positive else 'clean'].append(bad)
        if positive:
            q=np.quantile(d['obs'],.25);low=d['obs']<=q
            low_obs.append(bool(np.any(low&b&~m)))
    return {'clean_false_toggle_rate':float(np.mean(rates['clean'])),'marked_false_toggle_rate':float(np.mean(rates['marked'])),'mean_miou_delta':float(np.mean(scores)),'mean_miou_drop':float(np.mean(np.maximum(-np.array(scores),0))),'low_observability_false_carve_image_rate':float(np.mean(low_obs)),'images':len(items)}


def publish(out,result,core_only=False):
    a=result['variants']['pcmv'];gates=a['anti_shortcut'];pending=gates.get('pending',False)
    result['candidate_for_further_inference']=bool(not pending and a['delta_vs_base']>.01 and a['ci95_vs_base'][0]>0 and a['delta_vs_control_b']>0 and gates['clean_false_toggle_rate']<.02 and gates['marked_false_toggle_rate']<.02 and gates['mean_miou_drop']<.003)
    result['selection_rule']='Gain>0.01 vs best base, paired95% lower bound>0, beats proposal-only control, clean/full twin false-toggle<2% and mean mIoU drop<0.003; fixed before scorer fitting/evaluation.'
    atomic_json(out/('core_evaluation.json' if core_only else 'evaluation.json'),result)
    if not core_only and os.environ.get('SLURM_JOB_ID'):
        gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        other={line.split(',')[1].strip() for line in gpu_info.splitlines() if int(line.split(',')[0]) in (2,3)}
        while True:
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            if not any(line.split(',')[0].strip() in other for line in apps.splitlines()):break
            print('PCMV_DONE_HOLDING_SHARED_ALLOCATION_FOR_USER_GPUS_2_3',flush=True);time.sleep(15)


def fit(out,core_only=False):
    # The early CPU fit and final GPU-worker completion can meet at the same
    # instant. Serialize them so each scorer is fitted exactly once.
    with (out/'scorer-fit.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'evaluation.json').exists():return
        return _fit(out,core_only)


def _fit(out,core_only=False):
    from .native_probe import ensure
    if (out/'repair-required.json').exists():
        deadline=time.monotonic()+600
        while not all((out/f'repair-{rank}.json').exists() for rank in (0,1)):
            assert time.monotonic()<deadline,'Real feature repair did not complete'
            time.sleep(1)
    if (out/'core_evaluation.json').exists():
        if core_only:raise RuntimeError('Core fit already completed; no second training')
        ensure(out);data=load(out);audit=[d for d in data if d['role']=='audit_control'];result=json.loads((out/'core_evaluation.json').read_text())
        for name,v in result['variants'].items():
            with (out/f'{name}.pkl').open('rb') as f:checkpoint=pickle.load(f)
            v['anti_shortcut']=control_audit(audit,predict(audit,checkpoint['models'],checkpoint['indices']),checkpoint['threshold'])
        result['finalization']='Reused the single fit and final57-image evaluation; only the400-control audit ran afterward. No validation refit/re-evaluation.';publish(out,result);return
    if not core_only:ensure(out)
    start=time.monotonic();data=load(out,skip_audit=core_only);train=[d for d in data if d['role']=='train'];cal=[d for d in data if d['role']=='calibration'];cc=[d for d in data if d['role']=='calibration_control'];valid=[d for d in data if d['role']=='validation'];audit=[d for d in data if d['role']=='audit_control'];assert len(valid)==57 and len(cal)==23
    x=np.concatenate([d['features'] for d in train]);y=np.concatenate([d['gains'] for d in train]);weights=np.concatenate([np.full(len(d['features']),(3 if d['kind']=='real' else 1)/len(d['features'])) for d in train]);base_rows=[dict(image_metrics(d['base'],d['gt']),image_name=d['id']) for d in valid];base=np.array([r['miou'] for r in base_rows]);result={'protocol':'Single repeatedly used PixelSeal57-image developmental fold;90 real scorer training,23 inner calibration;113-real-trained base used for all features (in-sample meta-training limitation). One fit and final evaluation; not full OOF. At most one toggle to avoid nonadditive two-action damage.','base':summarize(base_rows),'base_per_image':base_rows,'variants':{},'train_images':len(train),'proposal_train_examples':len(x)}
    for name,indices in [('control_b',data[0]['basic_indices']),('pcmv',list(range(x.shape[1])))]:
        models=[]
        for action in range(2):
            m=HistGradientBoostingRegressor(max_iter=220,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.,early_stopping=False,random_state=42);m.fit(x[:,indices],y[:,action],sample_weight=weights);models.append(m)
        cp=predict(cal,models,indices);control_preds=predict(cc,models,indices);choices=[]
        for threshold in [0,.002,.005,.01,.02,.03,.05,.08,.12,1.]:
            audit_cal=control_audit(cc,control_preds,threshold)
            if audit_cal['clean_false_toggle_rate']>=.02 or audit_cal['marked_false_toggle_rate']>=.02 or audit_cal['mean_miou_drop']>=.003:continue
            scores=[metrics(compose(d['base']>=.5,d['masks'],g,threshold,max_toggles=1)[0],d['gt']) for d,g in zip(cal,cp)];choices.append({'threshold':threshold,'miou':float(np.mean(scores)),'control_audit':audit_cal})
        chosen=max(choices,key=lambda r:(r['miou'],r['threshold']));threshold=chosen['threshold'];pr=predict(valid,models,indices);rows,acts,maps=outcomes(valid,pr,threshold);delta=np.array([r['miou'] for r in rows])-base;aud=control_audit(audit,predict(audit,models,indices),threshold) if audit else {'pending':True,'images':0}
        wrong_rows,_,_=outcomes(valid,predict(valid,models,indices,wrong=True),threshold)
        result['variants'][name]={'summary':summarize(rows),'delta_vs_base':float(delta.mean()),'ci95_vs_base':ci(delta),'per_image':rows,'actions':{d['id']:a for d,a in zip(valid,acts)},'calibration':chosen,'calibration_grid':choices,'anti_shortcut':aud,'wrong_message_miou':float(np.mean([r['miou'] for r in wrong_rows])),'wrong_message_note':'Only new message features swapped; base and proposal masks held fixed. Physical target unchanged.','feature_indices':indices}
        with (out/f'{name}.pkl').open('wb') as f:pickle.dump({'models':models,'indices':indices,'threshold':threshold,'feature_keys':data[0]['feature_keys']},f)
        np.savez_compressed(out/f'{name}-maps.npz',names=[d['id'] for d in valid],probabilities=np.array(maps,dtype=np.float32),targets=[d['gt'] for d in valid])
        print('PCMV_FINAL',name,result['variants'][name]['summary']['all']['miou'],float(delta.mean()),aud,flush=True)
    a=result['variants']['pcmv'];b=result['variants']['control_b'];d=np.array([r['miou'] for r in a['per_image']])-np.array([r['miou'] for r in b['per_image']]);a['delta_vs_control_b']=float(d.mean());a['ci95_vs_control_b']=ci(d)
    # GT diagnostic only: exact combination of confusion matrices, not gains.
    oracle=[dict(exact_oracle(item['base']>=.5,item['masks'],item['gt']),image_name=item['id']) for item in valid]
    result['oracle_per_image']=oracle;result['one_toggle_oracle_miou']=float(np.mean([r['one_toggle'] for r in oracle]));result['two_toggle_oracle_miou']=float(np.mean([r['two_disjoint_toggles'] for r in oracle]));result['oracle_note']='Exact<=2 disjoint proposal toggles, GT-only diagnostic; no-op included. Deployment is<=1 toggle. Never used to select feature/proposal parameters.';result['seconds_fit_evaluate']=time.monotonic()-start
    publish(out,result,core_only=core_only)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--core-only',action='store_true');a=p.parse_args();fit(a.out,core_only=a.core_only)
