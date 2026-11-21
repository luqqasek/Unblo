"""E1/E2, strictly nested train/calibration/refit; matched watermark-only controls."""
import pickle
import argparse
import time
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *
from evaluate import summarize, METRICS

def logit(p):
    p=np.clip(p,1e-5,1-1e-5);return np.log(p)-np.log1p(-p)

def pixel_features(w,f,control):
    a=logit(w)
    # Both have exactly three fitted coefficients including intercept.
    return np.stack([np.ones_like(a),a,a*a/5 if control else logit(f)],axis=-1)

def fit_e1(w,f,g,penalty,control):
    x=pixel_features(w,f,control).reshape(len(w),-1,3);y=g.reshape(len(w),-1)
    rng=np.random.default_rng(42);ix=rng.choice(x.shape[1],1024,replace=False)
    x=x[:,ix].reshape(-1,3).astype('float64');y=y[:,ix].ravel().astype(float)
    anchor=np.array([0.,1.,0.])
    def objective(beta):
        z=x@beta;d=beta-anchor; loss=np.mean(np.logaddexp(0,z)-y*z)+penalty*np.dot(d,d)/2
        grad=x.T@(expit(z)-y)/len(y)+penalty*d
        return loss,grad
    result=minimize(objective,anchor,jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-10})
    assert result.success,result.message
    return result.x

def predict_e1(beta,w,f,control):return expit(pixel_features(w,f,control)@beta).astype('float32')

def map_stats(p):
    p=np.clip(p,1e-6,1-1e-6);flat=p.reshape(len(p),-1)
    return np.column_stack([flat.mean(1),flat.std(1),np.quantile(flat,[.1,.5,.9],axis=1).T,
                            (flat>=.5).mean(1),(-(flat*np.log(flat)+(1-flat)*np.log1p(-flat))).mean(1)])

def image_features(w,f,bits,control):
    ws=map_stats(w);fs=map_stats(f)
    # Same nominal number of coefficients; control adds watermark-only moments.
    if control:fs=ws**2
    return np.column_stack([ws,fs,bits]).astype('float64')

def labels(w,f,g,control):
    a=np.zeros_like(w) if control else f;b=np.ones_like(w) if control else 1-f
    result=[]
    for wi,ai,bi,gi in zip(w,a,b,g):
        scores=[curve(p,gi,[.5])[0] for p in (wi,ai,bi)]
        j=int(np.argmax(scores));result.append(j if scores[j]>scores[0]+.02 else 0)
    return np.array(result)

def fit_e2(w,f,g,bits,c,control):
    x=image_features(w,f,bits,control);scaler=StandardScaler().fit(x);y=labels(w,f,g,control)
    if len(np.unique(y))==1:return dict(constant=int(y[0]),scaler=scaler)
    clf=LogisticRegression(C=c,max_iter=2000,random_state=42).fit(scaler.transform(x),y)
    return dict(clf=clf,scaler=scaler)

def predict_e2(model,w,f,bits,control,alpha):
    probs=np.zeros((len(w),3))
    if 'constant' in model:probs[:,model['constant']]=1
    else:
        clf=model['clf'];probs[:,clf.classes_]=clf.predict_proba(model['scaler'].transform(image_features(w,f,bits,control)))
    a=np.zeros_like(w) if control else f;b=np.ones_like(w) if control else 1-f
    mixed=(1-alpha*(1-probs[:,0,None,None]))*w+alpha*(probs[:,1,None,None]*a+probs[:,2,None,None]*b)
    return mixed.astype('float32'),probs

def calibrate(p,g):
    cuts=np.linspace(.05,.95,19,dtype='float32');scores=np.mean([curve(a,b,cuts) for a,b in zip(p,g)],axis=0)
    j=int(scores.argmax());return float(scores[j]),float(cuts[j])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-inputs',action='store_true');args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    assert json.loads((OUT/'eval_detectivesam/gate.json').read_text())['pass']
    records,base,gt,splits,_=cohort();names=[r['image_id'] for r in records];idx={n:i for i,n in enumerate(names)}
    f=np.stack([np.load(OUT/'detectivesam'/f'{n}.npz')['probability'] for n in names])
    bits={}
    import torch
    for family in METHODS:
        z=torch.load(HERE.parent/'frozen_extractor/results'/f'{family}_features.pt',map_location='cpu',weights_only=True)
        bits.update(dict(zip(z['names'],np.asarray(z['bit_accuracy']))))
    variants=('E1_control','E1_forensic','E2_control','E2_forensic')
    rows=[];configs=[];out=OUT/'fusion';out.mkdir(exist_ok=True)
    for family in METHODS:
        for fold,s in enumerate(splits[family],1):
            inputpath=out/f'{family}_fold{fold}_inputs.npz'
            if args.wait_inputs:
                if not inputpath.exists():print('WAIT_INPUTS',family,fold,flush=True)
                while not inputpath.with_suffix('.json').exists():time.sleep(5)
            inputs=np.load(inputpath)
            ds={}
            for k in ('fit','cal','refit'):
                ns=inputs[k+'_names'].tolist();ids=np.array([idx[n] for n in ns])
                assert not set(ns)&set(s['validation'])
                ds[k]=(inputs[k+'_probability'],f[ids],gt[ids],np.array([bits[n] for n in ns]),ns)
            assert not set(ds['fit'][4])&set(ds['cal'][4])
            assert set(ds['refit'][4])==set(s['train'])
            val=np.array([idx[n] for n in s['validation']]);bw=base[val];bf=f[val];bb=np.array([bits[n] for n in s['validation']])
            for variant in variants:
                file=out/f'{family}_fold{fold}_{variant}.npz';configpath=file.with_suffix('.json')
                control=variant.endswith('control');e1=variant.startswith('E1')
                if file.exists():
                    saved=np.load(file);pred=saved['probability'];threshold=float(saved['threshold']);config=json.loads(configpath.read_text())
                else:
                    wt,ft,yt,bt,_=ds['fit'];wc,fc,yc,bc,_=ds['cal'];wr,fr,yr,br,_=ds['refit']
                    candidates=[]
                    for reg in ([.001,.01,.1] if e1 else [.01,.1,1.]):
                        model=fit_e1(wt,ft,yt,reg,control) if e1 else fit_e2(wt,ft,yt,bt,reg,control)
                        for alpha in ([1.] if e1 else [.25,.5,1.]):
                            cp=predict_e1(model,wc,fc,control) if e1 else predict_e2(model,wc,fc,bc,control,alpha)[0]
                            score,threshold=calibrate(cp,yc)
                            candidates.append(dict(reg=reg,alpha=alpha,inner_miou=score,threshold=threshold))
                    best=max(candidates,key=lambda r:r['inner_miou']);threshold=best['threshold']
                    fitted=fit_e1(wr,fr,yr,best['reg'],control) if e1 else fit_e2(wr,fr,yr,br,best['reg'],control)
                    if e1:pred=predict_e1(fitted,bw,bf,control);gate=np.empty((len(val),0))
                    else:pred,gate=predict_e2(fitted,bw,bf,bb,control,best['alpha'])
                    config=dict(family=family,fold=fold,variant=variant,selected=best,candidates=candidates,train_names=s['train'],validation_names=s['validation'],
                                calibration_names=ds['cal'][4],fusion_fit_names=ds['fit'][4],
                                coefficients=fitted.tolist() if e1 else None,parameter_count=3 if e1 else 3*16,
                                protocol='strict inner cross-fitting; inner meta-validation excluded from all fit input producers; outer GT unused')
                    np.savez_compressed(file,names=s['validation'],probability=pred,threshold=threshold,gate_probability=gate)
                    with file.with_suffix('.pkl').open('wb') as handle:pickle.dump(fitted,handle)
                    save(configpath,config)
                configs.append(config)
                for j,i in enumerate(val):rows.append(dict(**records[i],model='detectivesam_fusion',variant=variant,**measure(pred[j],gt[i],threshold)))
                print(f'FUSION {family} fold{fold} {variant}: mIoU={np.mean([r["miou"] for r in rows if r["family"]==family and r["fold"]==fold and r["variant"]==variant]):.5f} t={threshold}',flush=True)
    for i,r in enumerate(records):rows.append(dict(**r,model='detectivesam_fusion',variant='baseline',**measure(base[i],gt[i],r['baseline_threshold'])))
    df=pd.DataFrame(rows);df.to_csv(out/'per_image_metrics.csv',index=False);summ=summarize(df);summ.to_csv(out/'summary_by_family_area.csv',index=False)
    folds=df.groupby(['family','fold','variant'])[METRICS].mean().reset_index();folds.to_csv(out/'per_fold.csv',index=False)
    comparisons=[];deltas=[]
    for family in ['all',*METHODS]:
        frame=df if family=='all' else df[df.family==family]
        for a,b in [('E1_forensic','baseline'),('E1_forensic','E1_control'),('E2_forensic','baseline'),('E2_forensic','E2_control')]:
            aa=frame[frame.variant==a].set_index('image_id');bb=frame[frame.variant==b].set_index('image_id').loc[aa.index]
            d=aa.miou-bb.miou
            fd=(aa.groupby('fold').miou.mean()-bb.groupby('fold').miou.mean()).to_dict()
            comparisons.append(dict(family=family,variant=a,control=b,delta=float(d.mean()),ci95=ci(d),fold_deltas=fd,
                                    improved=int((d>0).sum()),damaged=int((d<0).sum()),
                                    baseline_failures_improved=int(((bb.miou<.4)&(d>.05)).sum()),previously_good_damaged=int(((bb.miou>=.5)&(d<-.05)).sum())))
            if family=='all':
                for n in aa.index:deltas.append(dict(image_id=n,variant=a,control=b,delta_miou=d.loc[n],family=aa.loc[n,'family'],fold=int(aa.loc[n,'fold'])))
    pd.DataFrame(deltas).to_csv(out/'paired_deltas.csv',index=False)
    save(out/'comparisons.json',comparisons);save(out/'configurations.json',configs)
    # E3 only after simple models improve true ranking and mIoU relative to control.
    overall=summ[(summ.family=='all')&(summ.stratum=='all')].set_index('variant')
    promising=[]
    for variant,control in [('E1_forensic','E1_control'),('E2_forensic','E2_control')]:
        c=next(r for r in comparisons if r['family']=='all' and r['variant']==variant and r['control']==control)
        cbase=next(r for r in comparisons if r['family']=='all' and r['variant']==variant and r['control']=='baseline')
        small=summ[(summ.family=='all')&(summ.stratum=='small_le5pct')].set_index('variant')
        large=summ[(summ.family=='all')&(summ.stratum=='gt50pct')].set_index('variant')
        ranking=small.loc[variant,'ap']>small.loc[control,'ap']+.005 or large.loc[variant,'clean_iou']>large.loc[control,'clean_iou']+.01
        if c['delta']>.005 and c['ci95'][0]>0 and cbase['delta']>0 and sum(d>0 for d in c['fold_deltas'].values())>=2 and ranking:promising.append(variant)
    save(out/'summary.json',dict(status='complete',n_unique=510,model='detectivesam',variants=list(variants),metrics=summ.to_dict('records'),comparisons=comparisons,
         e3_gate=bool(promising),e3_candidates=promising,
         e3_criterion='delta_vs_control>0.005, lower paired CI>0, delta_vs_original_baseline>0, >=2 positive folds, and small AP+0.005 or large clean IoU+0.01'))
    print(overall[['miou','fg_iou','clean_iou','auc','ap','boundary_f1']].to_string(),flush=True)
    print('E3_GATE',promising,flush=True)

if __name__=='__main__':main()
