"""Known-message gating with nested model selection and label-free feature extraction."""
import argparse
import joblib
import pandas as pd
from scipy.special import expit,logsumexp
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix,f1_score,balanced_accuracy_score,precision_recall_fscore_support,log_loss
from common_moe import *

KINDS=['majority','bit_lr','decoder_lr','decoder_hgb','pooled_lr']


def extract(method,device):
    path=OUT/f'{method}_gate_features.npz'
    if path.exists():return
    data=load_data(method)
    ds=cv.model.LocalisationDataset(ROOT/'data','train')
    lookup={r['image_name']:i for i,r in enumerate(ds.meta)}
    wm=cv.model.load_watermark(method,models_dir=ROOT/'checkpoints',device=f'cuda:{device}' if method!='stegastamp' else 'cpu')
    logits=[];messages=[]
    with torch.inference_mode():
        for j in range(0,len(data['names']),8):
            items=[ds[lookup[n]] for n in data['names'][j:j+8]]
            images=torch.from_numpy(np.stack([v['image'] for v in items])).permute(0,3,1,2).float()/255
            logits.append(wm.logits(images).cpu().numpy())
            messages.extend(v['message'] for v in items)
    z=np.concatenate(logits);msg=np.stack(messages);signed=z*(2*msg-1)
    # Messages are explicitly cast before subtraction to prevent uint8 underflow.
    signed=z*(2*msg.astype(float)-1)
    columns=['bit_accuracy','signed_mean','signed_std','signed_q10','signed_q50','signed_q90',
             'abs_mean','abs_std','abs_q10','abs_q50','abs_q90','known_message_loglik','expected_bit_probability','entropy']
    acc=((z>0)==msg).mean(1);conf=expit(z)
    entropy=-np.mean(conf*np.log(np.clip(conf,1e-12,1))+(1-conf)*np.log(np.clip(1-conf,1e-12,1)),axis=1)
    x=np.column_stack([acc,signed.mean(1),signed.std(1),*np.quantile(signed,[.1,.5,.9],axis=1),
                       abs(z).mean(1),abs(z).std(1),*np.quantile(abs(z),[.1,.5,.9],axis=1),
                       -np.logaddexp(0,-signed).mean(1),expit(signed).mean(1),entropy])
    pooled=torch.cat([f.mean((-2,-1)) for f in data['features']],1).numpy()
    prior=pd.read_csv(HERE.parent/'diagnostic_audit/results/decoder_metrics.csv').set_index('image_id')
    np.testing.assert_allclose(acc,prior.loc[data['names'],'bit_accuracy'],atol=1e-10)
    np.savez_compressed(path,names=data['names'],decoder=x,pooled=pooled,columns=columns,logits=z,messages=msg)
    print('FEATURES',method,x.shape,pooled.shape,flush=True)


def fit_gate(kind,x,pool,y,ids):
    if kind=='majority':model=DummyClassifier(strategy='prior');input=x[:,0:1]
    elif kind=='bit_lr':model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000));input=x[:,0:1]
    elif kind=='decoder_hgb':model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=4,max_depth=2,
        min_samples_leaf=10,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=42);input=x
    elif kind=='decoder_lr':model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000));input=x
    else:
        # PCA fits ONLY the current training rows; no global dimensionality reduction.
        pca=make_pipeline(StandardScaler(),PCA(n_components=8,svd_solver='full'))
        pca.fit(pool[ids]);input=np.column_stack([x,pca.transform(pool)])
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000))
    model.fit(input[ids],y[ids])
    probs=model.predict_proba(input)
    full=np.zeros((len(x),3));full[:,model.classes_]=probs
    return (dict(model=model,pca=pca if kind=='pooled_lr' else None,kind=kind),full)


def gate_metrics(y,p,area):
    pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=[0,1,2])
    prec,rec,_,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    confidence=p.max(1);correct=pred==y
    ece=0.
    for low in np.arange(0,1,.1):
        ids=(confidence>=low)&(confidence<(low+.1) if low<.9 else confidence<=1)
        if ids.any():ece+=ids.mean()*abs(correct[ids].mean()-confidence[ids].mean())
    near=(abs(area-.05)<=.02)|(abs(area-.5)<=.1)
    uncertain=confidence<.6
    return dict(n=len(y),confusion=cm,macro_f1=f1_score(y,pred,average='macro',labels=[0,1,2],zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y,pred),accuracy=correct.mean(),precision=prec,recall=rec,
        log_loss=log_loss(y,p,labels=[0,1,2]),ece10=ece,
        near_n=int(near.sum()),near_accuracy=correct[near].mean() if near.any() else None,
        near_confidence=confidence[near].mean() if near.any() else None,
        uncertain_n=int(uncertain.sum()),uncertain_accuracy=correct[uncertain].mean() if uncertain.any() else None)


def run(method):
    data=load_data(method);features=np.load(OUT/f'{method}_gate_features.npz')
    assert features['names'].tolist()==data['names']
    x,pool=features['decoder'],features['pooled'];y=regime(data['gt']);area=data['gt'].mean((1,2))
    oof={k:np.zeros((170,3)) for k in KINDS+['selected']};foldrows=[];reg_oof=np.zeros(170)
    for fold in (1,2,3):
        fit,val,trains,parts=split_ids(data,method,fold);inner_ids=np.concatenate(parts)
        inner={};scores={}
        for kind in KINDS:
            pp=[]
            for tr,va in zip(trains,parts):
                _,p=fit_gate(kind,x,pool,y,tr);pp.append(p[va])
            inner[kind]=np.concatenate(pp)
            scores[kind]=log_loss(y[inner_ids],inner[kind],labels=[0,1,2])
            model,p=fit_gate(kind,x,pool,y,fit);oof[kind][val]=p[val]
            joblib.dump(model,OUT/f'{method}_fold{fold}_{kind}.joblib')
            foldrows.append(dict(method=method,fold=fold,variant=kind,**gate_metrics(y[val],p[val],area[val])))
        chosen=min(KINDS[1:],key=lambda k:scores[k])
        oof['selected'][val]=oof[chosen][val]
        np.savez_compressed(OUT/f'{method}_fold{fold}_gate.npz',train_ids=fit,val_ids=val,inner_ids=inner_ids,
                            inner_gate=inner[chosen],outer_gate=oof[chosen][val],chosen=chosen)
        save(OUT/f'{method}_fold{fold}_gate.json',dict(chosen=chosen,inner_logloss=scores,
             train_names=[data['names'][i] for i in fit],validation_names=[data['names'][i] for i in val],
             crossfit=[dict(train=[data['names'][i] for i in tr],validation=[data['names'][i] for i in va]) for tr,va in zip(trains,parts)]))
        reg=make_pipeline(StandardScaler(),Ridge(alpha=10));reg.fit(x[fit],area[fit]);reg_oof[val]=reg.predict(x[val]).clip(0,1)
    summary={k:gate_metrics(y,p,area) for k,p in oof.items()}
    s=summary['selected'];cm=s['confusion'];go=bool(s['macro_f1']>summary['majority']['macro_f1']+.15 and
        s['recall'][0]>=.60 and s['recall'][2]>=.85 and cm[0,2]/cm[0].sum()<=.10 and cm[2,0]/cm[2].sum()<=.10)
    rows=[]
    for i,n in enumerate(data['names']):
        for k,p in oof.items():rows.append(dict(method=method,name=n,variant=k,gt_regime=int(y[i]),area=float(area[i]),
            p_small=p[i,0],p_medium=p[i,1],p_large=p[i,2],area_pred=reg_oof[i]))
    pd.DataFrame(rows).to_csv(OUT/f'{method}_gate_per_image.csv',index=False)
    bins=[]
    for a,b in [(0,.05),(.05,.2),(.2,.5),(.5,.8),(.8,1.01)]:
        ix=(area>=a)&(area<b);bins.append(dict(low=a,high=b,n=int(ix.sum()),actual=area[ix].mean(),predicted=reg_oof[ix].mean()))
    result=dict(method=method,summary=summary,folds=foldrows,go=go,area_regression=dict(mae=abs(reg_oof-area).mean(),spearman=spearmanr(reg_oof,area).statistic,bins=bins))
    save(OUT/f'{method}_phase_a.json',result)
    print('PHASE A',method,'GO',go,s,flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',type=int,default=1);args=ap.parse_args()
    setup_gpu(args.device);provenance()
    results=[]
    for method in METHODS:extract(method,args.device);results.append(run(method))
    save(OUT/'phase_a.json',dict(status='complete',methods=results))
    lines=['# Phase A: regime estimator','', 'Outer 3-fold; models selected only by inner log loss. No segmentation-map gate features.','',
           '| Method | Selected macro F1 | Balanced accuracy | Small recall | Large recall | Log loss | ECE | Go |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        s=r['summary']['selected'];lines.append(f"| {r['method']} | {s['macro_f1']:.4f} | {s['balanced_accuracy']:.4f} | {s['recall'][0]:.4f} | {s['recall'][2]:.4f} | {s['log_loss']:.4f} | {s['ece10']:.4f} | {r['go']} |")
    (HERE/'PHASE_A.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
