"""Confirm the promising raw-vs-STN-aligned linear-probe pair on all saved folds."""
from probes import *
from metrics_audit import boot

def main():
    method='stegastamp';c=load_cohort(method);names=c['names'];g=c['gt'];y=torch.from_numpy(g.astype(np.float32));splits=json.loads((cv.OUT/'folds.json').read_text())[method]
    raw=torch.load(cv.model.OUT/'stegastamp_features.pt',weights_only=True)['features'];aligned=torch.load(OUT/'cache/stegastamp_aligned_features.pt',weights_only=True)['features']
    rows=[];details=[]
    for fold,split in enumerate(splits,1):
        ids={k:np.array([names.index(n) for n in v]) for k,v in split.items()}
        for variant,ff in [('all',raw),('all_aligned',aligned)]:
            path=OUT/'cache'/f'confirm_alignment_{variant}_fold{fold}.npz'
            if fold==1:z=np.load(OUT/'cache'/f'probe_stegastamp_{variant}.npz');p=z['probability'];threshold=float(z['threshold'])
            elif path.exists():z=np.load(path);p=z['probability'];threshold=float(z['threshold'])
            else:
                print('ALIGNMENT CONFIRM',fold,variant,flush=True)
                head=train_probe(ff,y,ids['inner_train'],30,[0,1,2,3]);p=cv.predict(head,ff,ids['inner_validation'])
                threshold=float(max([.3,.4,.5,.6,.7],key=lambda t:ious(p>=t,g[ids['inner_validation']])[0].mean()))
                head=train_probe(ff,y,ids['train'],30,[0,1,2,3]);p=cv.predict(head,ff,ids['validation'])
                torch.save(dict(state_dict=head.state_dict(),threshold=threshold,fit_names=split['train']),path.with_suffix('.pt'))
                np.savez_compressed(path,probability=p,threshold=threshold,names=np.array(names)[ids['validation']])
            row=dict(fold=fold,variant=variant,threshold=threshold,**report_prob(p,g[ids['validation']],threshold));rows.append(row)
            from sklearn.metrics import roc_auc_score,average_precision_score
            for i,j in enumerate(ids['validation']):
                details.append(dict(image_id=names[j],fold=fold,variant=variant,miou=official.miou_image(p[i]>=threshold,g[j]),fg_iou=float(ious(p[i]>=threshold,g[j])[1]),auc=roc_auc_score(g[j].ravel(),p[i].ravel()),ap=average_precision_score(g[j].ravel(),p[i].ravel()),gt_fraction=float(g[j].mean())))
            print(row,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'alignment_probe_confirmation.csv',index=False);df=pd.DataFrame(details);df.to_csv(OUT/'alignment_probe_per_image.csv',index=False)
    a=df[df.variant=='all'].set_index('image_id');b=df[df.variant=='all_aligned'].set_index('image_id').loc[a.index]
    summary={col:dict(raw=float(a[col].mean()),aligned=float(b[col].mean()),delta=float((b[col]-a[col]).mean()),paired_ci=boot(b[col]-a[col])) for col in ['miou','fg_iou','auc','ap']}
    save_json(OUT/'alignment_probe_confirmation.json',dict(n=170,folds=3,seed=42,summary=summary,caveat='Development confirmation includes screening fold; not independent model selection test'))
    print(summary,flush=True)

if __name__=='__main__':main()
