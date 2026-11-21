"""Small deterministic diagnostic probes and tiny-set sanity; no production writes."""
from common import *
import pandas as pd
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score

class Probe(nn.Module):
    def __init__(self,features,fit,levels):
        super().__init__();self.levels=levels
        self.maps=nn.ModuleList([nn.Conv2d(features[k].shape[1],1,1) for k in levels])
        for k in levels:
            f=features[k][fit];self.register_buffer(f'mean{k}',f.mean((0,2,3))[None,:,None,None]);self.register_buffer(f'std{k}',f.std((0,2,3))[None,:,None,None].clamp_min(1e-3))
    def forward(self,features):
        out=[]
        for layer,k in zip(self.maps,self.levels):
            x=(features[k]-getattr(self,f'mean{k}'))/getattr(self,f'std{k}')
            out.append(F.interpolate(layer(x),size=(128,128),mode='bilinear',align_corners=False))
        return sum(out)[:,0]/len(out)

def train_probe(features,y,fit,epochs,levels,seed=42):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);head=Probe(features,fit,levels)
    opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.0001)
    for epoch in range(epochs):
        ids=rng.permutation(fit)
        for start in range(0,len(ids),8):
            batch=ids[start:start+8];opt.zero_grad();logits=head([f[batch] for f in features]);loss=F.binary_cross_entropy_with_logits(logits,y[batch]);loss.backward();opt.step()
    return head

def report_prob(p,g,t):
    s,fg,bg,*_=ious(p>=t,g);mixed=[i for i in range(len(g)) if g[i].any() and not g[i].all()]
    return dict(miou=float(s.mean()),fg_iou=float(fg.mean()),roc_auc_macro=float(np.mean([roc_auc_score(g[i].ravel(),p[i].ravel()) for i in mixed])),
        ap_macro=float(np.mean([average_precision_score(g[i].ravel(),p[i].ravel()) for i in mixed])),
        pooled_roc_auc=roc_auc_score(g.ravel(),p.ravel()),pooled_ap=average_precision_score(g.ravel(),p.ravel()),n=len(g))

def run_probes():
    rows=[];splits=json.loads((cv.OUT/'folds.json').read_text())
    for method in METHODS:
        cache=torch.load(cv.model.OUT/f'{method}_features.pt',weights_only=True);features=cache['features'];names=cache['names']
        c=load_cohort(method);g=c['gt'];y=torch.from_numpy(g.astype(np.float32));split=splits[method][0]
        ids={k:np.array([names.index(n) for n in v]) for k,v in split.items()}
        variants=[(f'level{k+1}',[k],'cache32') for k in range(4)]+[('early',[0,1],'cache32'),('late',[2,3],'cache32'),('all',[0,1,2,3],'cache32'),('all_shuffled',[0,1,2,3],'shuffled32'),('early_high',[0,1],'cache128')]
        if method=='stegastamp':variants.append(('all_aligned',[0,1,2,3],'aligned32'))
        for label,levels,res in variants:
            path=OUT/f'probe_{method}_{label}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            print('PROBE',method,label,flush=True)
            ff=features
            if res=='aligned32':
                aligned=torch.load(OUT/'cache/stegastamp_aligned_features.pt',weights_only=True);assert aligned['names']==names;ff=aligned['features']
            elif res=='cache128':
                high=torch.load(HERE.parent/'head_ablation/results/features128'/f'{method}_features.pt',weights_only=True)
                assert high['names']==names;ff=high['features']
            elif res=='shuffled32':
                ff=[];rng=np.random.default_rng(20261121)
                # Independently shuffle each image at each scale; train and validation both shuffled.
                for f in features:
                    v=f.clone().flatten(2)
                    for i in range(len(v)):v[i]=v[i,:,rng.permutation(v.shape[-1])]
                    ff.append(v.reshape(f.shape))
            # Predefined 30 epochs; select threshold on existing inner holdout only.
            head=train_probe(ff,y,ids['inner_train'],30,levels)
            p=cv.predict(head,ff,ids['inner_validation']);innerg=g[ids['inner_validation']]
            candidates=np.array([.3,.4,.5,.6,.7]);threshold=float(max(candidates,key=lambda t:ious(p>=t,innerg)[0].mean()))
            head=train_probe(ff,y,ids['train'],30,levels)
            p=cv.predict(head,ff,ids['validation']);measured=report_prob(p,g[ids['validation']],threshold)
            row=dict(method=method,variant=label,feature_source=res,levels=levels,fold=1,seed=42,epochs=30,threshold=threshold,
                parameters=sum(v.numel() for v in head.parameters()),train_n=len(ids['train']),**measured)
            torch.save(dict(state_dict=head.state_dict(),levels=levels,threshold=threshold,fit_names=split['train']),OUT/'cache'/f'probe_{method}_{label}.pt')
            np.savez_compressed(OUT/'cache'/f'probe_{method}_{label}.npz',probability=p,names=np.array(names)[ids['validation']],threshold=threshold)
            save_json(path,row);rows.append(row);pd.DataFrame(rows).to_csv(OUT/'feature_probe_results.csv',index=False)
            print('PROBE DONE',row,flush=True)
            del head,ff
    pd.DataFrame(rows).to_csv(OUT/'feature_probe_results.csv',index=False)

def tiny():
    rows=[]
    for method in METHODS:
        path=OUT/f'tiny_{method}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        cache=torch.load(cv.model.OUT/f'{method}_features.pt',weights_only=True);features=cache['features'];c=load_cohort(method);g=c['gt'];y=torch.from_numpy(g.astype(np.float32))
        # Five prespecified area quantiles: includes small foreground and large masks.
        order=np.argsort(g.mean((1,2)),kind='stable');ids=order[[8,42,84,127,161]]
        head,opt,rng=cv.initialize(features);history=[]
        for epoch in range(1,301):
            cv.train_epoch(head,opt,rng,features,y,ids)
            if epoch in (1,25,50,100,200,300):
                p=cv.predict(head,features,ids);v=report_prob(p,g[ids],.5);history.append(dict(epoch=epoch,**v));print('TINY',method,epoch,v['miou'],flush=True)
        row=dict(method=method,protocol='IN-SAMPLE engineering sanity, not OOF',seed=42,names=np.array(c['names'])[ids].tolist(),history=history,
            **report_prob(p,g[ids],.5))
        save_json(path,row);rows.append(row);torch.save(head.state_dict(),OUT/'cache'/f'tiny_{method}.pt')
    save_json(OUT/'tiny_overfit.json',rows)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['probes','tiny'],default='probes');args=ap.parse_args()
    run_probes() if args.mode=='probes' else tiny()
