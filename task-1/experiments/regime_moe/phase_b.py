"""Low-dimensional conditional thresholds; all fitting happens inside outer training."""
import argparse
import itertools
import time
from scipy.special import expit,logit
from common_moe import *
from metrics_moe import one,aggregate


def train_baseline_inner(data,train,valid,epoch,seed,device,path):
    if path.exists():
        ck=torch.load(path,weights_only=True);m=cv.model.Head(data['channels']).to(device);m.load_state_dict(ck['state_dict'])
    else:
        torch.manual_seed(seed);m=cv.model.Head(data['channels']).to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed)
        target=torch.from_numpy(data['gt']).to(device,dtype=torch.float32)
        for _ in range(epoch):cv.train_epoch(m,opt,rng,data['features'],target,train)
        torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},epoch=epoch,
                        train_ids=train.tolist(),validation_ids=valid.tolist()),path)
    return predict(m,data['features'],valid)


def global_threshold(p,g):
    values=[(t,float(miou(p,g,t).mean())) for t in cv.model.CONFIG['threshold_candidates']]
    return max(values,key=lambda a:a[1])[0],values


def conditional_threshold(p,g,gate,base_t):
    candidates=[(0.,0.,0.)]+[x for x in itertools.product([-.7,0.,.7],repeat=3) if x!=(0.,0.,0.)]
    rows=[]
    for offsets in candidates:
        t=expit(logit(base_t)+gate@np.array(offsets))
        rows.append(dict(offsets=offsets,miou=float(miou(p,g,t).mean())))
    best=max(rows,key=lambda x:x['miou'])
    return best['offsets'],rows


def record(method,fold,seed,variant,p,g,t,names,gate,extra=None,directory=None):
    directory=OUT if directory is None else directory
    ts=np.broadcast_to(t,(len(p),)).copy()
    rows=[dict(name=n,gate_small=float(gate[i,0]),gate_medium=float(gate[i,1]),gate_large=float(gate[i,2]),
               gate_uncertain=bool(gate[i].max()<.6),predicted_regime=int(gate[i].argmax()),
               **one(prob,gt,float(ts[i]))) for i,(n,prob,gt) in enumerate(zip(names,p,g))]
    stem=f'{method}_{variant}_fold{fold}_seed{seed}'
    np.savez_compressed(directory/f'{stem}.npz',names=names,probability=p,mask=p>=ts[:,None,None],threshold=ts,target=g,gate=gate)
    result=dict(method=method,fold=fold,seed=seed,variant=variant,metrics=aggregate(rows),per_image=rows,**(extra or {}))
    save(directory/f'{stem}.json',result)
    return result


def fit(method,fold,seed,device,data):
    stem=f'{method}_fold{fold}_seed{seed}'
    resultpath=OUT/f'{stem}_calibration.json'
    if resultpath.exists():return
    start=time.perf_counter();fit,val,trains,parts=split_ids(data,method,fold)
    model,ck=baseline(method,fold,seed,data,device)
    outer=predict(model,data['features'],val)
    # Verify actual immutable baseline artifacts rather than introducing a fresh control.
    oldpath=baseline_path(method,fold,seed)
    if method!='stegastamp' and seed==42:
        old=np.load(cv.OUT/f'{method}_oof.npz');expected=old['probability'][val]
    else:
        old=np.load(oldpath.with_suffix('.npz'));assert old['names'].tolist()==[data['names'][i] for i in val];expected=old['probability']
    error=float(np.max(abs(outer-expected)));assert error<2e-5,(method,error)
    inner=[]
    for j,(tr,va) in enumerate(zip(trains,parts)):
        inner.append(train_baseline_inner(data,tr,va,ck['epoch'],seed,device,OUT/f'{stem}_inner{j}_M.pt'))
    inner=np.concatenate(inner);ids=np.concatenate(parts)
    gate=np.load(OUT/f'{method}_fold{fold}_gate.npz')
    np.testing.assert_array_equal(gate['inner_ids'],ids);np.testing.assert_array_equal(gate['val_ids'],val)
    global_t,scores=global_threshold(inner,data['gt'][ids])
    offsets,conditional=conditional_threshold(inner,data['gt'][ids],gate['inner_gate'],global_t)
    ts=expit(logit(global_t)+gate['outer_gate']@np.array(offsets))
    names=[data['names'][i] for i in val]
    for variant,t in [('A0',ck['threshold']),('Bglobal',global_t),('A1',ts)]:
        record(method,fold,seed,variant,outer,data['gt'][val],t,names,gate['outer_gate'])
    # A gate-only empty/full baseline has no spatial information at all.
    constant=np.broadcast_to((gate['outer_gate'][:,2]>=.5)[:,None,None],outer.shape).astype(np.float32)
    record(method,fold,seed,'GateConstant',constant,data['gt'][val],.5,names,gate['outer_gate'])
    np.savez_compressed(OUT/f'{stem}_inner_baseline.npz',probability=inner,ids=ids,target=data['gt'][ids],gate=gate['inner_gate'])
    save(resultpath,dict(method=method,fold=fold,seed=seed,baseline_epoch=ck['epoch'],baseline_threshold=ck['threshold'],
        global_threshold=global_t,global_scores=scores,offsets=offsets,conditional_scores=conditional,
        reproduction_max_error=error,seconds=time.perf_counter()-start))
    print('PHASE B',stem,'global',global_t,'offsets',offsets,'seconds',time.perf_counter()-start,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',type=int,default=1)
    ap.add_argument('--methods',nargs='+',default=list(METHODS));args=ap.parse_args()
    setup_gpu(args.device);device=torch.device(f'cuda:{args.device}')
    assert (OUT/'phase_a.json').exists(),'Document Phase A first'
    for method in args.methods:
        data=load_data(method,device)
        for fold in (1,2,3):
            for seed in SEEDS:fit(method,fold,seed,device,data)
        del data;torch.cuda.empty_cache()
    if not all((OUT/f'{m}_fold{f}_seed{s}_calibration.json').exists() for m in METHODS for f in (1,2,3) for s in SEEDS):return
    summary=[]
    for method in METHODS:
        for variant in ('A0','Bglobal','A1','GateConstant'):
            runs=[json.loads(p.read_text()) for p in OUT.glob(f'{method}_{variant}_fold*_seed*.json')]
            summary.append(dict(method=method,variant=variant,**aggregate([x for r in runs for x in r['per_image']])))
    save(OUT/'phase_b.json',dict(status='complete',summary=summary))
    lines=['# Phase B: calibration only','', 'All thresholds fitted on inner held-out predictions and cross-fitted gates; outer predictions unchanged.','',
           '| Method | Variant | mIoU | Small AP | Small FG IoU | Large clean IoU |','|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['method']} | {r['variant']} | {r['miou']:.4f} | {r['small_ap']:.4f} | {r['small_fg_iou']:.4f} | {r['large_clean_iou']:.4f} |")
    (HERE/'PHASE_B.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
