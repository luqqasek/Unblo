"""Shared frozen feature trunk, frozen M expert, and tiny weighted S/L decoder copies."""
import argparse
import copy
import time
import torch.nn.functional as F
from common_moe import *
from phase_b import record,global_threshold

KINDS=('S','L','U')


@torch.inference_mode()
def trunk(base,features):
    base.eval();pieces=[]
    for j in range(0,len(features[0]),8):
        pieces.append(torch.cat([F.interpolate(p(f[j:j+8]),(32,32),mode='bilinear',align_corners=False)
                                for p,f in zip(base.projections,features)],1))
    return torch.cat(pieces)


def initialize(base,kind,seed):
    head=copy.deepcopy(base.fuse).requires_grad_(True)
    if kind=='L':
        with torch.no_grad():head[-1].weight.neg_();head[-1].bias.neg_()
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    return head,opt,np.random.default_rng(seed)


def epoch_train(head,opt,rng,x,target,weights,ids):
    head.train()
    shuffled=rng.permutation(ids)
    for j in range(0,len(shuffled),8):
        ii=shuffled[j:j+8];opt.zero_grad(set_to_none=True)
        logits=F.interpolate(head(x[ii]),(128,128),mode='bilinear',align_corners=False)[:,0]
        g=target[ii];p=logits.sigmoid()
        bce=F.binary_cross_entropy_with_logits(logits,g,reduction='none').mean((1,2))
        dice=1-(2*(p*g).sum((1,2))+1)/(p.sum((1,2))+g.sum((1,2))+1)
        loss=((.5*bce+.5*dice)*weights[ii]).sum()/weights[ii].sum()
        loss.backward();opt.step()


@torch.inference_mode()
def infer(head,x,ids,kind):
    head.eval()
    p=torch.cat([F.interpolate(head(x[ids[j:j+8]]),(128,128),mode='bilinear',align_corners=False)[:,0].sigmoid()
                 for j in range(0,len(ids),8)]).cpu().numpy()
    return 1-p if kind=='L' else p


def mixtures(s,m,l,u,g):
    """Deployable outputs: this interface deliberately accepts no target or GT regime."""
    gs,gm,gl=[g[:,i,None,None] for i in range(3)]
    stack=np.stack([s,m,l],1);ii=np.arange(len(m))
    return dict(A2=gs*s+(1-gs)*m,A3=gl*l+(1-gl)*m,A4=gs*s+gm*m+gl*l,
                A5=(s+m+l)/3,Hard=stack[ii,g.argmax(1)],
                S_only=s,L_only=l,U0=u)


def oracle_diagnostic(s,m,l,gt):
    """Evaluation-only GT routing, never called by deployable mixture inference."""
    return np.stack([s,m,l],1)[np.arange(len(m)),regime(gt)]


def fit(method,fold,seed,device,data):
    stem=f'{method}_fold{fold}_seed{seed}';meta=OUT/f'{stem}_experts.json'
    if meta.exists():return
    start=time.perf_counter();fit,val,trains,parts=split_ids(data,method,fold)
    gt=torch.from_numpy(data['gt']).to(device,dtype=torch.float32);reg=regime(data['gt'])
    weights={k:torch.from_numpy(np.where(reg==({'S':0,'L':2}.get(k,-1)),4.,1.)).to(device,dtype=torch.float32) for k in KINDS}
    targets={k:1-gt if k=='L' else gt for k in KINDS}
    models=[];xs=[]
    for j in range(len(parts)):
        ck=torch.load(OUT/f'{stem}_inner{j}_M.pt',weights_only=True)
        base=cv.model.Head(data['channels']).to(device);base.load_state_dict(ck['state_dict']);base.requires_grad_(False)
        xs.append(trunk(base,data['features']))
        models.append({k:initialize(base,k,seed) for k in KINDS})
    ids=np.concatenate(parts);history={k:[] for k in KINDS};best={k:-1. for k in KINDS}
    stale={k:0 for k in KINDS};chosen={};best_prob={}
    for epoch in range(1,51):
        for kind in KINDS:
            if stale[kind]>=8:continue
            probs=[]
            for models_i,x,tr,va in zip(models,xs,trains,parts):
                head,opt,rng=models_i[kind]
                epoch_train(head,opt,rng,x,targets[kind],weights[kind],tr)
                probs.append(infer(head,x,va,kind))
            p=np.concatenate(probs);score=float(miou(p,data['gt'][ids],.5).mean());history[kind].append(score)
            if score>best[kind]:best[kind]=score;stale[kind]=0;chosen[kind]=epoch;best_prob[kind]=p.copy()
            else:stale[kind]+=1
        if epoch%10==0:print(stem,'expert epoch',epoch,'best',best,'selected',chosen,flush=True)
        if all(v>=8 for v in stale.values()):break
    del models,xs,head,opt,base
    base,ck=baseline(method,fold,seed,data,device);base.requires_grad_(False)
    x=trunk(base,data['features']);outer={};states={};reproduction={}
    for kind in KINDS:
        head,opt,rng=initialize(base,kind,seed)
        for _ in range(chosen[kind]):epoch_train(head,opt,rng,x,targets[kind],weights[kind],fit)
        outer[kind]=infer(head,x,val,kind)
        states[kind]={k:v.cpu() for k,v in head.state_dict().items()}
        restored=copy.deepcopy(base.fuse);restored.load_state_dict(states[kind])
        reproduction[kind]=float(np.max(abs(infer(restored,x,val,kind)-outer[kind])))
        assert reproduction[kind]<1e-7
    outer_base=np.load(OUT/f'{method}_A0_fold{fold}_seed{seed}.npz')
    inner_base=np.load(OUT/f'{stem}_inner_baseline.npz');np.testing.assert_array_equal(ids,inner_base['ids'])
    gate=np.load(OUT/f'{method}_fold{fold}_gate.npz')
    inner_mix=mixtures(best_prob['S'],inner_base['probability'],best_prob['L'],best_prob['U'],gate['inner_gate'])
    outer_mix=mixtures(outer['S'],outer_base['probability'],outer['L'],outer['U'],gate['outer_gate'])
    inner_mix['A6']=oracle_diagnostic(best_prob['S'],inner_base['probability'],best_prob['L'],data['gt'][ids])
    outer_mix['A6']=oracle_diagnostic(outer['S'],outer_base['probability'],outer['L'],data['gt'][val])
    rows={};calibrations={}
    for variant,p in outer_mix.items():
        t,scores=global_threshold(inner_mix[variant],data['gt'][ids]);calibrations[variant]=dict(threshold=t,scores=scores)
        r=record(method,fold,seed,variant,p,data['gt'][val],t,[data['names'][i] for i in val],gate['outer_gate'],
                 extra=dict(oracle=variant=='A6',expert_epochs=chosen))
        rows[variant]=r['metrics']
    torch.save(dict(states=states,baseline_source=str(baseline_path(method,fold,seed).relative_to(ROOT)),
        baseline_sha256=sha(baseline_path(method,fold,seed)),channels=data['channels'],epochs=chosen,
        train_names=[data['names'][i] for i in fit],validation_names=[data['names'][i] for i in val]),OUT/f'{stem}_experts.pt')
    np.savez_compressed(OUT/f'{stem}_expert_maps.npz',names=[data['names'][i] for i in val],small=outer['S'],
        baseline=outer_base['probability'],clean=1-outer['L'],unweighted=outer['U'],gate=gate['outer_gate'],
        inner_ids=ids,inner_small=best_prob['S'],inner_baseline=inner_base['probability'],inner_clean=1-best_prob['L'],
        inner_unweighted=best_prob['U'],inner_gate=gate['inner_gate'])
    save(meta,dict(method=method,fold=fold,seed=seed,epochs=chosen,history=history,
        calibrations=calibrations,metrics=rows,seconds=time.perf_counter()-start,reproduction=reproduction,
        expert_parameters=sum(p.numel() for p in base.fuse.parameters()),
        frozen_head_parameters=sum(p.numel() for p in base.parameters()),device=torch.cuda.get_device_name(device)))
    print('EXPERTS COMPLETE',stem,'seconds',time.perf_counter()-start,'A4',rows['A4'],flush=True)


def screening():
    gates=[]
    for method in METHODS:
        a=json.loads((OUT/f'{method}_A0_fold1_seed42.json').read_text())['metrics']
        if not (OUT/f'{method}_fold1_seed42_experts.json').exists():continue
        for v in ('A2','A3','A4'):
            m=json.loads((OUT/f'{method}_{v}_fold1_seed42.json').read_text())['metrics']
            small=m['small_ap']>a['small_ap'] and m['small_topk']>a['small_topk']
            large=m['large_clean_iou']>a['large_clean_iou'] and m['large_boundary_f1']>a['large_boundary_f1']
            passed=(small if v=='A2' else large if v=='A3' else small or large) and m['miou']>=a['miou']-.01
            gates.append(dict(method=method,variant=v,passed=bool(passed),small_pass=bool(small),large_pass=bool(large),baseline=a,candidate=m))
    save(OUT/'screening.json',gates)
    return gates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',type=int,default=1)
    ap.add_argument('--methods',nargs='+',default=list(METHODS));ap.add_argument('--stage',choices=['screen','confirm'],default='screen');args=ap.parse_args()
    assert (HERE/'PHASE_A.md').exists() and (HERE/'PHASE_B.md').exists(),'Document phases A and B before C'
    setup_gpu(args.device);device=torch.device(f'cuda:{args.device}')
    go={r['method']:r['go'] for r in json.loads((OUT/'phase_a.json').read_text())['methods']}
    if args.stage=='confirm':gates=screening()
    for method in args.methods:
        if not go[method]:continue
        if args.stage=='confirm' and not any(r['method']==method and r['passed'] for r in gates):continue
        data=load_data(method,device)
        jobs=[(1,42)] if args.stage=='screen' else [(f,s) for f in (1,2,3) for s in SEEDS]
        for f,s in jobs:fit(method,f,s,device,data)
        del data;torch.cuda.empty_cache()
    if all(not go[m] or (OUT/f'{m}_fold1_seed42_experts.json').exists() for m in METHODS):
        decisions=screening()
        if args.stage=='screen':
            lines=['# Screening ekspertow: fold1, seed42','','|Metoda|Wariant|Awans|Small AP/TopK|Large clean/boundary|','|---|---|---|---|---|']
            lines+=['|{method}|{variant}|{passed}|{small_pass}|{large_pass}|'.format(**r) for r in decisions]
            (HERE/'SCREENING.md').write_text('\n'.join(lines),encoding='utf-8')
        print('SCREENING',[(r['method'],r['variant'],r['passed']) for r in decisions],flush=True)


if __name__=='__main__':main()
