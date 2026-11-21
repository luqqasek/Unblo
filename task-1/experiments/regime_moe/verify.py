"""Leakage, inference isolation, official metrics and checkpoint/prediction checks."""
import importlib.util
import itertools
import joblib
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score
from common_moe import *
from metrics_moe import one,aggregate
from phase_a import fit_gate
from phase_c import trunk,infer,mixtures,initialize,oracle_diagnostic
report_spec=importlib.util.spec_from_file_location('regime_moe_report',HERE/'report.py')
report_module=importlib.util.module_from_spec(report_spec);report_spec.loader.exec_module(report_module)
load_runs=report_module.load_runs


def main():
    setup_gpu(1);device=torch.device('cuda:1')
    spec=importlib.util.spec_from_file_location('official_moe_metric',ROOT/'starter-kit/task-1/wmloc/utils.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    config=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in config['sources_sha256'].items())
    gates=json.loads((OUT/'screening.json').read_text())
    promoted=[m for m in METHODS if any(g['method']==m and g['passed'] for g in gates)]
    model_errors={};rank_error=0.;miou_error=0.;checked=0;sets=0
    runs=load_runs();lookup={(r['method'],r['variant'],r['fold'],r['seed']):r for r in runs}
    for method in METHODS:
        data=load_data(method,device);features=np.load(OUT/f'{method}_gate_features.npz');y=regime(data['gt'])
        assert features['names'].tolist()==data['names'] and np.isfinite(features['decoder']).all()
        for fold in (1,2,3):
            fit,val,trains,parts=split_ids(data,method,fold)
            manifest=json.loads((OUT/f'{method}_fold{fold}_gate.json').read_text())
            chosen=min([k for k in manifest['inner_logloss'] if k!='majority'],key=manifest['inner_logloss'].get)
            assert chosen==manifest['chosen']
            assert manifest['train_names']==[data['names'][i] for i in fit]
            gate=np.load(OUT/f'{method}_fold{fold}_gate.npz');np.testing.assert_array_equal(gate['inner_ids'],np.concatenate(parts))
            np.testing.assert_allclose(gate['outer_gate'].sum(1),1,atol=1e-12)
            np.testing.assert_allclose(gate['inner_gate'].sum(1),1,atol=1e-12)
            # Deliberately corrupt OUTER targets: trained gate and outer probabilities must not change.
            changed=y.copy();changed[val]=(changed[val]+1)%3
            _,pred=fit_gate(chosen,features['decoder'],features['pooled'],changed,fit)
            np.testing.assert_allclose(pred[val],gate['outer_gate'],atol=1e-12)
            for seed in SEEDS:
                stem=f'{method}_fold{fold}_seed{seed}'
                cal=json.loads((OUT/f'{stem}_calibration.json').read_text())
                assert cal['global_threshold']==max(cal['global_scores'],key=lambda v:v[1])[0]
                assert cal['offsets']==max(cal['conditional_scores'],key=lambda v:v['miou'])['offsets']
                a0=np.load(OUT/f'{method}_A0_fold{fold}_seed{seed}.npz')
                a1=np.load(OUT/f'{method}_A1_fold{fold}_seed{seed}.npz')
                np.testing.assert_array_equal(a0['probability'],a1['probability'])
                expected_t=expit(logit(cal['global_threshold'])+gate['outer_gate']@np.array(cal['offsets']))
                np.testing.assert_allclose(a1['threshold'],expected_t,atol=1e-12)
                for j,(tr,va) in enumerate(zip(trains,parts)):
                    inner=torch.load(OUT/f'{stem}_inner{j}_M.pt',weights_only=True)
                    assert inner['train_ids']==tr.tolist() and inner['validation_ids']==va.tolist()
                    assert inner['epoch']==cal['baseline_epoch']
                for variant in ('A0','A1','Bglobal'):
                    r=lookup[(method,variant,fold,seed)]['per_image']
                    b=lookup[(method,'A0',fold,seed)]['per_image']
                    for x,z in zip(r,b):
                        for k in ('ap','auc','topk'):rank_error=max(rank_error,abs(x[k]-z[k]))
                path=OUT/f'{stem}_experts.pt'
                if not path.exists():continue
                sets+=1;ck=torch.load(path,weights_only=True)
                assert ck['train_names']==[data['names'][i] for i in fit]
                assert ck['validation_names']==[data['names'][i] for i in val]
                assert sha(ROOT/ck['baseline_source'])==ck['baseline_sha256']
                meta=json.loads(path.with_suffix('.json').read_text())
                base,_=baseline(method,fold,seed,data,device);base.requires_grad_(False)
                x=trunk(base,data['features']);maps=np.load(OUT/f'{stem}_expert_maps.npz')
                for kind,key in [('S','small'),('L','clean'),('U','unweighted')]:
                    assert ck['epochs'][kind]==np.argmax(meta['history'][kind])+1
                    head,opt,rng=initialize(base,kind,seed)
                    assert all(p.requires_grad for p in head.parameters()) and not any(p.requires_grad for p in base.parameters())
                    if kind=='L':
                        initial=infer(head,x,val,'L');reference=predict(base,data['features'],val)
                        np.testing.assert_allclose(initial,reference,atol=2e-6)
                    head.load_state_dict(ck['states'][kind]);p=infer(head,x,val,kind)
                    expected=1-maps[key] if kind=='L' else maps[key]
                    error=float(np.max(abs(p-expected)));assert error<2e-5,(method,kind,error)
                    model_errors[stem+'_'+kind]=error
                generated=mixtures(maps['small'],maps['baseline'],1-maps['clean'],maps['unweighted'],maps['gate'])
                corrupted={k:v.copy() for k,v in generated.items()}
                generated['A6']=oracle_diagnostic(maps['small'],maps['baseline'],1-maps['clean'],data['gt'][val])
                corrupted['A6']=oracle_diagnostic(maps['small'],maps['baseline'],1-maps['clean'],~data['gt'][val])
                for variant,p in generated.items():
                    z=np.load(OUT/f'{method}_{variant}_fold{fold}_seed{seed}.npz')
                    np.testing.assert_allclose(p,z['probability'],atol=2e-7)
                    if variant!='A6':np.testing.assert_array_equal(p,corrupted[variant])
                    cc=meta['calibrations'][variant]
                    assert cc['threshold']==max(cc['scores'],key=lambda v:v[1])[0]
            print('VERIFY',method,fold,flush=True)
        del data;torch.cuda.empty_cache()
    assert rank_error==0
    for r in runs:
        stem=f'{r["method"]}_{r["variant"]}_fold{r["fold"]}_seed{r["seed"]}'
        z=np.load(OUT/f'{stem}.npz');assert np.isfinite(z['probability']).all()
        np.testing.assert_array_equal(z['mask'],z['probability']>=z['threshold'][:,None,None])
        assert len(r['per_image'])==len(z['names'])
        for p,g,rr in zip(z['mask'],z['target'],r['per_image']):
            error=abs(official.miou_image(p,g)-rr['miou']);miou_error=max(miou_error,error);checked+=1
        assert aggregate(r['per_image'])==r['metrics']
    assert miou_error<1e-12
    for method in METHODS:
        rr=[r for r in runs if r['method']==method and r['variant']=='A4']
        assert len(rr)==(9 if method in promoted else 1)
        for seed in {r['seed'] for r in rr}:
            names=[x['name'] for r in rr if r['seed']==seed for x in r['per_image']]
            assert len(set(names))==len(names)==(170 if method in promoted else 57)
    assert (HERE/'PHASE_B.md').stat().st_mtime<min(p.stat().st_mtime for p in OUT.glob('*_experts.pt'))
    assert (HERE/'SCREENING.md').stat().st_mtime<min(p.stat().st_mtime for p in OUT.glob('*_experts.pt') if 'fold1_seed42' not in p.name)
    save(OUT/'verification.json',dict(status='passed',segmentation_records=len(runs),per_image_records=checked,
        expert_checkpoint_sets=sets,experts_reproduced=len(model_errors),max_prediction_error=max(model_errors.values()),
        official_miou_max_error=miou_error,calibration_ranking_difference=rank_error,
        source_hashes_unchanged=True,outer_label_corruption_gate_test=True,outer_GT_independent_nonoracle_mixtures=True,
        matched_inner_partitions=True,deployable_mixture_interface_has_no_GT=True,no_test_or_synthetic_inputs=True,phase_order_verified=True,
        promoted_methods=promoted,code_sha256={p.name:sha(p) for p in HERE.glob('*.py')}))
    print('ALL CHECKS PASSED',checked,flush=True)


if __name__=='__main__':main()
