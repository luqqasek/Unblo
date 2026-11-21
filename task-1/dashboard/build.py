"""Build a curated static experiment dashboard from measured artifacts (stdlib only)."""
import json
import shutil
from polish import translate, localize
from datetime import datetime, timezone
from pathlib import Path
from report_view import render_report
from leader import leader_evidence

HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE.parent/'experiments'
DIST=HERE/'dist'


def build():
    DIST.mkdir(parents=True,exist_ok=True)
    rows=[]; submissions=[]; experiments=[]; gallery=[]; missing=[]
    def read(path):
        p=EXPERIMENTS/path
        if not p.exists(): missing.append(path); return None
        return json.loads(p.read_text(encoding='utf-8'))
    def row(exp,method,variant,protocol,value,detail='',folds=None):
        rows.append(dict(experiment=exp,method=method,variant=variant,protocol=protocol,miou=value,detail=detail,folds=folds))
    b=read('baseline/results/metrics.json')
    if b:
        for variant,scores in b['scores'].items():
            for m in ['pixelseal','trustmark','stegastamp']:
                row('baseline',m,variant,'Holdout · 34 / method',scores[m],'Spatial prior fitted on 136 images per method; seed 42.')
    for m in ['trustmark','stegastamp']:
        d=read(f'official_localization/results/{m}_metrics.json')
        if d:
            for variant,value in d['scores'].items():
                if variant.startswith('official_'):
                    row('official_localization',m,variant,'Holdout · 34 / method',value,'StegaStamp weights from community mirror; original identity unverified.' if m=='stegastamp' else 'Official Q detector. Default fallback may fill the image.')
    d=read('official_localization_cv/results/cv_metrics.json')
    if d:
        for m,variants in d['methods'].items():
            for variant,v in variants.items():
                row('official_localization_cv',m,variant,'5-fold CV · 170 / method',v['detector'],f"Prior {v['baseline']:.4f}; delta {v['delta']:+.4f}; improved folds {v['positive_folds']}/5.",v['folds'])
    d=read('trustmark_patches/results/cv_metrics.json')
    if d:
        for variant,v in d['variants'].items():
            value=v['vs_prior']
            row('trustmark_patches','trustmark',variant,'5-fold CV · 170 / method',value['detector'],f"Delta vs full-image top1 {v['vs_full_top1']['delta']:+.4f}. Multiple variants evaluated on the same folds.",value['folds'])
    for m in ['pixelseal','trustmark','stegastamp']:
        d=read(f'frozen_extractor/results/{m}_metrics.json')
        if d:
            half=next(s for s in d['threshold_scores'] if s['threshold']==.5)
            row('frozen_extractor',m,'Frozen head · threshold 0.5','Holdout · 34 / method',half['miou'],'Checkpoint selected on this validation split.')
            row('frozen_extractor',m,'Frozen head · calibrated','Holdout · 34 / method',d['selected']['miou'],f"Threshold {d['selected']['threshold']}; epoch {d['epoch']}; bit accuracy {d['mean_bit_accuracy']:.3f}. Threshold selected on validation; not independent testing.")
    d=read('frozen_extractor/results/cv3/metrics.json')
    if d:
        for m,v in d['methods'].items():
            stability=v['stability']
            folds=[dict(fold=f['fold'],detector=f['selected']['miou'],epoch=f['epoch'],threshold=f['threshold']) for f in v['folds']]
            row('frozen_extractor',m,'Frozen head · nested selection','3-fold CV · 170 / method',v['oof_miou'],
                f"Fold mean {stability['mean']:.4f}; sample SD {stability['sd']:.4f}; range {stability['minimum']:.4f}–{stability['maximum']:.4f}. Epoch and threshold selected inside training folds; head refitted before outer evaluation.",folds)
        v=d['overall']; stability=v['stability']
        row('frozen_extractor','overall','Frozen heads · all 510 images','3-fold CV · 170 / method',v['oof_miou'],
            f"Fold mean {stability['mean']:.4f}; sample SD {stability['sd']:.4f}. Each image evaluated once out of fold. SD measures split variability, not seed stability.",
            [dict(fold=i+1,detector=value) for i,value in enumerate(v['fold_miou'])])
    d=read('head_ablation/results/summary.json')
    if d:
        baseline=read('frozen_extractor/results/cv3/metrics.json')
        for m,v in baseline['methods'].items():
            row('frozen_extractor',m,'Baseline · screening fold','Screening · 57 / method',v['folds'][0]['selected']['miou'])
        row('frozen_extractor','overall','Baseline · screening fold','Screening · 57 / method',baseline['overall']['fold_miou'][0])
        for variant,v in d['variants'].items():
            detail='CV gate passed.' if v['passed'] else 'CV skipped: screening gate failed.'
            for r in v['screen']:
                row('head_ablation',r['method'],variant,'Screening · 57 / method',r['selected']['miou'],detail)
            row('head_ablation','overall',variant,'Screening · 57 / method',sum(r['selected']['miou'] for r in v['screen'])/3,detail)
            if v['cv']:
                for m in ['pixelseal','trustmark','stegastamp','overall']:
                    selected=[r for r in v['cv'] if m=='overall' or r['method']==m]
                    scores=[x for r in selected for x in r['selected']['per_image']]
                    folds=[]
                    for f in (1,2,3):
                        values=[x for r in selected if r['fold']==f for x in r['selected']['per_image']]
                        folds.append(dict(fold=f,detector=sum(values)/len(values)))
                    row('head_ablation',m,variant,'3-fold CV · 170 / method',sum(scores)/len(scores),'Includes the screening fold; selection bias remains.',folds)
    losses=read('loss_screen/results/summary.json')
    if losses:
        for variant,v in losses['variants'].items():
            detail='CV gate passed.' if v['passed'] else 'CV skipped: screening gate failed.'
            for r in v['screen']:
                row('loss_screen',r['method'],variant,'Screening · 57 / method',r['selected']['miou'],detail)
            row('loss_screen','overall',variant,'Screening · 57 / method',sum(r['selected']['miou'] for r in v['screen'])/3,detail)
            if v['cv']:
                for m in ['pixelseal','trustmark','stegastamp','overall']:
                    selected=[r for r in v['cv'] if m=='overall' or r['method']==m]
                    values=[x for r in selected for x in r['selected']['per_image']]
                    folds=[]
                    for f in (1,2,3):
                        a=[x for r in selected if r['fold']==f for x in r['selected']['per_image']]
                        folds.append(dict(fold=f,detector=sum(a)/len(a)))
                    row('loss_screen',m,variant,'3-fold CV · 170 / method',sum(values)/len(values),'Includes the screening fold; selection bias remains.',folds)
    messages=read('local_message/results/summary.json')
    if messages:
        for variant,v in messages['variants'].items():
            detail='Control run.' if variant=='control' else ('CV gate passed.' if v['passed'] else 'CV skipped: screening gate failed.')
            for r in v['screen']:
                row('local_message',r['method'],variant,'Screening · 57 / method',r['selected']['miou'],detail)
            row('local_message','overall',variant,'Screening · 57 / method',sum(r['selected']['miou'] for r in v['screen'])/3,detail)
            if v['cv']:
                for m in ['pixelseal','trustmark','stegastamp','overall']:
                    selected=[r for r in v['cv'] if m=='overall' or r['method']==m]
                    values=[x for r in selected for x in r['selected']['per_image']]
                    folds=[]
                    for f in (1,2,3):
                        a=[x for r in selected if r['fold']==f for x in r['selected']['per_image']]
                        folds.append(dict(fold=f,detector=sum(a)/len(a)))
                    row('local_message',m,variant,'3-fold CV · 170 / method',sum(values)/len(values),'Includes the screening fold; selection bias remains.',folds)
    decoder=read('decoder_evidence/results/summary.json')
    if decoder:
        for method,data in decoder['methods'].items():
            for r in data['runs']:
                row('decoder_evidence',method,r['variant'],'Screening · 57 / method',r['selected']['miou'],
                    f"Counterfactual message mIoU {r['counterfactual']['miou']:.4f}. {data['kind']}.")
            if len(data['cv'])==3:
                values=[x for r in data['cv'] for x in r['selected']['per_image']]
                row('decoder_evidence',method,data['winner'],'3-fold CV · 170 / method',sum(values)/len(values),
                    'Includes the screening fold; selection bias remains.',
                    [dict(fold=r['fold'],detector=r['selected']['miou']) for r in data['cv']])
    decoder_stable=read('decoder_evidence/results/stable_summary.json')
    if decoder_stable:
        for r in decoder_stable['runs']:
            row('decoder_evidence','stegastamp',f"Message + stable selection / fold {r['fold']}",
                'Diagnostic repeats · same CV folds',r['selected']['miou'],
                f"Previous {r['baseline']:.4f}; delta {r['delta']:+.4f}; wrong message {r['counterfactual']['miou']:.4f}.")
        if len(decoder_stable['runs'])==3:
            values=[x for r in decoder_stable['runs'] for x in r['selected']['per_image']]
            row('decoder_evidence','stegastamp','Message + stable selection','3-fold CV · 170 / method',sum(values)/len(values),
                'Includes the screening fold; selection bias remains.',
                [dict(fold=r['fold'],detector=r['selected']['miou'],delta=r['delta']) for r in decoder_stable['runs']])
    stable=read('stegastamp_stable_selection/results/metrics.json')
    if stable:
        for seed in (42,7,123):
            runs=[r for r in stable['runs'] if r['seed']==seed]
            scores=[v for r in runs for v in r['per_image']]
            row('stegastamp_stable_selection','stegastamp',f'Stratified inner CV / seed {seed}',
                '3-fold CV · 170 / method',sum(scores)/len(scores),
                'Development comparison on existing outer folds; epoch and threshold selected only on inner folds.',
                [dict(fold=r['fold'],detector=r['miou'],delta=r['delta']) for r in runs])
        for run in stable['runs']:
            row('stegastamp_stable_selection','stegastamp',f"Fold {run['fold']} / seed {run['seed']}",
                'Diagnostic repeats · same CV folds',run['miou'],
                f"Previous {run['baseline']:.4f}; delta {run['delta']:+.4f}; epoch {run['epoch']}; threshold {run['threshold']}.")
    stn_alignment=read('stegastamp_stn_alignment/results/summary.json')
    if stn_alignment and stn_alignment['status']=='complete':
        for seed in stn_alignment['seeds']:
            for variant,label in [('raw','Raw features'),('aligned','Inverse-STN features')]:
                folds=[dict(fold=r['fold'],detector=r['metrics']['miou'][variant],delta=r['metrics']['miou']['delta'])
                       for r in stn_alignment['folds'] if r['seed']==seed['seed']]
                row('stegastamp_stn_alignment','stegastamp',f"{label} / seed {seed['seed']}",
                    '3-fold CV · 170 / method',seed['metrics']['miou'][variant],
                    'Full Head trained from scratch; matched stable inner selection. Raw baseline checkpoints reused and verified. Development comparison.',folds)
        for variant,label in [('raw','Raw features'),('aligned','Inverse-STN features')]:
            row('stegastamp_stn_alignment','stegastamp',f'{label} / seed mean',
                '3-fold CV × 3 seeds · 170 unique images',stn_alignment['metrics']['miou'][variant],
                'Same 170 images across seeds; not 510 independent observations. No public leaderboard evaluation.')
    synthetic=read('synthetic_pretrain_2k/results/summary.json')
    if synthetic and synthetic['status']=='complete':
        for variant,label in [('control','Real-only matched control'),('synthetic','2000 synthetic pretrain + real finetune')]:
            for seed in synthetic['seeds']:
                folds=[dict(fold=r['fold'],detector=r['miou'],epoch=r['epoch'],threshold=r['threshold'])
                       for r in synthetic['folds'] if r['variant']==variant and r['seed']==seed['seed']]
                row('synthetic_pretrain_2k','stegastamp',f"{label} / seed {seed['seed']}",
                    '3-fold CV · 170 / method',seed[variant]['miou'],
                    'Same inverse-STN Head and GPU; nested epoch/threshold selection. External clean COCO covers; no test predictions.',folds)
            row('synthetic_pretrain_2k','stegastamp',f'{label} / seed mean',
                '3-fold CV × 3 seeds · 170 unique images',synthetic['metrics']['miou'][variant],
                f"Paired synthetic delta {synthetic['metrics']['miou']['delta']:+.5f}; CI95 {synthetic['metrics']['miou']['delta_ci95']}. 2000 synthetic training images, 200 separate synthetic validation images.")
        for seed in synthetic['synthetic_only']:
            row('synthetic_pretrain_2k','stegastamp',f"Synthetic only / seed {seed['seed']}",
                'External synthetic training · 170 real images',seed['miou'],
                'Fixed threshold 0.5; no real-image training or threshold selection. Same development evaluation images.')
        row('synthetic_pretrain_2k','stegastamp','Synthetic only / seed mean',
            'External synthetic training · 170 real images',synthetic['finetuning_vs_pretrain']['miou']['synthetic_only'],
            'Mean of three seeds on the same 170 real images; not an ensemble and not 510 independent observations.')
    rgb_srm=read('rgb_srm/results/summary.json')
    if rgb_srm:
        for r in rgb_srm['comparisons']:
            folds=[dict(fold=f['fold'],seed=f['seed'],detector=f['miou'],epoch=f['epoch'],threshold=f['threshold'])
                   for f in rgb_srm['folds'] if f['method']==r['method'] and f['variant']==r['variant']]
            row('rgb_srm',r['method'],r['variant'],r['protocol'],r['miou'],
                f"AUC {r['auc']:.4f}; AP {r['ap']:.4f}; small AP {r['small_ap']:.4f}; small Top-K {r['small_topk']:.4f}; small FG IoU {r['small_fg_iou']:.4f}; large clean IoU {r['large_clean_iou']:.4f}; parameters {r['parameters']}. GPU; no submission.",folds)
    scaling=read('synthetic_scaling/results/summary.json')
    if scaling and scaling['status']=='complete':
        for r in scaling['comparisons']:
            folds=[dict(fold=f['fold'],seed=f['seed'],detector=f['miou'],epoch=f['epoch'],threshold=f['threshold'])
                   for f in scaling['folds'] if f['method']==r['method'] and f['variant']==r['variant']]
            row('synthetic_scaling',r['method'],r['variant']+' / seed mean',
                '3-fold CV × 3 seeds · 170 unique images',r['miou'],
                f"After real finetuning. FG IoU {r['fg_iou']:.4f}; clean IoU {r['clean_iou']:.4f}; macro AP {r['ap']:.4f}; small FG IoU {r['small_fg_iou']:.4f}; seed SD {r['seed_sd']:.4f}. Nested 2k/4k pool; matched 2500/5000 updates. No submission.",folds)
    regime_moe=read('regime_moe/results/summary.json')
    if regime_moe:
        for r in regime_moe['comparisons']:
            folds=[dict(fold=f['fold'],seed=f['seed'],detector=f['miou']) for f in regime_moe['folds']
                   if f['method']==r['method'] and f['variant']==r['variant']]
            label=r['variant']+(' / oracle diagnostic' if r['variant']=='A6' else '')
            row('regime_moe',r['method'],label,r['protocol'],r['miou'],
                f"Macro AP {r['ap']:.4f}; small AP {r['small_ap']:.4f}; small Top-K {r['small_topk']:.4f}; small FG IoU {r['small_fg_iou']:.4f}; large clean IoU {r['large_clean_iou']:.4f}; large boundary F1 {r['large_boundary_f1']:.4f}. {r['role']}. No submission.",folds)
    diagnosis=read('stegastamp_diagnosis/results/metrics.json')
    if diagnosis:
        for run in diagnosis['runs']:
            row('stegastamp_diagnosis','stegastamp',
                f"Fold {run['fold']} / seed {run['seed']} / {run['mode']}",
                'Diagnostic repeats · same CV folds',run['miou'],
                f"Epoch {run['epoch']}; threshold {run['threshold']}. Same saved inner/outer splits; diagnostic result, not an independent test.")
        probe=read('stegastamp_diagnosis/results/epoch_probe.json')
        for run in probe or []:
            row('stegastamp_diagnosis','stegastamp',f"Fold 2 / seed 42 / {run['epoch']} epochs",
                'Diagnostic repeats · same CV folds',run['miou'],
                'Controlled training length; same seed and threshold. Post-hoc diagnosis, not independent evaluation.')
    remark=read('remark_localization/results/summary.json')
    if remark and remark['status']=='local_complete':
        remark_labels={'baseline':'Existing TrustMark head','B_full_selected':'ReMark full-image residual',
                       'C_local_selected':'ReMark local windows','R5_selected':'Frozen ReMark features',
                       'generic_selected':'Generic blur / resize / edges'}
        for r in remark['families']:
            row('remark_localization','trustmark',remark_labels[r['method']],
                '3-fold CV · 170 / method',r['miou'],
                f"Macro AUC {r['auc']:.4f}; AP {r['ap']:.4f}; oracle Top-K {r['topk_precision']:.4f}. Map and threshold selected only on inner calibration; frozen remover on GPU.")
    forensics=read('pretrained_forensics/results/summary.json')
    if forensics:
        for r in forensics['metrics']:
            if r['stratum']!='all' or r['family']=='all': continue
            if r['variant'] not in ('baseline','selected','direct_fixed','inverse_fixed','oracle_orientation_fixed','oracle_orientation_threshold'): continue
            oracle=r['variant'].startswith('oracle_')
            row('pretrained_forensics',r['family'],f"{r['model']} / {r['variant']}",
                'Oracle diagnostic · GT selection' if oracle else '3-fold CV · 170 / method',r['miou'],
                f"Macro AUC {r['auc']:.4f}; AP {r['ap']:.4f}; BF1 {r['boundary_f1']:.4f}. " +
                ('Diagnostic only; GT selects orientation/threshold.' if oracle else 'Frozen forensic model; calibration uses inner validation only. No submission.'))
        fusion_path=EXPERIMENTS/'pretrained_forensics/results/fusion/summary.json'
        if fusion_path.exists():
            fusion=json.loads(fusion_path.read_text(encoding='utf-8'))
            for r in fusion['metrics']:
                if r['stratum']=='all' and r['family']!='all':
                    row('pretrained_forensics',r['family'],f"DetectiveSAM fusion / {r['variant']}",
                        '3-fold CV · 170 / method',r['miou'],
                        f"Macro AUC {r['auc']:.4f}; AP {r['ap']:.4f}; BF1 {r['boundary_f1']:.4f}. Strict inner cross-fitting; matched watermark-only controls.")
    audit=read('diagnostic_audit/results/diagnostic_results.json')
    if audit:
        for r in audit['summary']:
            row('diagnostic_audit',r['watermark_method'],f"OOF audit / {r['variant']} / seed {r['seed']}",
                '3-fold CV · 170 / method',r['official_miou_mean'],
                f"Macro AUC {r['roc_auc_mean']:.4f}; AP {r['ap_mean']:.4f}; FG IoU {r['iou_watermark_mean']:.4f}; empty baseline {r['empty_baseline_miou_mean']:.4f}. Reused OOF predictions, not new independent validation.")
        alignment=read('diagnostic_audit/results/alignment_probe_confirmation.json')
        if alignment:
            for variant,key in [('Linear probe / raw','raw'),('Linear probe / inverse STN','aligned')]:
                row('diagnostic_audit','stegastamp',variant,'3-fold CV · 170 / method',alignment['summary']['miou'][key],
                    'Matched linear probes; development confirmation includes the screening fold. Not a production-head improvement.')
    titles={'synthetic_pretrain_2k/results/full_train/upload_receipt.json':'StegaStamp synthetic pretrain + real finetune; previous PixelSeal / TrustMark',
            'decoder_evidence/results/full_train/upload_receipt.json':'StegaStamp message correction + previous PixelSeal / TrustMark',
            'stegastamp_stable_selection/results/full_train/upload_receipt.json':'StegaStamp stable selection + previous PixelSeal / TrustMark',
            'baseline/results/upload_receipt.json':'Spatial prior for all methods',
            'trustmark_patches/results/test_top1_empty/upload_receipt.json':'TrustMark top1 / empty fallback + priors',
            'frozen_extractor/results/full_train/upload_receipt.json':'Frozen heads · full train (510 images)'}
    for receipt in sorted(EXPERIMENTS.glob('*/results/**/upload_receipt.json')):
        path=receipt.relative_to(EXPERIMENTS).as_posix()
        title=titles.get(path,receipt.relative_to(EXPERIMENTS).parts[0].replace('_',' '))
        d=read(path)
        if d:
            response=json.loads(d['response'])
            if response.get('status')=='success':
                experiment=receipt.relative_to(EXPERIMENTS).parts[0]
                receipt_target=DIST/'reports'/experiment/receipt.relative_to(EXPERIMENTS/experiment)
                receipt_target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(receipt,receipt_target)
                submissions.append(dict(title=title,id=response['submission_id'],score=response['score'],previous_best=response.get('previous_best'),delta=response.get('delta'),experiment=experiment,receipt=receipt_target.relative_to(DIST).as_posix()))
    submissions.sort(key=lambda s:s['id'],reverse=True)
    notes={
        'pretrained_forensics':('Pretrained manipulation localization','TruFor, Mesorch and DetectiveSAM on 510 real images: direct/inverse, geometry controls, complementarity and nested fusion.'),
        'synthetic_scaling':('Synthetic data: 2000 versus 4000 images','PixelSeal, TrustMark and StegaStamp: fixed data/update factorial, real-image finetuning, three folds and three seeds. Paired comparisons distinguish diversity from longer training.'),
        'research_handoff':('Research brief','Experiment history, unresolved questions, lessons and a ready-to-use research task. Evidence synthesis; no new model measurements.'),
        'regime_moe':('Regime-aware mixture of experts','Known-message regime estimation, cross-fitted calibration and shared-trunk specialists. Screening separates foreground localization from clean-region recovery; oracle routing is diagnostic only.'),
        'remark_localization':('ReMark localization signal','Frozen TrustMark Q remover: residuals, internal features, overlapping windows, clean and matched controls. Integration requires useful spatial evidence.'),
        'synthetic_pretrain_2k':('Synthetic pretraining: 2000 images','Real StegaStamp encoder on independent COCO photos; matched inverse-STN control versus synthetic pretrain and real finetuning, three folds and three seeds.'),
        'rgb_srm':('RGB / SRM features','Controlled GPU comparison: frozen extractor, RGB, RGB+SRM and feature fusion; small-watermark AP and Top-K determine screening promotion.'),
        'stegastamp_stn_alignment':('StegaStamp STN alignment — active optimization baseline','Selected by the user for further optimization because it fixes feature geometry, despite mixed CV results. Compare future variants on the same folds and seeds 42/7/123. See the leaderboard table for the latest accepted score.'),
        'diagnostic_audit':('Localization diagnostic audit','510 OOF images, spatial alignment, foreground ranking, area bias, feature probes and reproducible evidence. No test predictions changed.'),
        'stegastamp_stable_selection':('StegaStamp stable selection','Stratified three-fold inner selection compared with the previous procedure on identical outer folds and three seeds. Paired changes are measured without choosing the best seed.'),
        'stegastamp_diagnosis':('StegaStamp diagnosis','Mask geometry, errors, and three training seeds on identical CV splits. Fixed settings separate training randomness from inner selection.'),
        'baseline':('Spatial reference','Zero, one, and method-specific spatial priors. Establishes a cheap reference without reading RGB.'),
        'official_localization':('Native localization','TrustMark boxes and StegaStamp segmentation. Irregular masks and missing model hosting limit these approaches.'),
        'official_localization_cv':('Five-fold evaluation','All 170 images per method evaluated once. TrustMark gains are stable; StegaStamp gains are small and inconsistent.'),
        'trustmark_patches':('Boxes, patches & fallback','More boxes did not help. 2×2 patches lost detections. Removing the full-frame fallback gave the strongest TrustMark improvement.'),
        'frozen_extractor':('Frozen extractors + heads','Independent heads over four feature scales: holdout, 3-fold CV with inner selection, and full-train submission. Later RGB-only comparisons are documented in RGB / SRM features.')}
    for folder in sorted(EXPERIMENTS.iterdir()):
        if not folder.is_dir(): continue
        report=next((folder/name for name in ['REPORT.md','README.md'] if (folder/name).exists()),None)
        if report is None: continue
        title,note=notes.get(folder.name,(folder.name.replace('_',' ').title(),'New experiment — see its report for evidence.'))
        target=DIST/'reports'/folder.name/report.name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(report,target)
        report_sources = [report] + [p for p in sorted(set(folder.glob('*REPORT.md')) | set(folder.glob('RESEARCH.md'))) if p != report]
        if folder.name=='diagnostic_audit':
            report_sources=[report]+[p for p in sorted(folder.glob('*.md')) if p!=report]
            # Copy reviewable diagnostics only; never expose caches, models or prediction tensors.
            for source in (folder/'results').rglob('*'):
                if not source.is_file() or 'cache' in source.relative_to(folder/'results').parts: continue
                if source.suffix not in {'.csv','.json','.png','.jpg','.html','.txt'}: continue
                asset=target.parent/source.relative_to(folder)
                asset.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,asset)
        elif folder.name=='pretrained_forensics':
            report_sources=[report]+[p for p in sorted(folder.glob('*.md')) if p!=report]
            for source in (folder/'results').rglob('*'):
                if not source.is_file(): continue
                parts=source.relative_to(folder/'results').parts
                if any(p in {'vendor','weights','deps','inputs','crossfit','trufor','mesorch','detectivesam'} for p in parts): continue
                if source.suffix not in {'.csv','.json','.png','.jpg','.html','.txt'}: continue
                asset=target.parent/source.relative_to(folder)
                asset.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,asset)
        elif folder.name=='remark_localization':
            for source in (folder/'results').glob('*'):
                if source.is_file() and source.suffix in {'.csv','.json','.png'}:
                    asset=target.parent/'results'/source.name
                    asset.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,asset)
            for subdir,suffixes in [('controls',{'.csv','.json'}),('examples',{'.jpg','.html','.json'})]:
                for source in (folder/'results'/subdir).glob('*'):
                    if source.is_file() and source.suffix in suffixes:
                        asset=target.parent/'results'/subdir/source.name
                        asset.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,asset)
        elif folder.name in ('stegastamp_stn_alignment','rgb_srm','synthetic_pretrain_2k','regime_moe','synthetic_scaling'):
            if folder.name=='synthetic_scaling':
                report_sources=[report]+[p for p in sorted(folder.glob('*.md')) if p!=report]
            if folder.name=='regime_moe':
                report_sources=[report]+[p for p in sorted(folder.glob('*.md')) if p!=report]
                if (folder/'results/figures').exists():
                    shutil.copytree(folder/'results/figures',target.parent/'results/figures',dirs_exist_ok=True)
            if folder.name=='synthetic_pretrain_2k':
                report_sources=[report]+[p for p in sorted(folder.glob('*.md')) if p!=report]
                for name in ('training_summary.json','prediction_summary.json','upload_receipt.json'):
                    source=folder/'results/full_train'/name
                    if source.exists():
                        asset=target.parent/'results/full_train'/name
                        asset.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,asset)
            for source in (folder/'results').glob('*'):
                if source.is_file() and source.suffix in {'.csv','.json','.png'}:
                    asset=target.parent/'results'/source.name
                    asset.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,asset)
            if folder.name in ('rgb_srm','synthetic_pretrain_2k','regime_moe','synthetic_scaling') and (folder/'results/gallery').exists():
                shutil.copytree(folder/'results/gallery',target.parent/'results/gallery',dirs_exist_ok=True)
        for source in report_sources:
            if source!=report: shutil.copyfile(source,target.parent/source.name)
            (target.parent / source.with_suffix('.html').name).write_text(render_report(source, report_sources), encoding='utf-8')
        reports=[]
        for extra in sorted(set(folder.glob('*REPORT.md')) | set(folder.glob('RESEARCH.md'))):
            if extra==report: continue
            extra_target=target.parent/extra.name; shutil.copyfile(extra,extra_target)
            reports.append(dict(title=extra.stem.replace('_',' '),path=extra_target.with_suffix('.html').relative_to(DIST).as_posix()))
        experiments.append(dict(id=folder.name,title=title,note=note,report=target.with_suffix('.html').relative_to(DIST).as_posix(),reports=reports,measured=any(r['experiment']==folder.name for r in rows)))
        for source in sorted((folder/'results').glob('*example*.png')):
            target=DIST/'images'/folder.name/source.name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
            gallery.append(dict(experiment=folder.name,method=source.name.split('_')[0],src=target.relative_to(DIST).as_posix(),name=source.stem))
    data=dict(updated=datetime.now(timezone.utc).isoformat(),rows=rows,submissions=submissions,experiments=experiments,gallery=gallery,missing=missing)
    data['leader']=leader_evidence(submissions,experiments,DIST)
    if rgb_srm:
        data['rgb_srm_localization']=rgb_srm
    if regime_moe:
        data['regime_moe']=regime_moe
    frequency_path=EXPERIMENTS/'diagnostic_audit/results/frequency_audit.json'
    if frequency_path.exists():
        # Failure frequencies are counts, not another mIoU benchmark row.
        data['failure_frequency_audit']=json.loads(frequency_path.read_text(encoding='utf-8'))
    template=(HERE/'template.html').read_text(encoding='utf-8')
    template=template.replace('<nav>', '<nav><a href="index.html" lang="pl">Polski</a><a href="index-en.html" lang="en">English</a>')
    for filename, page, content in [
        ('index.html', translate(template).replace('<html lang="en">','<html lang="pl">'), localize(data)),
        ('index-en.html', template, data),
    ]:
        payload=json.dumps(content,ensure_ascii=False).replace('<','\\u003c')
        (DIST/filename).write_text(page.replace('__DATA__',payload),encoding='utf-8')
    (DIST/'data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(f'Built {DIST / "index.html"}: {len(rows)} measurements, {len(gallery)} images, {len(submissions)} submissions')


if __name__=='__main__': build()
