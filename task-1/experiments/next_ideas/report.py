"""Source-backed report; distinguish old OOF, new CV and actual submissions."""
import json
from pathlib import Path
import numpy as np
from experiments.native_localizer.common import atomic_json
from experiments.triforensic_wm.evaluate import summarize


def main():
    root=Path(__file__).parent;r=root/'results';a=json.loads((r/'error_budget.json').read_text());f=json.loads((r/'fallback.json').read_text());measures=[]
    lines=['# Next ideas: opportunity audit and StegaStamp adaptation','',
        'Accepted public best remains #2675, mIoU0.6440 until a new server receipt is recorded. This work was explicitly authorized after the TriForensic cancellation. [Frozen screening protocol](PROTOCOL.md).','',
        '## Findings before new training','',
        'All510 training masks contain both classes. Exact train metadata identifies85 full-canvas/restoration images per family; the remaining43 rectangular and42 fragment supports are geometry proxies, not definitive edit-type labels.','',
        '| Family | Protocol | mIoU | Empty masks | Full masks | Clean IoU on85 restored canvases |','|---|---|---:|---:|---:|---:|']
    restoration_error=0
    for family,d in a['families'].items():
        restoration=d['error_budget'][0];restoration_error+=restoration['contribution_to_510_error_clean'];m=d['metrics']['all']['miou']
        lines.append(f"| {family} | {'Historical #2409 OOF' if family=='pixelseal' else '#2675 OOF + fixed crops' if family=='trustmark' else 'Synthetic-only, all170 real'} | {m:.6f} | {int(d['empty_predictions'])} | {int(d['full_predictions'])} | {1-2*restoration['mean_clean_loss']:.6f} |")
        measures.append({'family':family,'name':'A1 baseline','protocol':d['protocol'],'miou':m,'note':'Saved label-disjoint maps; PS is historical #2409, not current full OOF.'})
    lines+=['',f"Clean-class errors on those255 restoration images contribute{restoration_error:.6f} score units, or{100*restoration_error/a['sum_weighted_error']:.1f}% of the measured total error. This is an error budget, not an achievable promised gain. The weighted error cells sum exactly to1 minus the510-image proxy mIoU{a['combined_proxy_miou']:.6f}.",'',
        'The510-image number mixes the historical PS OOF with current TM/SS recipes. The newer PS #2675 continuation has only a57-image screen; do not call this current full OOF or infer the hidden score from it.','',
        '## A5: forced small regions are not a universal fix','',
        '| Family | Fixed fallback on empty/full masks | mIoU | Delta | Bootstrap95% delta | Improved / worse |','|---|---|---:|---:|---|---|']
    for family,dd in f['families'].items():
        for mode,d in dd.items():
            lines.append(f"| {family} | {mode} | {d['miou']:.6f} | {d['delta']:+.6f} | [{d['ci95'][0]:+.6f},{d['ci95'][1]:+.6f}] | {d['improved']} / {d['worse']} |")
            measures.append({'family':family,'name':'A5 '+mode,'protocol':'Exploratory saved-map fallback; same images as A1 baseline','miou':d['miou'],'note':f"Delta{d['delta']:+.6f}; changed{d['changed_masks']} masks; fixed1% area, no GT for placement."})
    lines+=['','These two rules were specified before measurement. Neither establishes that hidden test masks must contain both classes. PS/TM intervals include zero. StegaStamp top1pct is a small promising signal, not a large capability jump. All intervals are exploratory after extensive reuse of these development images.','',
        '## A4: native StegaStamp real-label adaptation','',
        'Three original folds,150 fixed updates,75% exact synthetic replay /25% real training examples. Raw weights,FP32,threshold0.5; no augmentation or architecture change. Initialization must match archived probabilities within1e-4. Update50 is progress only;150 is the predeclared endpoint.']
    for step in (0,50,150):
        paths=[r/f'stegastamp/fold{k}/real-{step}.json' for k in (1,2,3)]
        if not all(p.exists() for p in paths):continue
        rows=[row for p in paths for row in json.loads(p.read_text())['per_image']];assert len(rows)==len({q['image_name'] for q in rows})==170
        summary=summarize(rows);out={'step':step,'summary':summary,'per_image':rows,'protocol':'Three-fold OOF, fixed150 endpoint; reused development folds, one seed'}
        if step:
            b=json.loads((r/'ss-oof-0.json').read_text());lookup={q['image_name']:q for q in b['per_image']};delta=np.array([q['miou']-lookup[q['image_name']]['miou'] for q in rows]);rng=np.random.default_rng(42);out['paired_delta']={'mean':float(delta.mean()),'ci95':np.quantile(delta[rng.integers(170,size=(5000,170))].mean(1),[.025,.975]).tolist(),'improved':int((delta>1e-8).sum()),'worse':int((delta<-1e-8).sum())}
        atomic_json(r/f'ss-oof-{step}.json',out);g=summary
        lines+=['',f"Update{step}: mIoU**{g['all']['miou']:.6f}**, FG≤1%{g['fg_le1']['fg_iou']:.6f}, FG1–5%{g['fg_1to5']['fg_iou']:.6f}, clean≤1%{g['clean_le1']['clean_iou']:.6f}, clean1–5%{g['clean_1to5']['clean_iou']:.6f}, clean on WM>50%{g['wm_over50']['clean_iou']:.6f}."]
        if step:lines.append(f"Paired change: `{json.dumps(out['paired_delta'])}`.")
        measures.append({'family':'stegastamp','name':f'A4 real adaptation / update{step}','protocol':out['protocol'],'miou':g['all']['miou'],'note':f"FG<=1%{g['fg_le1']['fg_iou']:.5f};FG1–5%{g['fg_1to5']['fg_iou']:.5f};clean on WM>50%{g['wm_over50']['clean_iou']:.5f}."})
    if not (r/'ss-oof-150.json').exists():lines+=['','**Training has not yet produced a complete150-update OOF result. No adaptation gain is claimed.**']
    else:
        before=json.loads((r/'ss-oof-0.json').read_text())['summary'];after=json.loads((r/'ss-oof-150.json').read_text())['summary']
        lines+=['','| Validation group | n | Before | After150 | Complete misses before / after |','|---|---:|---:|---:|---:|']
        for group,metric,miss in [('all','miou',None),('fg_le1','fg_iou','fg_miss'),('fg_1to5','fg_iou','fg_miss'),('clean_le1','clean_iou','clean_miss'),('clean_1to5','clean_iou','clean_miss'),('wm_over50','clean_iou','clean_miss')]:
            b,c=before[group],after[group];count=f"{b[miss]} / {c[miss]}" if miss else '—'
            lines.append(f"| {group} / {metric} | {c['n']} | {b[metric]:.6f} | {c[metric]:.6f} | {count} |")
        lines+=['','All three folds improved at the predeclared endpoint. Tiny clean-hole mean IoU improved, but complete misses increased from3 to4: average gains do not mean every failure was fixed. The ten validation gallery panels include largest gains, largest regressions and tiny-region failures; [per-image gallery index](results/gallery/index.json).', '', 'The reproduction check passed before training: largest archived-probability difference across folds was5.25e-6. This verifies this cluster run, not the separate historical RTX3070 reproduction discrepancy.']
        costs=[json.loads((r/f'stegastamp/fold{k}/completed.json').read_text()) for k in (1,2,3)]
        lines+=['',f"Compute: three concurrent single-GPU workers;450 optimizer updates total,7200 image views total (5400 synthetic replay +1800 real draws). Fixed2500-source synthetic pool, zero new embedding calls. Per-fold wall seconds: {[round(x['seconds'],1) for x in costs]}; training seconds: {[round(x['training_seconds'],1) for x in costs]}. Actual peak allocated model memory was13.05GB per GPU. The fourth GPU completed the three-family oracle diagnostics and was then idle until the training workers released the node."]
    selected=r/'selected_recipe.json'
    if selected.exists():
        cfg=json.loads(selected.read_text());lines+=['','## Frozen submission candidate','',f"Selected150-update adaptation: {cfg['needs_all170_refit']}. Postprocessing: `{cfg['postprocess']}`; fixed1% fallback affects only completely empty/full predictions. OOF mIoU before fallback{cfg['validation_miou_before_fallback']:.6f}, after fallback{cfg['validation_miou_final']:.6f}. The fallback adds{cfg['fallback_selection']['mean_delta']:+.6f}; paired95% interval{cfg['fallback_selection']['ci95']}. This remains exploratory selection on reused development folds.", '', 'Final refit uses all170 StegaStamp real images with the same150-update replay policy. It has no held-out validation score. Four-GPU DataParallel preserves effective batch16 and the optimizer-update count; minor floating-point reduction differences from the single-GPU folds are possible. Keep PixelSeal and TrustMark test masks from#2675 bit-for-bit. No new threshold tuning, TTA or test-label use. [Frozen recipe](results/selected_recipe.json).']
        measures.append({'family':'stegastamp','name':'A4 selected adaptation + fixed fallback','protocol':'Three-fold OOF; exploratory selection on reused development folds','miou':cfg['validation_miou_final'],'note':'Fixed150-update endpoint plus fixed1% fallback on empty/full predictions only.'})
    receipt=r/'submission/server_receipt.json'
    if receipt.exists():
        response=json.loads(receipt.read_text())['response'];lines+=['','## Leaderboard receipt','',f"Server response: `{json.dumps(response)}`. Only StegaStamp predictions changed; the public score is a separate test subset, not the OOF metric. [Submission report](SUBMISSION_REPORT.md)."]
    if (r/'extension_protocol.json').exists():
        lines+=['','## One bounded extension to450 updates','', 'The150-update model is preserved and is the first submission candidate. A fixed450 endpoint was declared before its results: restore150 model/optimizer/RNG, keep the same replay and learning rates, train300 more updates. Three fold workers plus one all170 worker share one4-A100 node. Select only for paired OOF gain>=0.005, positive bootstrap lower bound and no>0.01 regression in FG<=5% or clean IoU on WM>50%. The all170 worker supplies no validation evidence. Public scores do not enter selection. [Predeclared extension](results/extension_protocol.json).']
        extended=r/'extended/evaluation.json'
        if extended.exists():
            d=json.loads(extended.read_text());lines+=['',f"Measured450 raw mIoU{d['raw_summary']['all']['miou']:.6f}; with inherited postprocessing{d['summary']['all']['miou']:.6f}. Change versus selected150: `{json.dumps(d['paired_delta'])}`. Minority guard passed: {d['minority_guard_passed']}; selected: {d['selected']}."]
            measures.append({'family':'stegastamp','name':'A4 bounded450 continuation','protocol':'Same170 development OOF; inherited fixed postprocess; endpoint declared before450 results','miou':d['summary']['all']['miou'],'note':f"Selected: {d['selected']}; delta versus150 {d['paired_delta']['mean']:+.6f}; no public-score selection."})
        else:lines+=['','The450 continuation has not produced its final validation result.']
    lines+=['','## A2: oracle ladder','',
        'GT defines diagnostic views only. Successful crop decoding does not prove the original edit type; failed decoding does not prove physical signal absence. Outside-GT blur/mean replacements change context and produce boundaries. The official wrapper performs one native resize (PS256/TM245/SS400), without an extra intermediate256 resize.']
    for family in ('pixelseal','trustmark','stegastamp'):
        p=r/f'oracle/{family}.json'
        if not p.exists():continue
        d=json.loads(p.read_text());lines+=['',f'### {family}','', '| Group | n | Full ≥90% | GT bbox ≥90% | Outside GT blurred ≥90% | Outside GT mean ≥90% |','|---|---:|---:|---:|---:|---:|']
        for group,v in d['summary'].items():lines.append(f"| {group} | {v['n']} | {v['full']['n_ge90']} | {v['GT_bbox_direct_native']['n_ge90']} | {v['outside_GT_blurred']['n_ge90']} | {v['outside_GT_mean']['n_ge90']} |")
    if all((r/f'oracle/{family}.json').exists() for family in ('pixelseal','trustmark','stegastamp')):
        lines+=['','At the unchanged90% gate, exact GT boxes recover3/43 small PixelSeal,21/45 small TrustMark and8/45 small StegaStamp cases. Both PS and SS are below the proposed20% small-region crop-search milestone. Outside-mask blur/mean recovers zero small cases at this gate for all three families. Therefore this audit does not justify a large new PS/SS crop search or promise that masking background will recover small fragments. It also does not prove there is no local presence signal. Existing TrustMark verification is preserved.']
    lines+=['','## What the audit overstated','',
        'Re-embedding and gradient×HPF were already measured in message_localizer: real re-embedding macro AUC0.49812 PS /0.51902 TM. Two-class per-image losses and an unchanged frozen verifier already exist. A verified payload is not a pixel-perfect pseudo-mask, particularly when clean holes remain inside a decodable crop. The independent fair-coin null is not sufficient to lower a search verifier threshold after many correlated/adaptive crops.','',
        'A63-bit grayscale DCT pHash search found no pairs within Hamming distance4 among510 training images. This limited check does not prove absence of edited siblings. There is no test self-training and no use of held-out real masks in a synthetic mask bank.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');atomic_json(r/'measurements.json',measures);print({'measured_rows':len(measures),'restoration_error_share':restoration_error/a['sum_weighted_error']})


if __name__=='__main__':main()
