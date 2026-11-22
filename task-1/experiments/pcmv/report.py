"""Publish only observed metrics and accepted server receipts."""
import json
from pathlib import Path
from experiments.native_localizer.common import atomic_json


def main():
    root=Path(__file__).parent;results=root/'results';measure=[];lines=['# PCMV: one fixed-fold experiment and stronger PixelSeal deployment','','Current work uses the strongest verified deployable PixelSeal baseline. TrustMark and StegaStamp remain exactly as in #3054. One PixelSeal scorer screen is planned; no full cross-validation or backbone search. GPUs0–1 only; GPUs2–3 reserved for the user’s other experiment.','']
    p=results/'base-candidate/evaluation.json'
    if p.exists():
        d=json.loads(p.read_text());decision=p.with_name('decision.json');selected=json.loads(decision.read_text())['selected'] if decision.exists() else d['selected_for_test'];lines+=['## Stronger checkpoint verification','',f"EMA1561 global/carrier branch, no refiner or Noiseprint inference inputs. Same57-image development fold: {d['baseline']['all']['miou']:.6f} → **{d['summary']['all']['miou']:.6f}**, delta {d['delta']:+.6f}, paired95% CI {d['ci95']}. Selected for test: {selected}. This checkpoint trained113 real images plus synthetic replay; no all170 refit. Reused fold and checkpoint selection create optimism; this is not a fresh held-out estimate.",'','The first short job completed with the earlier minority guard and produced no test predictions. Per the user’s priority to use the strongest overall model, the subsequent decision retained this checkpoint despite FG1–5% IoU falling0.012724→0. Global validation and public scores improved; the tiny-watermark problem remains unresolved. The original decision/evaluation artifacts are preserved.','', '| Bucket | n | Metric | Previous | Candidate |','|---|---:|---|---:|---:|']
        for group,metric in [('fg_le1','fg_iou'),('fg_1to5','fg_iou'),('fg_5to20','fg_iou'),('fg_gt20','fg_iou'),('clean_le1','clean_iou'),('clean_1to5','clean_iou'),('wm_over50','clean_iou')]:
            a,b=d['baseline'][group],d['summary'][group];lines.append(f"| {group} | {b['n']} | {metric} | {a[metric]:.6f} | {b[metric]:.6f} |")
        measure.append({'family':'pixelseal','name':'Global carrier EMA1561 / exact inference','protocol':'Repeated57-image development fold, one seed,113-real training; no all170 refit','miou':d['summary']['all']['miou'],'note':f"Delta{d['delta']:+.6f}; minority guard{d['minority_guard']}; per-bucket regressions retained in report."})
        receipt=results/'base-candidate/submission/server_receipt.json'
        if receipt.exists():
            r=json.loads(receipt.read_text());response=r['response'];lines+=['','## Leaderboard response','',f"Actual HTTP status: {r['http_status']}; response: `{json.dumps(response)}`."]
            if r['http_status']==200 and response.get('status')=='success':atomic_json(receipt.with_name('upload_receipt.json'),dict(r,response=json.dumps(response)))
    else:lines+=['Checkpoint verification is pending. No new validation score or leaderboard gain is claimed.']
    p=results/'screen-v1/evaluation.json'
    if p.exists():
        d=json.loads(p.read_text());lines+=['','## Single scorer fit / final evaluation','',d['protocol'],'','| Variant | mIoU | Delta vs strongest base | Paired95% CI |','|---|---:|---:|---|',f"| Base | {d['base']['all']['miou']:.6f} | — | — |"]
        for name,v in d['variants'].items():
            lines.append(f"| {name} | {v['summary']['all']['miou']:.6f} | {v['delta_vs_base']:+.6f} | {v['ci95_vs_base']} |")
            measure.append({'family':'pixelseal','name':'PCMV '+name,'protocol':'One57-image developmental fold, inner action calibration, final evaluation','miou':v['summary']['all']['miou'],'note':f"Delta{v['delta_vs_base']:+.6f}; anti-shortcut{v['anti_shortcut']}"})
        lines+=['',f"Exact one/two-disjoint-toggle oracle (GT-only diagnostic): {d['one_toggle_oracle_miou']:.6f} / {d['two_toggle_oracle_miou']:.6f}. Deployment allows at most one toggle. Candidate for further inference: {d['candidate_for_further_inference']}. Full buckets, miss counts, actions and wrong-message swap are saved in evaluation.json. The wrong-message test holds the existing base/proposals fixed; it tests only new message features."]
    else:lines+=['','PCMV fitting/evaluation has not completed. Source implementation and planned budget are not measured results.']
    lines+=['','## Scope and safeguards','','See PROTOCOL.md for the frozen screening budget and SOURCES.md for pinned official implementation references. SAM2 resizing is restricted to geometry proposals. Official watermark decoder probes retain original512 coordinates, exact uint8 replacement bytes and the original decoder preprocessing. Counterfactual boundaries may introduce distribution shifts; scores are learned features, not calibrated Gaussian significance. No GT enters proposal generation or feature extraction. Prior submissions and checkpoints remain preserved.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');atomic_json(results/'measurements.json',measure)


if __name__=='__main__':main()
