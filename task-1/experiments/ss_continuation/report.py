"""Report fixed-fold continuation separately from historical full OOF."""
import json
from pathlib import Path


def main():
    here = Path(__file__).parent
    out = here / 'results'
    out.mkdir(exist_ok=True)
    lines = ['# StegaStamp: bounded450→1000 continuation', '',
             '[Frozen protocol](PROTOCOL.md). One fixed57-image validation fold and one matching all170 refit, at most two A100 workers. Restore the best450-step weights, optimizer and RNG; unchanged12 synthetic +4 real views/update, LRs, loss, FP32 and pixel processing.', '',
             'The earlier150→450 run improved full170-image OOF0.724414→0.744788. That motivates this extension but does not predict its gain. This experiment is a single-fold screen and must not be compared directly with the full-OOF number.', '',
             'Two candidates were declared before training: raw1000 and equal mean probabilities450/1000. Threshold0.5 and the previously selected empty/full top1% fallback are fixed. Public scores do not choose the endpoint, blend or threshold.']
    measures = []
    p = out / 'evaluation.json'
    if p.exists():
        r = json.loads(p.read_text())
        lines += ['', '## Final validation', '', f"Matched baseline mIoU: {r['baseline']['all']['miou']:.6f}. Selected: {r['selected']}.", '',
                  '| Candidate | mIoU | Delta | Paired95% CI | Minority guard | Passed |', '|---|---:|---:|---|---|---|']
        for name, v in r['candidates'].items():
            lines.append(f"| {name} | {v['summary']['all']['miou']:.6f} | {v['delta']:+.6f} | {v['ci95']} | {v['minority_guard_passed']} | {v['passed']} |")
            measures.append({'family': 'stegastamp', 'variant': name, 'protocol': 'One reused57-image fold; declared1000 endpoint; inherited fallback', 'miou': v['summary']['all']['miou'], 'note': f"Delta{v['delta']:+.6f}; selected={r['selected'] == name}"})
        for name, v in r['candidates'].items():
            lines += ['', f'### {name}: minority metrics', '', '| Group | n | Metric | Before | After | Misses before / after |', '|---|---:|---|---:|---:|---:|']
            for key, metric, miss in [('fg_le1','fg_iou','fg_miss'),('fg_1to5','fg_iou','fg_miss'),('fg_5to20','fg_iou','fg_miss'),('fg_gt20','fg_iou','fg_miss'),('clean_le1','clean_iou','clean_miss'),('clean_1to5','clean_iou','clean_miss'),('wm_over50','clean_iou','clean_miss')]:
                a, b = r['baseline'][key], v['summary'][key]
                lines.append(f"| {key} | {b['n']} | {metric} | {a[metric]:.6f} | {b[metric]:.6f} | {a[miss]} / {b[miss]} |")
    else:
        lines += ['', 'Training/evaluation is not yet complete. No new endpoint score or submission is claimed.']
    p = out / 'completed.json'
    if p.exists():
        cost = json.loads(p.read_text())
        lines += ['', f"Completed in {cost['seconds']/60:.2f} minutes using two A100 workers on one node (job{cost['job_id']}); the sum of allocated GPU time is approximately {2*cost['seconds']/60:.2f} GPU-minutes. Training used550 additional updates per worker and8800 image views per worker; the all170 refit is not validation evidence."]
    p = out / 'submission/assembly_validation.json'
    if p.exists():
        assembly = json.loads(p.read_text())
        lines += ['', f"Prepared all340 StegaStamp test predictions and replaced that family in submission#{assembly['base_submission']}. PixelSeal and TrustMark remain bit-identical to that newer validated base. The earlier#3188 model parent is retained in recipe provenance. Submission validation passed1020 unique names, binary uint8[1020,128,128], and<20MB."]
    p = out / 'submission/server_receipt.json'
    if p.exists():
        r = json.loads(p.read_text())
        lines += ['', '## Actual public receipt', '', f"HTTP{r['http_status']}: `{json.dumps(r['response'])}`. Hidden final performance is unknown."]
    (here / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out / 'measurements.json').write_text(json.dumps(measures, indent=2)+'\n')


if __name__ == '__main__':
    main()
