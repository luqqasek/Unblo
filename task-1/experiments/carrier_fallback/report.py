"""Publish the cached screen, deployment check, and actual upload receipt."""
import json
from pathlib import Path


def main():
    here = Path(__file__).parent
    out = here / 'results'
    r = json.loads((out / 'evaluation.json').read_text())
    lines = ['# PixelSeal: fixed local-carrier fallback', '',
             'Starting system: accepted submission #3188, public mIoU0.6736. This screen uses the existing EMA1561 local branch, not the rejected new region classifier. No model training, threshold sweep or GT-dependent region placement.', '',
             '[Predeclared protocol](PROTOCOL.md). One reused57-image development fold, not full OOF. Only9 completely empty base predictions are eligible. The inherited global prediction remains unchanged everywhere else.', '',
             '| Candidate | mIoU | Delta | Paired95% CI | Improved / worse | Selected |',
             '|---|---:|---:|---|---:|---|']
    measures = []
    for name, v in r['candidates'].items():
        lines.append(f"| {name} | {v['summary']['all']['miou']:.6f} | {v['delta']:+.6f} | {v['ci95']} | {v['improved']} / {v['worse']} | {name == r['selected']} |")
        measures.append({'family': 'pixelseal', 'variant': name, 'protocol': 'Cached fixed57-image development screen, two predeclared rules', 'miou': v['summary']['all']['miou'], 'note': f"Delta{v['delta']:+.6f}; selected={name == r['selected']}"})
    lines += ['', 'The selected rule emits the164 highest local-carrier-logit cells when the base mask is empty. The dense rectangle rule instead maximizes interior-versus-ring contrast times square-root area; it regressed and was rejected. Neither changes raw image pixels.', '',
              'The cached selected result reduces FG complete misses on ≤5% masks from16 to8. Mean FG IoU remains low:0.05452 on≤1% and0.01382 on1–5%. Most of the global gain comes from one image; this is a modest exploratory improvement, not a solution to tiny localization.', '',
              '| Group | n | Baseline | Selected | FG misses before / after |',
              '|---|---:|---:|---:|---:|']
    for key in ('fg_le1', 'fg_1to5', 'fg_5to20', 'fg_gt20'):
        a = r['baseline'][key]
        b = r['candidates'][r['selected']]['summary'][key]
        lines.append(f"| {key} | {b['n']} | {a['fg_iou']:.6f} | {b['fg_iou']:.6f} | {a['fg_miss']} / {b['fg_miss']} |")
    lines += ['', 'Clean IoU on WM>50% stays0.267867; no clean-hole case is modified. Image-level bootstrap resamples57 image differences; it does not remove bias from repeatedly inspecting this fold or selecting between two candidates.']
    p = out / 'production/deployment_validation.json'
    if p.exists():
        d = json.loads(p.read_text())
        lines += ['', '## Actual deployment verification', '',
                  f"Same checkpoint hash, independently instantiated carrier/JND, same native pixels and BF16 recipe. Maximum local-logit difference from archived inference: {d['reproduction']['max_abs_logit_difference']}. Some tied top164 rankings differ, so exact mask reproduction failed. We evaluated the actual deployment output before any test prediction, without changing the rule or acceptance gate.", '',
                  f"Actual deployed validation mIoU **{d['summary']['all']['miou']:.6f}**, delta **{d['delta']:+.6f}**, paired95% CI{d['ci95']}; passed gate: {d['passed']}."]
        measures.append({'family': 'pixelseal', 'variant': 'Actual deployment / local top164', 'protocol': 'Recomputed fixed57-image validation; unchanged rule and gate', 'miou': d['summary']['all']['miou'], 'note': f"Delta{d['delta']:+.6f}; gate={d['passed']}; BF16 tie differences reported"})
    p = out / 'production/completed.json'
    if p.exists():
        d = json.loads(p.read_text())
        lines += ['', '## Test predictions', '', f"All340 PixelSeal masks prepared; {d['changed_n']} formerly empty predictions change. Every changed mask has164 positive cells. TrustMark and StegaStamp stay bit-identical to #3188. No test GT, resampling, TTA or new payload handling."]
    p = out / 'submission/server_receipt.json'
    if p.exists():
        d = json.loads(p.read_text())
        lines += ['', '## Server receipt', '', f"HTTP{d['http_status']}; actual response: `{json.dumps(d['response'])}`. This is the public split only; no hidden final score or per-family public score is available."]
    else:
        lines += ['', 'No accepted leaderboard receipt is recorded yet.']
    (here / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out / 'measurements.json').write_text(json.dumps(measures, indent=2)+'\n')


if __name__ == '__main__':
    main()
