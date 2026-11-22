"""Build the report only from saved screening outputs; no selection on test scores."""
import json
from pathlib import Path
from experiments.native_localizer.common import atomic_json

def main():
    root=Path(__file__).parent;r=root/'results';paths=sorted(r.glob('real-*.json'),key=lambda p:int(p.stem.split('-')[-1]));lines=['# Shared IML-ViT clean-hole adaptation','','[Frozen protocol](PROTOCOL.md), [official source and checkpoint provenance](../next_ideas/VERIFIED_FORENSICS_RESEARCH.md). Fixed original fold1 across three families; repeatedly used development data. No test labels, boxes or per-image GT-selected polarity at inference. No new submission is implied by this report.'];measure=[]
    if (r/'probe.json').exists():
        p=json.loads((r/'probe.json').read_text());lines+=['','## Integration and numerical checks','',f"Official checkpoint strict-load passed: {p['checkpoint_strict_loaded']}; {p['parameters']:,} parameters. Raw normalization and first image projection FP32, deeper BF16. Probability difference versus FP32 on the probe image: mean{p['fp32_bf16_probability_mean']:.6f}, max{p['fp32_bf16_probability_max']:.6f}. These are not identical predictions. Native512 versus official padded1024 mean probability difference{p['native_vs_padded_probability_mean_abs']:.6f}; pixels are never resized. Cold probe latencies include initialization and are not comparable throughput benchmarks."]
    if (r/'replica_check.json').exists():lines+=['',f"Single-GPU versus4-GPU replica check: `{(r/'replica_check.json').read_text().strip()}`."]
    lines+=['','The first attempt failed before any optimizer update because upstream FPN stores ordinary Python-list aliases to registered layers; DataParallel used GPU0 weights in another replica. The wrapper now resolves registered simfp modules on each replica, preserving operations and checkpoint keys. A second startup reached validation0 but stopped on JSON logging of a NumPy integer; the counter now uses a Python int. Both failures occurred before training. Prior oracle JSON row availability was GT-filtered; it is not used for routing. Full-canvas agreement is newly computed for all171 images. Accepted historical TM rectangles were audited:28 total, max area9.613%; an explicit15% assertion protects integration.','', '## Measured validation','', '| Update | Mean mIoU | Baseline | Delta | Paired95% CI | Clean IoU at WM>50% | Routed images |','|---|---:|---:|---:|---|---:|---:|']
    for path in paths:
        d=json.loads(path.read_text());s=d['summary'];b=d['baseline'];lines.append(f"| {d['step']} | {s['all']['miou']:.6f} | {b['all']['miou']:.6f} | {d['paired_delta']['mean']:+.6f} | {d['paired_delta']['ci95']} | {s['wm_over50']['clean_iou']:.6f} | {d['routed']} |")
        for family,v in d['by_family'].items():
            measure.append({'family':family,'name':f'IML shared hole adaptation / update{d["step"]}','protocol':'Fixed original fold1, equal blend, original full-message gate, native512; exploratory reused validation','miou':v['after']['all']['miou'],'note':f"Family baseline{v['before']['all']['miou']:.6f}; clean IoU WM>50%{v['after']['wm_over50']['clean_iou']:.6f}."})
    if not paths:lines+=['','No validation result exists yet. No quality gain is claimed.']
    else:
        d=json.loads(paths[-1].read_text());lines+=['','## Latest per-family and minority metrics','', '| Family | Group/metric | n | Baseline | Current | Complete misses before/current |','|---|---|---:|---:|---:|---|']
        for f,v in d['by_family'].items():
            for g,k,miss in [('all','miou',None),('fg_le1','fg_iou','fg_miss'),('fg_1to5','fg_iou','fg_miss'),('fg_5to20','fg_iou','fg_miss'),('fg_gt20','fg_iou','fg_miss'),('clean_le1','clean_iou','clean_miss'),('clean_1to5','clean_iou','clean_miss'),('wm_over50','clean_iou','clean_miss')]:
                b,a=v['before'][g],v['after'][g];m=f"{b[miss]}/{a[miss]}" if miss else '—';lines.append(f"| {f} | {g}/{k} | {a['n']} | {b[k]:.6f} | {a[k]:.6f} | {m} |")
        lines+=['',f"Routed non-restoration images (offline diagnostic): {d['routed_non_restoration']}. Macro per-image AP/AUC, defined counts and full per-image changes are retained in `{paths[-1].name}`. The fixed gate uses original message agreement plus predicted baseline area; GT is used only for scoring/diagnostics. All unrouted masks are preserved exactly, including SS empty/full fallback and TM verified small rectangles."]
    if (r/'completed.json').exists():
        c=json.loads((r/'completed.json').read_text());lines+=['','## Decision and cost','',f"Actual completion: `{json.dumps(c)}`. No test inference was performed by this screen. A positive flag is development evidence, not a promised leaderboard score."]
    if (r/'config.json').exists():
        c=json.loads((r/'config.json').read_text());lines+=['',f"Training uses{c['real_training_images']} real restoration images from fold-training only and384 hard-composited synthetic views from existing full-image encodings. Effective batch8:4 real +4 synthetic. All91.8M parameters are trainable; one AdamW, encoder LR1e-5, decoder LR3e-5, existing per-image halfBCE/halfdualDice. BN running stats fixed; no dropout, geometric/photo augmentation or new embedder calls. Source/weight hashes and runtime versions: `config.json`. Same-content and cross-cover hole edits preserve uint8 values outside the declared edit mask."]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');atomic_json(r/'measurements.json',measure);print({'iml_measurement_rows':len(measure)})
if __name__=='__main__':main()
