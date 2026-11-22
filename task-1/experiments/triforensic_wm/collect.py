"""Build report/dashboard rows from measured artifacts, including regressions."""
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent;results=root/'results';measurements=[];lines=['# TriForensic-WM measured screening','',
        'Single fixed fold1:113 training and57 validation images per family. Reused development validation; not independent test performance. Public best remains #2675=0.6440.','',
        'One unified optimizer/OneCycle loop, co-batched full images and matched patches. No staged freezing or separate patch pretraining. See [protocol](PROTOCOL.md).','',
        '| Family / update | mIoU | Delta vs original baseline | FG ≤1% | FG1–5% | Clean ≤1% | Clean1–5% | Clean on WM>50% |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for family in ('pixelseal','trustmark'):
        for run in sorted((results/family).glob('*')) if (results/family).exists() else []:
            bpath=run/'real-baseline.json'
            if not bpath.exists():continue
            baseline=json.loads(bpath.read_text());b=baseline['summary']['all']['miou']
            paths=sorted(run.glob('real-*.json'),key=lambda p:-1 if p.stem=='real-baseline' else int(p.stem.split('-')[-1]))
            for p in paths:
                d=json.loads(p.read_text());g=d['summary'];m=g['all']['miou'];step=p.stem.removeprefix('real-')
                row={'family':family,'run':run.name,'step':step,'miou':m,'delta':m-b,'groups':g,'local_groups':d['local_summary'],'protocol':'Single fixed fold1 screening,57 images','artifact':str(p.relative_to(root))};measurements.append(row)
                fmt=lambda v:'undefined' if v is None else f'{v:.5f}'
                lines.append(f"| {family} / {step} | {m:.5f} | {m-b:+.5f} | {fmt(g['fg_le1']['fg_iou'])} | {fmt(g['fg_1to5']['fg_iou'])} | {fmt(g['clean_le1']['clean_iou'])} | {fmt(g['clean_1to5']['clean_iou'])} | {fmt(g['wm_over50']['clean_iou'])} |")
            gate=run/'gate1000.json'
            if gate.exists():lines+=['',f'{family} / {run.name} gate1000: `{gate.read_text().strip()}`.']
            end=run/'completed.json'
            if not end.exists():end=run/'paused.json'
            if end.exists():
                e=json.loads(end.read_text());lines+=['',f"{family} / {run.name}: {e['step']} optimizer updates; status `{e['stop']}`; measured runtime {e['seconds']/60:.2f} minutes; {e['views']} counted views."]
            else:lines+=['',f'{family} / {run.name}: **training in progress**; the rows above are measured interim checkpoints, not a completed experiment.']
            curve=run/'learning-curves.png'
            if curve.exists():lines+=['',f"[{family} learning curves]({curve.relative_to(root).as_posix()})."]
            comparison=run/'comparison-2675.json'
            if comparison.exists():
                c=json.loads(comparison.read_text());lines+=['',f"Compared with the newer #2675 recipe on the SAME57 images: {c['baseline_2675_samefold']:.6f} → {c['triforensic_1561']:.6f} (delta{c['delta']:+.6f}). This is validation, not a new public score."]
            fusion=run/'fusion-audit-1000.json'
            if fusion.exists():lines+=['','Fusion audit at1000: the final refiner adds only about0.000042 mIoU over its own pre-refiner base. On tiny watermark GT cells, base logits average−5.300 and the additive correction is only+0.009. Local AUC improvement has NOT become successful small-mask segmentation. The global branch also receives learned detached local skips, so no isolated causal claim about backbone versus local features is justified.']
    lines+=['','## Interpretation and limits','',
        'Initialization/update0 is not a trained result. FP32 original baseline and the mixed-precision initialization are reported separately. The final mask metric and local-only diagnostic are distinct; a high local AUC alone does not establish successful segmentation. Fixed threshold0.5; no validation-driven threshold search.','',
        'Read per-image JSON for complete misses,AP/AUC and rectangular/irregular breakdowns. Full/empty synthetic controls and low-observability false vetoes are separate diagnostics. Synthetic interior AUC excludes cells whose RF29 touches a seam or patch padding; strips narrower than29 cannot contribute interior cells.','',
        'No new submission is produced by this screening report. Large checkpoints,full probability maps and exact uint8 pools remain on JURECA under native-localizer/triforensic-{runs,data}.','',
        'The first training launch failed before optimization due to a missing synchronized verifier module. The dependency was copied,imports checked,and training restarted. No failed-launch result is presented as a trained score.']
    lines+=['','## Early stopping and production selection','',
        'PixelSeal was stopped at saved update1561 following the user\'s instruction to avoid prolonged training without targeted gains. mIoU0.581635 improves over the previous matched screen0.570883, but all16 watermark cases≤5% and all4 clean-hole cases≤1% remain complete misses. Clean1–5% IoU0.105786 is above the original baseline0.083069 but below update1000\'s0.122340. This does not solve tiny localization.','',
        'A four-GPU attempt to disable carrier activation recomputation was slower despite higher VRAM usage. It was cancelled; discarded updates are archived separately and are not part of the retained checkpoint. Subsequent runs keep recomputation enabled.','',
        'The selected PixelSeal recipe is queued for an all170-image refit with synthetic replay, the same3500-step OneCycle schedule stopped at1561, and fresh new branches. This is final fitting, not a second training phase of the screening model. Test prediction and one authorized upload follow artifact validation; no new public score is claimed here.']
    results.mkdir(exist_ok=True);(results/'measurements.json').write_text(json.dumps(measurements,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(lines)+'\n');print({'measured_rows':len(measurements)})
if __name__=='__main__':main()
