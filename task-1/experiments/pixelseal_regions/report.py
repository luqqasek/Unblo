"""Publish measured screening artifacts without inferring leaderboard scores."""
import json
from pathlib import Path


def main():
    here=Path(__file__).parent;out=here/'results';out.mkdir(exist_ok=True);files=sorted(out.glob('evaluation-*.json'),key=lambda p:int(p.stem.split('-')[-1]));measure=[]
    lines=['# PixelSeal: native whole-region presence screen','','This experiment keeps submission #3188 (public mIoU0.6736) as its baseline. Known-geometry recognition and search coverage are diagnostics; only GT-free masks count toward validation mIoU. One repeatedly inspected57-image development fold, not full OOF.','', '[Predeclared protocol](PROTOCOL.md). Exact original embedder and quantization; fresh trainable region representation; no90%-payload gate. Real clean-source generation and new model fitting exclude the outer validation fold.']
    p=out/'geometry_coverage.json'
    if p.exists():
        g=json.loads(p.read_text());lines+=['','## Geometric coverage only','',f"The predefined search contains {g['candidates']:,} candidates per512 image before local refinement. GT is used only to score this coverage, never to choose candidates at inference.",'','| Group | n | Candidates covering GT at IoU≥0.5 | Mean best IoU |','|---|---:|---:|---:|']
        for k in ('fg_le1','fg_1to5'):
            r=g[k];lines.append(f"| {k} | {r['n']} | {r['recall_iou05']} | {r['mean_best_iou']:.6f} |")
        lines+=['','This removes the specific tiny-rectangle proposal coverage failure of PCMV. It does not establish that the trained model identifies the correct rectangle.']
    if files:
        d=json.loads(files[-1].read_text());lines+=['','## Completed fixed-fold evaluation','',d['protocol'],'',f"Baseline {d['baseline']['all']['miou']:.6f} → candidate **{d['summary']['all']['miou']:.6f}**, delta {d['delta']:+.6f}; paired95% bootstrap {d['ci95']}. Applied corrections: {d['applied']}/57. Threshold selected on inner calibration: {d['selected_threshold']}. Candidate for test inference: {d['candidate_for_test']}.",'','| Bucket | n | Metric | Baseline | Candidate | Complete misses before / after |','|---|---:|---|---:|---:|---|']
        for k,m,miss in [('fg_le1','fg_iou','fg_miss'),('fg_1to5','fg_iou','fg_miss'),('fg_5to20','fg_iou','fg_miss'),('fg_gt20','fg_iou','fg_miss'),('clean_le1','clean_iou','clean_miss'),('clean_1to5','clean_iou','clean_miss'),('wm_over50','clean_iou','clean_miss')]:
            b=d['baseline'][k];r=d['summary'][k];lines.append(f"| {k} | {r['n']} | {m} | {b[m]:.6f} | {r[m]:.6f} | {b[miss]} / {r[miss]} |")
        lines+=['','## Held-out recognition at known geometry','','| Source/group | Pairs | ROC-AUC | Marked scores above exact clean twin |','|---|---:|---:|---:|']
        for k,r in d['recognition'].items():lines.append(f"| {k} | {r['n_pairs']} | {r['auc']} | {r['paired_order_accuracy']} |")
        measure.append({'family':'pixelseal','variant':'Native region presence / inner-calibrated search','protocol':'One57-image reused development fold;90 real training /23 calibration; fixed endpoint','miou':d['summary']['all']['miou'],'note':f"Delta {d['delta']:+.6f}; corrections {d['applied']}; test candidate {d['candidate_for_test']}"})
        lines+=['','Known-box recognition uses privileged geometry and cannot be read as a deployable detection score. The search operates without GT. Regional source variants are clustered by original cover; paired accuracy is not independent pixel evidence.']
    else:lines+=['','## Status','','Training/evaluation has not completed. No new model mIoU or submission is claimed.']
    p=out/'precision_audit.json'
    if p.exists():
        r=json.loads(p.read_text());lines+=['','## Matched precision audit','',f"Frozen EMA400, same182 source-held-out pairs and known geometry, evaluated on {r['device']}. This is an inference diagnostic, not an FP32 retraining experiment.",'','| Arithmetic | ROC-AUC | Pair ties | Pair ordering (ties count half) |','|---|---:|---:|---:|']
        for k,v in r['summary'].items():lines.append(f"| {k} | {v['auc']:.6f} | {v['tie_fraction']:.3%} | {v['paired_auc_with_half_ties']:.6f} |")
        lines+=['','BF16 coarsens the final readout, but full FP32 still gives near-chance recognition. This checkpoint has not learned useful held-out presence recognition. That does not prove PixelSeal is undetectable or rule out a longer, differently designed training run.']
    p=out/'config.json'
    if p.exists():
        r=json.loads(p.read_text());lines+=['','## Data and optimizer exposure','',f"{r['trainable_parameters']:,} trainable parameters. Effective batch12:8 synthetic paired views and4 real training images. At400 updates this is4,800 views (3,200 paired views and1,600 real draws), not4,800 unique images. The generated bank contains556 training pairs from331 sources,118 calibration pairs from82 sources and64 audit pairs from64 sources. All90 real training images remain eligible for supervised training; only75 yield valid wholly clean embedding windows. Calibration has23 real sources;18 yield such windows. There are738 actual exact embedding calls and no post-embedding image resampling."]
    p=out/'completed.json'
    if p.exists():
        r=json.loads(p.read_text());lines+=['','## Executed budget','',f"Completed update {r['step']}; elapsed {r['seconds_total']/60:.2f} minutes inside the process; {r['gpu_processes']} GPU workers. Job {r['job_id']}. Stop reason: {r['stopped_reason']}. Queue and local implementation time excluded."]
        lines+=['','Training stopped at the predeclared400-update recognition gate. No test inference or leaderboard submission was made because the selected pipeline applies zero corrections. Submission#3188 remains the best accepted artifact. The A100 job completed and released its node. The Windows fallback process produced no training updates; Windows RTX3070 was used only for the completed precision audit.','', 'Executed cluster source is preserved with hashes in `results/execution_provenance.json` and `results/executed_source/`. Current entrypoints additionally support one GPU at the same effective batch and remove an unnecessary evaluator import; the trained recipe is unchanged.','', 'Reproduction in an allocated two-GPU job: `sbatch experiments/pixelseal_regions/run.sh`. The script uses the original source paths and skips already generated pairs. Set the reservation to the permitted queue for the account. The checkpoint includes weights, optimizer and scheduler state; automatic resume is not implemented.','', '![Training loss and held-out recognition](results/learning_curves.png)']
    (here/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'measurements.json').write_text(json.dumps(measure,indent=2)+'\n')


if __name__=='__main__':main()
