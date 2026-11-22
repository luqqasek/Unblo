"""Build the evidence report from completed fixed endpoints only."""
import json
from pathlib import Path

def main():
    here=Path(__file__).resolve().parent;root=here/'results';measurements=[];cost=[]
    lines=['# Local presence: small watermark experiments','',
      'Development screens on one Slurm node at a time, four NVIDIA A10040GB workers. '
      'Production submission#2409/public0.6369 is unchanged. Learned branches use fixed fold1 '
      '(57 validation images per family); frozen proposal screens reuse170-image OOF. '
      'These repeatedly inspected development estimates are not independent hidden-test estimates.','',
      '| Run | Endpoint | n | mIoU | Delta | FG ≤1% | FG 1–5% | FG 5–20% | FG >20% | Clean ≤1% | Clean 1–5% | Clean in >50% WM |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    runs=['pixelseal-presence','trustmark-presence','pixelseal-holes','tm-tiled','ps-spatial-dice','ps-spatial-iou','tm-scan','ps-adapt-control-spatial','ps-adapt-real-spatial','tm-adapt-control-spatial','tm-adapt-real-spatial']
    for run in runs:
        completed=root/run/'completed.json'
        if not completed.exists():continue
        d=json.loads(completed.read_text());family='trustmark' if run.startswith(('trustmark','tm-')) else 'pixelseal'
        protocol='Existing 170-image OOF plus frozen inference' if run in ['tm-tiled','tm-scan'] else 'Single fixed fold1 screening, 57 images'
        for stage in ['baseline','final']:
            g=d[stage];delta=g['all']['miou']-d['baseline']['all']['miou'];measurements.append({'run':run,'family':family,'stage':stage,'protocol':protocol,'groups':g,'delta':delta})
            vals=[g['all']['miou'],delta,g['fg_le1']['fg_iou'],g['fg_1to5']['fg_iou'],g['fg_5to20']['fg_iou'],g.get('fg_gt20',{}).get('fg_iou'),g['clean_le1']['clean_iou'],g['clean_1to5']['clean_iou'],g['wm_over50']['clean_iou']]
            lines.append(f'| {run} | {stage} | {g["all"]["n"]} | '+' | '.join(f'{v:.5f}' if v is not None else 'see per-image JSON' for v in vals)+' |')
        cost.append(f'- {run}: {d["seconds"]:.1f}s worker elapsed; peak PyTorch allocated {d["peak_bytes"]/1e9:.2f}GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.')
    lines+=['','## What changed and what did not','',
      'The initial 1x1 corrections were rejected: PS presence -0.03602, PS clean-hole veto -0.03312, TM presence -0.01969 mIoU. '
      'Class-balanced integration produced too many false positives or false clean holes. '
      'At1000 updates the standalone PS local encoder has tiny-region macro AUC0.74975/AP0.05317 versus base0.52268/0.00678 on the same7 images. '
      'That ranking improvement is not a usable segmentation improvement by itself.','',
      'Spatial aggregation plus ordinary BCE reduced overcorrection: PS Dice +0.00224 and soft-IoU +0.00265 mIoU. '
      'Both recover only1/7 tiny masks; all9 masks in1–5% remain missed. This is small and validation-adaptive, not a robust breakthrough.','',
      'Tiled NRGMark adds only +0.00062 mIoU with13 improvements/9 regressions and no new tiny recovery. '
      'The bounded empty-mask scan instead recovers2 previously missed tiny TrustMark rectangles: overall0.65265→0.65705, tiny FG IoU0.08059→0.24539, misses8/9→6/9. '
      'The remaining168 masks are unchanged, including clean holes and1–5% rectangles. Both exact final crops pass90% raw known-message agreement. '
      'One wrong query per image yields0/18 acceptances over the searched candidates; this is a limited control, not a universal false-positive guarantee. '
      'Paired image-bootstrap95% interval for mean delta is[0,0.01107], reflecting only two changed cases.','',
      'Clean-hole localization remains unresolved. Do not deploy the negative one-sided veto or infer a physical detection limit from this short screen. '
      'The next matched screen continues each local encoder for500 updates:128 synthetic views versus96 synthetic+32 real outer-training patches, '
      'followed by the same600-update spatial correction. Measured PS control0.55803/real0.55937 both trail the earlier spatial0.56013. '
      'TM control0.64795 is +0.00133 over its same-fold base; TM real0.63122 is -0.01539. Real-patch adaptation is rejected in this screen.','',
      'No new submission or test prediction was made in this experiment series. Preserve#2409 and all old masks.','',
      '## Compute','',*cost,'',
      'Three successful Slurm allocations total14m17s node time (approximately0.952 allocated GPU-hours), excluding queues and local preparation. '
      'Slurm accounting is recorded separately. Four independent workers share one node; short spatial workers finish before the longer scan workers. '
      'Do not add parallel worker durations and call that wall time.','',
      '## Protocol','',
      'New local encoder:372737parameters, analytical/gradient-checked RF27, fixed30 SRM(Y) filters plus8 constrained RGB filters, '
      'high-pass RGB, luminance and texture observability proxy; no spatial normalization. Original PS JND stays in the unchanged base. '
      'All raw pixels and local layers areFP32, TF32off. First integration usedFP16 storage for extracted deep features; later spatial screens useFP32 caches.','',
      'Exact default embedders and PNG quantization; up to512 train/128 dev source identities for new encodings. '
      'PixelSeal thin rectangles short12–32px/AR≤16; TrustMark short16–64px/AR≤2. '
      'Separate small-crop embeddings and full-source remnants/clean restorations; matched64px RGB twins, no post-embedding resize/JPEG/blur. '
      'A dedicated strict interior-only/JND-quartile transfer test has not yet been run; patch banks include boundary-straddling examples.','',
      'Local pretraining1000updates×128 views. Initial correction300updates×32views; spatial correction600updates×16views,75%synthetic replay. '
      'Base model frozen, validation never enters optimizer batches, threshold0.5 and final update fixed per screen. '
      'Later experiments were proposed after seeing previous validation results, so this is development selection. '
      'Spatial local-feature pathRF83; the full predictor also receives contextual base logits, so its total RF is not bounded83.','',
      'TrustMark proposal-only tiles:9×256 and36×160, resized1024 only for detection. Verification always original image crops and native245 decoder, raw100bits, noECC. '
      'Empty-mask scan:18 eligible images selected by baseline output,46,642 crops/image (w/h24–80step8,AR≤2,position stride16), '
      'top32 soft agreement proposals, exact refinement4/2/1px, final hard agreement≥90%. GT boxes never seed inference. '
      'GPU batched crop resize agrees with exact resize within1e-4; final verification bypasses the approximate batched path.','',
      'Metrics: official per-image mean of clean/foreground IoU, empty union=1; AP uses sklearn average_precision_score per image, absent positive classes excluded; '
      'ROC-AUC excludes single-class targets. Buckets and complete misses are saved in each completed.json; full per-image outputs are preserved. '
      'Clean-hole buckets mean clean fraction≤1%,1–5%, and all images with watermark coverage>50%.','',
      'Not run: Noiseprint++, annular features, boundary head, RF intervention, full end-to-end refit or new full CV. '
      'The research claims about guaranteed0.70+, exact conditional-median collapse, and zero-risk proposal unions are hypotheses, not established conclusions.','',
      '## Reproduction and sources','',
      'Run each stage inside one Slurm allocation: `python -m experiments.local_presence.job`, then `job2`, then `job3`. '
      'Scripts preserve completed outputs; use separate output directories for reruns. Cluster root and baseline paths are recorded in code/configs. '
      'CPU aggregation: `python -m experiments.local_presence.aggregate_scan`; report: `python experiments/local_presence/report.py`. '
      'Use task-1 cwd and PYTHONPATH=.:src. Pinned environment and unchanged base checkpoint provenance remain in message_localizer/region_data.','',
      '[SRNet](https://ws.binghamton.edu/fridrich/Research/SRNet.pdf), '
      '[NRGMark](https://collomosse.com/pubs/Agarwal-WACV-2026.pdf), '
      '[constrained-filter author code](https://github.com/LuminolT/constrained-conv-TIFS2018). '
      'Fixed-filter source commit and SHA256 are in srm_source.json. Detailed changes in PLAN.md.']
    (here/'REPORT.md').write_text('\n'.join(lines)+'\n');(root/'measurements.json').write_text(json.dumps(measurements,indent=2)+'\n')
    print('completed measurements',len(measurements))
if __name__=='__main__':main()
