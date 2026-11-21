"""Summarize measured Experiment B results and checkpoint provenance."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'


def main():
    rows=[]; reports=[]
    for method in ['pixelseal','trustmark','stegastamp']:
        r=json.loads((OUT/f'{method}_metrics.json').read_text()); reports.append(r)
        s=r['selected']; z=np.load(OUT/f'{method}_validation.npz')
        assert len(r['val_names'])==34 and z['probability'].shape==(34,128,128)
        assert np.isfinite(z['probability']).all()
        target=z['target']; area=target.mean((1,2)); vals=np.asarray(s['per_image'])
        r['area_diagnostics']={label:dict(n=int(mask.sum()),miou=float(vals[mask].mean()) if mask.any() else None)
                               for label,mask in [('small_under_10pct',area<.1),('medium', (area>=.1)&(area<=.9)),('large_over_90pct',area>.9)]}
        rows.append(dict(method=method,miou=s['miou'],miou_at_half=next(x['miou'] for x in r['threshold_scores'] if x['threshold']==.5),
                         iou_clean=s['iou_clean'],iou_watermark=s['iou_watermark'],median=s['median'],std=s['std'],
                         threshold=s['threshold'],epoch=r['epoch'],bit_accuracy=r['mean_bit_accuracy'],bit_miou_pearson=r['bit_miou_pearson']))
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paths=[ROOT/'checkpoints/pixelseal/checkpoint.pth',ROOT/'.venv/Lib/site-packages/trustmark/models/decoder_Q.ckpt',ROOT/'checkpoints/stegastamp/stega_stamp_logits.onnx']
    provenance={str(p.relative_to(ROOT)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (OUT/'provenance.json').write_text(json.dumps(dict(weights_sha256=provenance,torch=torch.__version__,cuda=torch.cuda.is_available()),indent=2))
    lines=['# Experiment B: frozen extractors + segmentation heads','',
           'Run on original train data only, seed 42, 136 fit / 34 validation images per method. No synthetic data, augmentation, test-set access or leaderboard submission.',
           '', '| Method | mIoU @0.5 | Calibrated mIoU | Clean IoU | WM IoU | Median | Std | Threshold | Epoch |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+ ' | '.join(str(row[k]) if k in ['method','epoch'] else f'{row[k]:.4f}' for k in ['method','miou_at_half','miou','iou_clean','iou_watermark','median','std','threshold','epoch'])+' |')
    lines += ['', '## Setup and limitations', '',
              'Official wmloc loader and exact wmloc.utils score/miou_image. Fixed split matches baseline seed 42. Exact pixel hashes and 256-bit dHash found no cross-split duplicates/candidates (distance <=12). No source grouping metadata exists; perceptual checking cannot guarantee absence of shared-source leakage.',
              'Frozen backbones are evaluated with gradients disabled. Independent heads: per-scale 1x1 projection to 32 channels + GroupNorm/ReLU, bilinear fusion at 32x32, concatenation, two 3x3 convolutions, 1x1 output, bilinear upsample logits to 128x128, sigmoid. Feature maps exceeding 32x32 are average pooled before caching. This limits fine boundary detail.',
              'AdamW lr=0.001, weight decay=0.0001, batch=8, 50 epoch maximum, patience=8. Equal BCE and soft Dice loss. Checkpoint selected by official validation mIoU at threshold 0.5. Only after checkpoint selection, choose threshold among 0.3,0.4,0.5,0.6,0.7 separately per method. Report both fixed and calibrated scores; calibration and early stopping use this validation set, so results are selection estimates, not an independent final test.',
              'CPU PyTorch 2.5.1, four threads, float32. Physical RTX 3070 8GB and GTX 1080 Ti 11GB are present but installed PyTorch has no CUDA support. No mixed precision. Cached features avoid repeated backbone inference. No pretrained weights changed; StegaStamp derived ONNX only adds graph outputs, no initializer changes.',
              'Scope is Experiment B. RGB U-Net and optional localization-prior heads were not run. Thus superiority over RGB-only learning and benefit of priors remain untested. Existing native detector results are contextual references, not freshly trained alternatives.', '', '## Per-method evidence','']
    lines += ['| Method | Native localization reference | RGB U-Net | Frozen extractor + head | Extractor + prior |',
              '|---|---|---|---|---|',
              f"| PixelSeal | N/A: no native segmentation | Not run | {rows[0]['miou']:.4f} | Not run |",
              f"| TrustMark | 0.3242; corrected fallback 0.4722 | Not run | {rows[1]['miou']:.4f} | Not run |",
              f"| StegaStamp | 0.2669, community mirror | Not run | {rows[2]['miou']:.4f} | Not run |", '']
    for r in reports:
        lines += [f"### {r['method']}", '',f"Layers: {r['layers']}. Raw shapes (B,C,H,W): {r['raw_shapes']}. Cached shapes: {r['cache_shapes']}.",
                  f"Trainable head parameters: {r['parameters']}. Training/evaluation loop: {r['training_seconds']:.1f}s. Backbone extraction including loading and image IO: {r['feature_seconds_per_image']:.3f}s/image; cached head inference: {r['head_seconds_per_image']:.4f}s/image (CPU).",
                  f"Raw message bit accuracy mean: {r['mean_bit_accuracy']:.4f}; Pearson correlation with per-image calibrated mIoU: {r['bit_miou_pearson']}. Diagnostic only, n=34, not causal evidence.",
                  f"Area breakdown: {r['area_diagnostics']}.", '']
    lines += ['## Interpretation','',
              'Compare only the same 34 validation images: prior baseline PixelSeal 0.231726, TrustMark 0.272808, StegaStamp 0.283805. Native TrustMark top1 0.324179; corrected empty fallback 0.472191. Mirrored StegaStamp segmentation 0.266904 (official weight identity unverified). PixelSeal has no native segmentation output in its pixelwise=False checkpoint.',
              'Low-resolution fusion can miss thin transplanted objects and holes; BCE/Dice and only 136 fit images may favor smooth spatial regions. Check the ten rank-spaced examples per method before interpreting a high aggregate score as detailed watermark recovery. Global bit accuracy measures decoding, not spatial localization.',
              'Inspected examples: PixelSeal 148fa6a6293f (mIoU 0.2670) correctly covers the watch but extends predictions into clean background. TrustMark 8939e4777231 (0.3975) predicts the giraffe rather than the thin watermarked top strip; a558bea42351 (0.8038) follows a large screen region reasonably well. StegaStamp 0f715ead6d99 (0.3604) fills nearly the entire frame and misses clean holes, whereas 2e731007efaf (0.7288) recovers a lower-right clean hole approximately. This is evidence of some spatial signal but substantial boundary/semantic failure, not reliable pixel-exact localization.',
              'PixelSeal has the highest aggregate score in this run; TrustMark improves over the corrected native bbox reference on this holdout, and StegaStamp improves over the mirrored segmenter. The improvements versus spatial priors do not isolate the benefit of watermark-specific pretraining: the missing RGB-only control is necessary for that claim.',
              'Next: use an untouched or nested validation protocol, then compare the same heads against RGB-only and ablate the 32x32 pooling. Increasing spatial detail is specifically motivated by thin-region and hole errors. These results do not establish a need for synthetic data.',
              '', '## Reproduce','',
              'Single command from repository root: `task-1/.venv/Scripts/python.exe task-1/experiments/frozen_extractor/run.py`. It trains all methods, generates this report and runs verify.py. report.py and verify.py can also be run separately on saved outputs.',
              'The runner reuses feature caches and retrains heads deterministically from seed 42. Outputs: results/config.json, split_and_duplicates.json, *_features.pt, *_head.pt, *_metrics.json, *_validation.npz, 30 example panels and three distributions. Comparison CSV and provenance JSON are generated by report.py. All bulky results are ignored by Git.']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
