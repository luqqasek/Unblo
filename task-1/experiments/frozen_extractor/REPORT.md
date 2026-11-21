# Experiment B: frozen extractors + segmentation heads

Run on original train data only, seed 42, 136 fit / 34 validation images per method. No synthetic data, augmentation, test-set access or leaderboard submission.

| Method | mIoU @0.5 | Calibrated mIoU | Clean IoU | WM IoU | Median | Std | Threshold | Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pixelseal | 0.5312 | 0.5322 | 0.4307 | 0.6338 | 0.4911 | 0.1147 | 0.4000 | 17 |
| trustmark | 0.5049 | 0.5215 | 0.6180 | 0.4251 | 0.4866 | 0.0976 | 0.7000 | 4 |
| stegastamp | 0.4900 | 0.4927 | 0.4969 | 0.4885 | 0.4806 | 0.0734 | 0.4000 | 13 |

## Setup and limitations

Official wmloc loader and exact wmloc.utils score/miou_image. Fixed split matches baseline seed 42. Exact pixel hashes and 256-bit dHash found no cross-split duplicates/candidates (distance <=12). No source grouping metadata exists; perceptual checking cannot guarantee absence of shared-source leakage.
Frozen backbones are evaluated with gradients disabled. Independent heads: per-scale 1x1 projection to 32 channels + GroupNorm/ReLU, bilinear fusion at 32x32, concatenation, two 3x3 convolutions, 1x1 output, bilinear upsample logits to 128x128, sigmoid. Feature maps exceeding 32x32 are average pooled before caching. This limits fine boundary detail.
AdamW lr=0.001, weight decay=0.0001, batch=8, 50 epoch maximum, patience=8. Equal BCE and soft Dice loss. Checkpoint selected by official validation mIoU at threshold 0.5. Only after checkpoint selection, choose threshold among 0.3,0.4,0.5,0.6,0.7 separately per method. Report both fixed and calibrated scores; calibration and early stopping use this validation set, so results are selection estimates, not an independent final test.
CPU PyTorch 2.5.1, four threads, float32. Physical RTX 3070 8GB and GTX 1080 Ti 11GB are present but installed PyTorch has no CUDA support. No mixed precision. Cached features avoid repeated backbone inference. No pretrained weights changed; StegaStamp derived ONNX only adds graph outputs, no initializer changes.
Scope is Experiment B. RGB U-Net and optional localization-prior heads were not run. Thus superiority over RGB-only learning and benefit of priors remain untested. Existing native detector results are contextual references, not freshly trained alternatives.

## Per-method evidence

| Method | Native localization reference | RGB U-Net | Frozen extractor + head | Extractor + prior |
|---|---|---|---|---|
| PixelSeal | N/A: no native segmentation | Not run | 0.5322 | Not run |
| TrustMark | 0.3242; corrected fallback 0.4722 | Not run | 0.5215 | Not run |
| StegaStamp | 0.2669, community mirror | Not run | 0.4927 | Not run |

### pixelseal

Layers: ['stage0', 'stage1', 'stage2', 'stage3']. Raw shapes (B,C,H,W): [[1, 96, 64, 64], [1, 192, 32, 32], [1, 384, 16, 16], [1, 768, 8, 8]]. Cached shapes: [[170, 96, 32, 32], [170, 192, 32, 32], [170, 384, 16, 16], [170, 768, 8, 8]].
Trainable head parameters: 92705. Training/evaluation loop: 9.3s. Backbone extraction including loading and image IO: 0.102s/image; cached head inference: 0.0009s/image (CPU).
Raw message bit accuracy mean: 0.8222; Pearson correlation with per-image calibrated mIoU: 0.1642888400442684. Diagnostic only, n=34, not causal evidence.
Area breakdown: {'small_under_10pct': {'n': 6, 'miou': 0.48078271458030447}, 'medium': {'n': 14, 'miou': 0.5447403084862193}, 'large_over_90pct': {'n': 14, 'miou': 0.5417193723573036}}.

### trustmark

Layers: ['layer1', 'layer2', 'layer3', 'layer4']. Raw shapes (B,C,H,W): [[1, 256, 62, 62], [1, 512, 31, 31], [1, 1024, 16, 16], [1, 2048, 8, 8]]. Cached shapes: [[170, 256, 32, 32], [170, 512, 31, 31], [170, 1024, 16, 16], [170, 2048, 8, 8]].
Trainable head parameters: 169505. Training/evaluation loop: 6.3s. Backbone extraction including loading and image IO: 0.108s/image; cached head inference: 0.0017s/image (CPU).
Raw message bit accuracy mean: 0.7174; Pearson correlation with per-image calibrated mIoU: 0.407662691519744. Diagnostic only, n=34, not causal evidence.
Area breakdown: {'small_under_10pct': {'n': 16, 'miou': 0.4828746308513665}, 'medium': {'n': 10, 'miou': 0.5577085842980801}, 'large_over_90pct': {'n': 8, 'miou': 0.5536569818669249}}.

### stegastamp

Layers: ['gen_decoder/dec_conv_2/Relu:0', 'gen_decoder/dec_conv_4/Relu:0', 'gen_decoder/dec_conv_6/Relu:0', 'gen_decoder/dec_conv_7/Relu:0']. Raw shapes (B,C,H,W): [[1, 32, 200, 200], [1, 64, 100, 100], [1, 128, 25, 25], [1, 128, 13, 13]]. Cached shapes: [[170, 32, 32, 32], [170, 64, 32, 32], [170, 128, 25, 25], [170, 128, 13, 13]].
Trainable head parameters: 57889. Training/evaluation loop: 6.8s. Backbone extraction including loading and image IO: 0.135s/image; cached head inference: 0.0007s/image (CPU).
Raw message bit accuracy mean: 0.7621; Pearson correlation with per-image calibrated mIoU: 0.19169467358543887. Diagnostic only, n=34, not causal evidence.
Area breakdown: {'small_under_10pct': {'n': 13, 'miou': 0.4770094039506638}, 'medium': {'n': 8, 'miou': 0.48963774045092234}, 'large_over_90pct': {'n': 13, 'miou': 0.5103352281720742}}.

## Interpretation

Compare only the same 34 validation images: prior baseline PixelSeal 0.231726, TrustMark 0.272808, StegaStamp 0.283805. Native TrustMark top1 0.324179; corrected empty fallback 0.472191. Mirrored StegaStamp segmentation 0.266904 (official weight identity unverified). PixelSeal has no native segmentation output in its pixelwise=False checkpoint.
Low-resolution fusion can miss thin transplanted objects and holes; BCE/Dice and only 136 fit images may favor smooth spatial regions. Check the ten rank-spaced examples per method before interpreting a high aggregate score as detailed watermark recovery. Global bit accuracy measures decoding, not spatial localization.
Inspected examples: PixelSeal 148fa6a6293f (mIoU 0.2670) correctly covers the watch but extends predictions into clean background. TrustMark 8939e4777231 (0.3975) predicts the giraffe rather than the thin watermarked top strip; a558bea42351 (0.8038) follows a large screen region reasonably well. StegaStamp 0f715ead6d99 (0.3604) fills nearly the entire frame and misses clean holes, whereas 2e731007efaf (0.7288) recovers a lower-right clean hole approximately. This is evidence of some spatial signal but substantial boundary/semantic failure, not reliable pixel-exact localization.
PixelSeal has the highest aggregate score in this run; TrustMark improves over the corrected native bbox reference on this holdout, and StegaStamp improves over the mirrored segmenter. The improvements versus spatial priors do not isolate the benefit of watermark-specific pretraining: the missing RGB-only control is necessary for that claim.
Next: use an untouched or nested validation protocol, then compare the same heads against RGB-only and ablate the 32x32 pooling. Increasing spatial detail is specifically motivated by thin-region and hole errors. These results do not establish a need for synthetic data.

## Reproduce

Single command from repository root: `task-1/.venv/Scripts/python.exe task-1/experiments/frozen_extractor/run.py`. It trains all methods, generates this report and runs verify.py. report.py and verify.py can also be run separately on saved outputs.
The runner reuses feature caches and retrains heads deterministically from seed 42. Outputs: results/config.json, split_and_duplicates.json, *_features.pt, *_head.pt, *_metrics.json, *_validation.npz, 30 example panels and three distributions. Comparison CSV and provenance JSON are generated by report.py. All bulky results are ignored by Git.