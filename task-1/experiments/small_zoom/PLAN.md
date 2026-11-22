# Small-region follow-up, 2026-11-22

Preserve #2163 and existing checkpoints. No test images or leaderboard selection.
First inspect the completed RTX3070 diagnosis, then repeat probability diagnostics
on the latest control-adapt PS and objects-adapt TM OOF. Oracle thresholds remain
diagnostics, never deployment selection.

Bounded first inference test: original Adobe NRGMark TrustMark-Q bounding-box
detector, frozen and independent of competition labels. Source commit
4c60b9fdb4125fca3001f0803e6890a6b55371ac; checkpoint SHA256
7e1da5667a4da070573d71bbf5a03d586f8836397bbcdf2c2044865ec37df0a5.
Keep trustmark==0.8.0 competition embedder/decoder unchanged. Isolate detector
source in a separate package. Detect proposals on ALL 170 images per family,
without GT, metadata edit type, or mask area. PixelSeal use is explicitly a
cross-family transfer diagnostic, not a matched pretrained localizer.

For each proposal, decode the original crop and a fixed local size/position
search with the original official decoder, full FP32. Select using known-message
bit accuracy / signed logits, not GT. Test fixed >=0.90 bit accuracy acceptance,
crop area <=10% and union with existing OOF mask; never choose a branch by GT size.
Record proposals before loading GT for evaluation; compare correct and random
message controls. Report all-image mIoU and FG IoU <=1%,1-5%,5-20%,>20%, plus clean
holes. Proposal recall and GT bbox compatibility are diagnostics only.

No assumptions that BCE gradient is intrinsically 100x smaller per positive pixel,
that AUC implies a particular IoU, or that crop decoding identifies edit history.
TruFor uses RGB/Noiseprint++ fusion, not the proposed literal SRM+Bayar recipe.
Prior object-data experiment already separated embedding rectangle and remnant;
do not repeat its geometry audit as if still missing.

## Follow-ups declared after native NRGMark failure

NRGMark Q yielded zero candidate recall at IoU .5 for <=5% at native512.
Second inference screen: add1536 proposal input, supported by the original paper's
1500/2048 image regime. Original pixels remain the crop decoder input. No training.

TrustMark bounded crop scan: top16 spatially separated maxima of smoothed OOF
probabilities plus a fixed4x4 center grid; sizes32:16:144, aspect<=2. Refine top8
correct-message and top8 random-message candidates by the fixed crop-size/offset
grid. Fixed acceptance bit accuracy>=.90, union with original OOF threshold.5.
All170 images, no GT/metadata area filtering. This is an adaptive development
experiment, not independent final performance. Preserve null-search maxima.

PixelSeal matched screen: initialize the same synthetic600-update control model,
train200 additional updates x32 views, LR5e-6/5e-5, seed941, FP32. Three arms:
legacy data;50%legacy+50%tiny-crop views; same data plus residual auxiliary.
Tiny views are75%marked/25%clean twins; effective exact tiny positives37.5% of
batch, split equally <=1% and1-5% in expectation.512 source-disjoint existing
train covers and64 existing dev covers, one new exact crop embedding each,
aspect1/4..4, area log-uniform .25-1% or1-5%. No post-embedding resampling.
Residual head predicts16 signed Y subpixels per128-grid cell, pixel-shuffled to512;
do not average away the weak signal. Balanced L1 within/outside known rectangle,
weight ramp0..0.1 in50 steps, zero-initialized head. Auxiliary metadata only used
on new exact pairs. Segmentator unchanged, no residual input at inference.
If training succeeds, fixed150 updates x(24synthetic+8real) on each of original
3folds with same real exposure RNG12342, LR2e-6/2e-5, fixed threshold.5.
