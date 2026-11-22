# One recommended next method: independent region-based mask correction

> Historical snapshot: this report describes the named earlier submission. The latest recorded best is #3054 (public mIoU 0.6677); see the [documentation index](../DOCUMENTATION_INDEX.md).

Research checked 2026-11-22. This is a recommendation, not an implemented or measured new model. No training or submission was performed. The best accepted submission is #2675, public mIoU 0.6440. Its PixelSeal selection evidence is one reused 57-image fold, not new full OOF; TrustMark and StegaStamp have different development protocols. See [submission report](../local_presence/SUBMISSION_REPORT.md).

## Decision

The strongest next experiment is **a watermark-specific region detector with its own mask prediction and calibrated acceptance**, operating as a correction to the existing mask. It should learn both small watermarked regions and small clean regions. A practical implementation is a Mask R-CNN-style proposal/ROI head over the existing high-resolution local carrier and contextual features. It predicts an independent regional mask and whether applying that mask is beneficial; it does not merely add a small scalar correction to an already saturated global logit.

This selection is an engineering inference from our failures and compatible primary research. There is no inspected paper or checkpoint proving it will solve our tiny PixelSeal marks. It is not a claim that the previously trained TriForensic carrier is already sufficient for precise localization.

## Why this rather than another generic local branch

The newer saved TriForensic endpoint is more informative than its stale short REPORT.md:

- [completed.json](../triforensic_wm/results/pixelseal/unified-v1/completed.json): at update1561, final PixelSeal mIoU 0.581635 on57 images. Foreground IoU is still0 for all7 tiny images and all9 images in1–5%; clean IoU is0 on all4 tiny-clean images.
- In the same artifact, the independent local map has tiny macro AUC0.81204 and AP0.08504, but foreground IoU only0.05466. Its global mIoU is0.30855. Thus there is useful ranking information, but using that map directly creates too many false positives.
- [fusion-audit-1000.json](../triforensic_wm/results/pixelseal/unified-v1/fusion-audit-1000.json) shows near-constant refiner corrections around+0.009 against tiny-region base logits around−5.3. The refiner is not making consequential decisions. These numbers describe update1000, not a decomposition of1561.
- [model.py](../triforensic_wm/model.py) and [losses.py](../triforensic_wm/losses.py) already contain SRM/Bayar, strict local features, Noiseprint, ring context, paired patches, focal/IoU losses, full joint training and low-observability controls. Recommending these again as absent would be incorrect.
- Earlier [local_presence experiments](../local_presence/REPORT.md) show that simply increasing local corrections can regress the global result. Small spatial refiners and real-patch adaptation did not reliably solve the targeted buckets.

The observed weak correction is not proof that its amplitude is the sole cause. Because the local map has low precision, multiplying that correction or lowering the threshold is not justified. The proposed regional head must learn coherent support and a decision to abstain, not just produce a larger delta.

## Primary evidence and its limits

[NRGMark, sections3–4](https://collomosse.com/pubs/Agarwal-WACV-2026.pdf) demonstrates a separate region localizer for existing watermark families with the original encoder and decoder frozen. It trains a distinct localizer per watermark method, using region classification and box regression. This supports a watermark-specific proposal model, rather than transferring a TrustMark detector to PixelSeal. However, the study trains on100K MIRFlickr images with much larger regions, reports localization IoU only for predicted watermarked detections, and does not test PixelSeal or our≤1% regime. Our small-region and clean-hole extension is unproven.

[Mask R-CNN](https://arxiv.org/abs/1703.06870) provides the architectural precedent for predicting an instance mask in parallel with regional classification and box regression. A mask head is needed here because a bounding rectangle would wrongly fill clean pixels around irregular remnants or remove watermarked pixels around irregular clean holes. Its COCO results are not watermark results.

[Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241) shows that instance-class confidence and actual mask quality need not agree, and learns mask IoU from region features and the predicted mask. This supports adding an explicit quality estimate. Predicting whether a correction improves the competition's whole-image mIoU is our proposed extension, not a claim made by that paper.

## Concrete proposed experiment

1. Start with PixelSeal, using the same113/57 split as #2675. Preserve the accepted all-data checkpoint. Extract native-pixel features from the existing local carrier and contextual backbone. Do not resize uint8 images to manufacture extra signal; ROI alignment resamples learned features only.
2. Train a region proposal head and mask head. Proposals come from image features, never validation GT. Include small/thin boxes and border-touching regions at multiple scales. Predict region classes `watermarked`, `clean`, and `background/no correction`. Both polarities receive explicit instance-level supervision; positive regions are sampled as regions rather than being diluted among tens of thousands of background pixels.
3. Use exact current synthetic pools and outer-training real masks. Derive connected regions/boxes from training labels only. Keep directly embedded tiny rectangles separate from surviving fragments and clean overwrites in diagnostics. Include matched clean twins and intact low-JND patches as hard negatives. Do not require full payload recovery for PixelSeal: existing GT-bbox diagnostics do not support that gate.
4. Inside each proposed region, predict an independent binary mask. For rectangles, box and mask losses encourage spatial coherence; for irregular cases, the mask head preserves shape. Outside accepted correction support, retain the old prediction. If proposals conflict, a learned quality/benefit score must select one or abstain; no unconditional union/subtraction.
5. Train quality on the actual change in official per-image mIoU caused by applying each candidate to a baseline prediction. Baseline predictions for real training examples should be generated out of fold within the113 training images, or another explicit non-leaking calibration design used. Calibrate acceptance using only inner training splits and source-disjoint synthetic dev. The57 outer validation images must not set the threshold or train the quality head.

This is one region-based correction method, not an ensemble of unrelated new watermark models. The novelty relative to the existing code is object-level proposal supervision, per-region independent segmentation, and explicit acceptance based on mask quality/benefit. Merely adding another dense residual head would not implement it.

## What would count as success

The first screen must compare against #2675's same-fold PixelSeal0.570883 and the newer TriForensic0.581635, clearly labeling the latter as an unsubmitted development candidate. Report all57 images, paired differences and bootstrap intervals, foreground/clean IoU and complete misses in≤1%,1–5%,5–20%, proposal recall, accepted false corrections and errors on non-target images. Quality gains require improvements on multiple independent images, not only higher local AUC or an oracle choice of proposals.

A useful predeclared continuation gate is at least+0.01 global PixelSeal mIoU versus the better same-fold candidate, recovery of at least2 of7 tiny cases at foreground IoU≥0.25, and no mean degradation on the remaining>5%-foreground images. These are proposed engineering criteria, not statistically proven thresholds. If proposal recall is poor, the representation/proposal stage remains the issue; if recall is high but accepted masks are poor, diagnose regional segmentation or calibration separately. Time to a first measured screen should be benchmarked; no30-minute training claim is established.

## Alternatives rejected for this next step

- More SRM/Noiseprint/local patch branches repeat already completed work.
- Unconditionally scaling the existing correction or unioning local detections risks the false positives already seen in local-only predictions.
- Full-message decoding as a mandatory PixelSeal gate would reject most smallest valid masks, even with oracle crops.
- Generic WAM/MaskWM/SyncSeal pretrained weights are not compatible fixes for existing PixelSeal images: their successful pipelines train or add a different embedded signal. The [existing compatibility review](../../analysis/model_specific_small_regions_research.md) records these boundaries.
- Decoder-only adapter training is compatible with a fixed embedder in principle ([Adaptable Watermarking](https://arxiv.org/html/2611.18363v1)), but its evidence concerns payload decoding across watermark models, not tiny-region segmentation or clean-hole masks. It is a weaker direct match to the measured failure than region-level detection and mask correction.
