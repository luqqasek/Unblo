# Small-region fixes for the actual watermark models

Checked 2026-11-22 against primary papers, official repositories and author comments. This is a literature/compatibility review, not a new experiment. No model, generator, training run or submission was changed.

## What the directly relevant benchmark establishes

[What Breaks Local Watermarks? (2026), Appendix A.5 and §4.4](https://arxiv.org/html/2609.16832v1) evaluates the released PixelSeal and TrustMark-Q weights without fine-tuning. PixelSeal is globally embedded, then its residual is masked; decoding still uses the whole image. It therefore diagnoses weak remnant decoding, not a trained PixelSeal localization fix. TrustMark-Q uses local rectangular embedding and predicted-region crop decoding, with 100 raw bits and ECC disabled. Small-region payload recovery is poor, and payload recovery is distinguished from localization.

Its mask distribution is materially different from ours: COCO/SA-1B masks cover at least 2%; procedural masks target 4–14%. It does not establish a solution for ≤1% regions. [MarkBench's released repository](https://github.com/garandorinc/markbench) provides the evaluation harness, interfaces, transformations and toy adapter; its README explicitly excludes watermark model implementations, weights and datasets. It is useful for methodology, not a downloadable improved PixelSeal decoder.

## PixelSeal: documented crop weakness, no verified compatible ready-made repair

The [Pixel Seal paper, Figure 8](https://arxiv.org/html/2512.16874v1) reports bit accuracy 0.58 at crop parameter 0.32 and 0.81 at 0.45. In the [official Crop implementation](https://github.com/facebookresearch/videoseal/blob/main/videoseal/augmentation/geometric.py), the size parameter scales both height and width: 0.32 corresponds to approximately 10% retained area, not 32%. These are payload-decoding experiments on crops, not pixelwise localization scores, and should not be compared numerically to our mIoU.

In [official issue #81](https://github.com/facebookresearch/videoseal/issues/81), author Pierre Fernandez explains the Y-only design as an imperceptibility/video-codec choice and suggests RGB's greater capacity can improve crop robustness; ChunkySeal is RGB. That changes the embedded signal. Substituting ChunkySeal cannot repair existing PixelSeal images automatically. It is explanatory evidence, not a compatible checkpoint upgrade.

No inspected primary source demonstrates a ready-made detector that solves ≤1% irregular regions in existing PixelSeal-256 images with the original encoder unchanged. This is a bounded search result, not a claim that no such work exists.

## TrustMark/NRGMark: an actual compatible approach, with a scale mismatch

[NRGMark](https://collomosse.com/pubs/Agarwal-WACV-2026.pdf) learns a region localizer while keeping watermark encoder/decoder frozen. Its per-watermark detector is trained on 100K MIRFlickr sources; the reported region range is 150–1350 pixels. This supports training a localizer for a fixed watermark and reading the detected crop with the original decoder. The study does not establish success on our tiny 512-canvas fragments, and reported positive-prediction localization IoU is not our per-image foreground/background mIoU.

Inference for this project: use this as support for a PixelSeal-specific proposal detector or a smaller-region TrustMark detector trained with the unchanged official embedder. Do not treat a TrustMark-trained proposal model's failure on PixelSeal as evidence that a PixelSeal-trained detector cannot work. Crop verification is most natural for a self-contained small embedding; a remnant of a large embedding may lack a decodable full message even when it contains watermark evidence.

## MaskWM-D does not mean “only train the decoder”

[Mask Image Watermarking, §3.1–3.2](https://arxiv.org/html/2504.12739v3) distinguishes MaskWM-D (mask introduced at decoding) from MaskWM-ED (mask also supplied at embedding). Crucially, D is still trained end-to-end: the paper explicitly says the extraction objective also guides the encoder to distribute its watermark. ED further conditions the encoder on the mask and concentrates signal in selected regions.

The transferable idea is localization-guided extraction, including suppressing irrelevant clean background before decoding. The published success cannot be attributed solely to a decoder attached to fixed PixelSeal/TrustMark. A fixed-encoder version would be a new experiment. [Official code and weights](https://github.com/hurunyi/MaskWM) are available for their own D/ED watermark families, including distortion-specific fine-tuning configurations. They are not compatible drop-in decoders for our existing marks.

## WAM and SyncSeal compatibility

[WAM's official training commands](https://github.com/facebookresearch/watermark-anything) jointly specify/train its own embedder and dense extractor, with released 32-bit models. Its local presence/per-pixel bit supervision is relevant as a training design, but the weights do not decode PixelSeal-256 or TrustMark-100 merely because all are invisible watermarks.

[SyncSeal, §1 and §3](https://arxiv.org/html/2509.15208v1) embeds an additional synchronization watermark after the primary watermark and uses it to estimate/invert geometry. It can upgrade a watermarking pipeline when that extra signal is embedded before editing. Our pre-existing images do not contain that signal. Its “works with existing watermarks” claim therefore does not mean its released extractor can retrospectively synchronize arbitrary competition images.

## A genuine decoder-only route: adapters

[Adaptable Watermarking, §3.2 and §4.2](https://arxiv.org/html/2611.18363v1) freezes the target embedder and trains decoder adapters with bitwise BCE. Experiments include TrustMark Q/P, WAM, InvisMark, RoSteALS and HiDDeN, but not PixelSeal. CWLoRA-16 is competitive with full fine-tuning in its cross-method payload-decoding task. This establishes feasibility of learning a decoder for an unchanged embedding process; it does not demonstrate tiny-region segmentation or cure localization-induced forgetting.

A compatible hypothesis is to train a small adapter/decoder branch on original exact embeddings while retaining message supervision, then evaluate localization separately. The multi-domain residual alignment loss belongs to encoder adaptation; it is not evidence that our auxiliary residual-prediction head should work. Paper training uses far longer schedules than our short probes; parameter efficiency does not imply proportionally faster training. No official implementation link was identified in the inspected paper; do not promise ready-to-load adapters.

## Practical interpretation

The clearest existing-model recipe is a watermark-specific proposal localizer with the original encoder/decoder fixed. The strongest decoder-only precedent is adapter training with explicit message loss. MaskWM/WAM offer architectural or training ideas, but their reported performance benefits from co-training the embedding signal. SyncSeal and RGB/ChunkySeal alter that signal and cannot be applied retrospectively as fixes. For irregular remnants, presence localization must be evaluated separately from successful full-payload recovery; otherwise real but undecodable watermark fragments will be rejected by construction.
