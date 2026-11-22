# PixelSeal-specific NRGMark adaptation on RTX 3070

Frozen before observing outcomes, 2026-11-22. Question: can a short adaptation of the released TrustMark proposal detector to exact PixelSeal local embeddings produce usable small-region proposals?

- Official unchanged PixelSeal-256 embedder and decoder, FP32/default strength/uint8. No post-embedding JPEG, blur or scaling.
- 128 existing original synthetic train sources and 32 existing dev sources; no locked or competition test sources. Original clean PNG hashes checked.
- Three independent local embeddings per source, area fractions 0.1–1%, 1–5%, 5–15%, aspect ratio <=16, minimum side8px. Embed rectangle and final mask recorded separately, equal in this first experiment.
- Half same-image, half cross-image compositing; each positive has an exact clean twin with identical content and paste geometry. Does not yet cover irregular remnants or clean holes in training; those remain in real evaluation.
- Released pinned NRGMark FasterRCNN initialization, all parameters fine-tuned except BN running statistics. Native512 input; no upscaling. SGD lr0.001/momentum0.9/wd0.0001, 400 fixed updates, effective batch2 (clean+marked), microbatch1, seed42. No real-label fitting.
- Compare before/after on source-disjoint synthetic dev and original real fold1 (57 images). Original saved control-adapt maps, not recomputed checkpoint inference, supply base masks. This is a predecessor development baseline, not newly estimated OOF of public best #2675.
- Proposal recall at bbox IoU0.5 and clean-twin positives; global per-image two-class mIoU, foreground IoU in <=1%,1–5%,5–20%,>20%,full_watermark clean-hole cohort, plus clean IoU.
- Separate raw proposal union from original-decoder verified union. At most4 proposals, candidate detector score>=0.05; bounded7-position/size refinements, minimum side8 and accepted area<=10%. Raw union requires class1 score>=0.5. Verified union uses integer >=231/256 bit agreement; fixed random-message control. No GT used to propose or accept boxes.
- Final endpoint only; no checkpoint/threshold selection from outer fold. No public submission. This is a small feasibility screen, not reproduction of 100K-source NRGMark training and not a definitive test of all proposal architectures.
