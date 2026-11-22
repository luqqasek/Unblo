# PixelSeal-specific proposals: RTX 3070 screen

A short feasibility experiment with the original PixelSeal embedder/decoder unchanged. This is not a new full OOF estimate, not a full NRGMark reproduction, and not a leaderboard submission.

GPU: NVIDIA GeForce RTX 3070; FP32, 400 SGD updates, effective batch2 with microbatch1; seed42. Sources:128 original synthetic train and32 original dev, disjoint by source identity. Three exact local embeddings per source, with matched clean twins. Half same-image, half cross-image composites. No post-embedding JPEG/blur/rescaling. All detector parameters trainable; BN running statistics frozen.

The same pretrained NRGMark detector is evaluated before and after PixelSeal adaptation. Evaluation uses original fold1 (57real images), fixed endpoints and thresholds, with no real-label fitting. The segmentation baseline uses saved control-adapt maps, avoiding the unresolved historical checkpoint-replay discrepancy. This is the predecessor development baseline, not new validation of the current #2675 model.

## Synthetic source-disjoint proposal evaluation

| Stage | Bucket | n | Proposal recall IoU≥0.5 | Mean best box IoU | Clean-twin positive rate |
|---|---|---:|---:|---:|---:|
| before | 0.1–1% | 32 | 0.00000 | 0.00052 | 0.00000 |
| before | 1–5% | 32 | 0.00000 | 0.01884 | 0.00000 |
| before | 5–15% | 32 | 0.21875 | 0.22428 | 0.00000 |
| after | 0.1–1% | 32 | 0.00000 | 0.00008 | 0.00000 |
| after | 1–5% | 32 | 0.03125 | 0.03409 | 0.00000 |
| after | 5–15% | 32 | 0.18750 | 0.15522 | 0.12500 |

Proposal recall considers the top4 detector outputs at score≥0.05 regardless of class; clean-twin positive rate uses class1 score≥0.5. These measure different operating points. Source images and exact content/paste geometry are shared between clean/marked twins, but train and dev source identities are disjoint.

## Real-image segmentation

Official per-image two-class mIoU; all other FG columns are foreground IoU. Clean holes is the overlapping full_watermark cohort; its clean-class IoU is also shown.

| Stage | Variant | mIoU | FG≤1% | FG1–5% | FG5–20% | FG>20% | FG holes | Clean holes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| before | baseline | 0.55748 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.89921 | 0.23200 |
| before | proposal_union | 0.55748 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.89921 | 0.23200 |
| before | verified_union | 0.55748 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.89921 | 0.23200 |
| before | wrong_verified_union | 0.55748 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.89921 | 0.23200 |
| after | baseline | 0.55748 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.89921 | 0.23200 |
| after | proposal_union | 0.55748 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.89921 | 0.23200 |
| after | verified_union | 0.55748 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.89921 | 0.23200 |
| after | wrong_verified_union | 0.55748 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.89921 | 0.23200 |

## Proposal/verification diagnostics

- before, ≤1%: 0/7 GT bounding boxes matched at IoU≥0.5. GT used only for scoring, never proposing/verifying.
- before, 1–5%: 0/9 GT bounding boxes matched at IoU≥0.5. GT used only for scoring, never proposing/verifying.
- before, 5–20%: 0/11 GT bounding boxes matched at IoU≥0.5. GT used only for scoring, never proposing/verifying.
- before, >20%: 2/30 GT bounding boxes matched at IoU≥0.5. GT used only for scoring, never proposing/verifying.
- before, accepted correct message: 0/57 images at integer≥231/256 correct bits.
- before, accepted random message: 0/57 images at integer≥231/256 correct bits.
- before, proposal_union: delta mIoU +0.000000; improved 0, worsened 0.
- before, verified_union: delta mIoU +0.000000; improved 0, worsened 0.
- after, ≤1%: 0/7 GT bounding boxes matched at IoU≥0.5. GT used only for scoring, never proposing/verifying.
- after, 1–5%: 0/9 GT bounding boxes matched at IoU≥0.5. GT used only for scoring, never proposing/verifying.
- after, 5–20%: 0/11 GT bounding boxes matched at IoU≥0.5. GT used only for scoring, never proposing/verifying.
- after, >20%: 0/30 GT bounding boxes matched at IoU≥0.5. GT used only for scoring, never proposing/verifying.
- after, accepted correct message: 0/57 images at integer≥231/256 correct bits.
- after, accepted random message: 0/57 images at integer≥231/256 correct bits.
- after, proposal_union: delta mIoU +0.000000; improved 0, worsened 0.
- after, verified_union: delta mIoU +0.000000; improved 0, worsened 0.

## Exact-crop decoder ceiling on synthetic dev

| Bucket | Marked bit accuracy | Marked passing ≥231/256 | Clean bit accuracy | Clean passing gate |
|---|---:|---:|---:|---:|
| 0.1–1% | 0.50903 | 0/32 | 0.50635 | 0/32 |
| 1–5% | 0.63904 | 1/32 | 0.50183 | 0/32 |
| 5–15% | 0.85303 | 18/32 | 0.49695 | 0/32 |

These crops use oracle embedding geometry on synthetic dev only; they are not a test-time detector. Failure here means the declared verification gate can reject a true region even with exact geometry. It does not prove absence of a local watermark-presence signal.

## Training-only memorization sanity

An additional 300 updates at LR 0.005 on four training composites plus their exact clean twins. No real/dev evaluation or checkpoint promotion. Marked examples localized at box IoU≥0.5: 4/4. Clean twins with marked-class score≥0.5: 4/4.

The detector can fit box locations in this tiny training set, but still predicts watermark presence on clean twins. This is consistent with content/paste shortcuts and poor watermark discrimination; it does not identify the causal feature or establish a generalization result.

## Measured outcome

After adaptation: verified union delta +0.000000; direct proposal union delta +0.000000, both relative to the same saved baseline. No model was promoted and no submission made.

## Scope and next decision

Only local rectangular embeddings were used for detector training. Irregular remnants and clean holes remain in real validation, but this run does not directly train those regimes. Original pixel crops are verified by the unchanged official decoder; the verifier can reject genuine undecodable fragments. At most4 proposals/image and7 refinements/proposal; no adaptive scale or threshold search. A single random-message query/image is a limited negative control, not a formal false-positive guarantee.

A negative short-run result cannot establish an information-theoretic limit or rule out a detector trained longer, with more data, at a different detector scale or with different small-object anchors. Likewise, an improvement on this repeatedly inspected development fold is not independent confirmation. No checkpoint or probability threshold was selected from this fold.

Machine-readable artifacts: protocol.json, training.json, before/after_proposals.json, before/after_metrics.json, completed.json, trained.pt and before/after_maps.npz. The saved maps are binary masks, not probability maps; no probability precision claim is made. Existing models and submissions are preserved.
