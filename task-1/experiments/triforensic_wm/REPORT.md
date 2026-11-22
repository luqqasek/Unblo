# TriForensic-WM measured screening

Single fixed fold1:113 training and57 validation images per family. Reused development validation; not independent test performance. Public best remains #2675=0.6440.

One unified optimizer/OneCycle loop, co-batched full images and matched patches. No staged freezing or separate patch pretraining. See [protocol](PROTOCOL.md).

| Family / update | mIoU | Delta vs original baseline | FG ≤1% | FG1–5% | Clean ≤1% | Clean1–5% | Clean on WM>50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| pixelseal / baseline | 0.55748 | +0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.08307 | 0.23200 |
| pixelseal / 0 | 0.55560 | -0.00188 | 0.00000 | 0.00000 | 0.00000 | 0.08293 | 0.23126 |
| pixelseal / 100 | 0.55795 | +0.00047 | 0.00000 | 0.00213 | 0.00000 | 0.08922 | 0.23861 |
| pixelseal / 200 | 0.56577 | +0.00828 | 0.00000 | 0.00000 | 0.00000 | 0.10178 | 0.25252 |
| pixelseal / 1000 | 0.57869 | +0.02120 | 0.00000 | 0.00000 | 0.00000 | 0.12234 | 0.26886 |
| pixelseal / 1561 | 0.58164 | +0.02415 | 0.00000 | 0.00000 | 0.00000 | 0.10579 | 0.26750 |

pixelseal / unified-v1 gate1000: `{
  "real_tiny_auc": 0.7353274765124639,
  "synthetic_interior_auc": 0.8181818181818181,
  "passed": true
}`.

pixelseal / unified-v1: 1561 optimizer updates; status `user_authorized_early_stop`; measured runtime 27.65 minutes; 224784 counted views.

[pixelseal learning curves](results/pixelseal/unified-v1/learning-curves.png).

Compared with the newer #2675 recipe on the SAME57 images: 0.570883 → 0.581635 (delta+0.010752). This is validation, not a new public score.

Fusion audit at1000: the final refiner adds only about0.000042 mIoU over its own pre-refiner base. On tiny watermark GT cells, base logits average−5.300 and the additive correction is only+0.009. Local AUC improvement has NOT become successful small-mask segmentation. The global branch also receives learned detached local skips, so no isolated causal claim about backbone versus local features is justified.
| trustmark / baseline | 0.64662 | +0.00000 | 0.00000 | 0.52530 | 0.00000 | 0.10383 | 0.24857 |
| trustmark / 0 | 0.64448 | -0.00214 | 0.00000 | 0.51613 | 0.00000 | 0.10132 | 0.24738 |
| trustmark / 100 | 0.64543 | -0.00118 | 0.00000 | 0.52913 | 0.00000 | 0.09513 | 0.24133 |

trustmark / unified-v1: **training in progress**; the rows above are measured interim checkpoints, not a completed experiment.

## Interpretation and limits

Initialization/update0 is not a trained result. FP32 original baseline and the mixed-precision initialization are reported separately. The final mask metric and local-only diagnostic are distinct; a high local AUC alone does not establish successful segmentation. Fixed threshold0.5; no validation-driven threshold search.

Read per-image JSON for complete misses,AP/AUC and rectangular/irregular breakdowns. Full/empty synthetic controls and low-observability false vetoes are separate diagnostics. Synthetic interior AUC excludes cells whose RF29 touches a seam or patch padding; strips narrower than29 cannot contribute interior cells.

No new submission is produced by this screening report. Large checkpoints,full probability maps and exact uint8 pools remain on JURECA under native-localizer/triforensic-{runs,data}.

The first training launch failed before optimization due to a missing synchronized verifier module. The dependency was copied,imports checked,and training restarted. No failed-launch result is presented as a trained score.

## Early stopping and production selection

PixelSeal was stopped at saved update1561 following the user's instruction to avoid prolonged training without targeted gains. mIoU0.581635 improves over the previous matched screen0.570883, but all16 watermark cases≤5% and all4 clean-hole cases≤1% remain complete misses. Clean1–5% IoU0.105786 is above the original baseline0.083069 but below update1000's0.122340. This does not solve tiny localization.

A four-GPU attempt to disable carrier activation recomputation was slower despite higher VRAM usage. It was cancelled; discarded updates are archived separately and are not part of the retained checkpoint. Subsequent runs keep recomputation enabled.

The selected PixelSeal recipe is queued for an all170-image refit with synthetic replay, the same3500-step OneCycle schedule stopped at1561, and fresh new branches. This is final fitting, not a second training phase of the screening model. Test prediction and one authorized upload follow artifact validation; no new public score is claimed here.
