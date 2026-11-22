# TriForensic-WM measured screening

Single fixed fold1:113 training and57 validation images per family. Reused development validation; not independent test performance. Public best remains #2675=0.6440.

One unified optimizer/OneCycle loop, co-batched full images and matched patches. No staged freezing or separate patch pretraining. See [protocol](PROTOCOL.md).

| Family / update | mIoU | Delta vs original baseline | FG ≤1% | FG1–5% | Clean ≤1% | Clean1–5% | Clean on WM>50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| pixelseal / baseline | 0.55748 | +0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.08307 | 0.23200 |
| pixelseal / 0 | 0.55560 | -0.00188 | 0.00000 | 0.00000 | 0.00000 | 0.08293 | 0.23126 |
| pixelseal / 100 | 0.55795 | +0.00047 | 0.00000 | 0.00213 | 0.00000 | 0.08922 | 0.23861 |
| pixelseal / 200 | 0.56577 | +0.00828 | 0.00000 | 0.00000 | 0.00000 | 0.10178 | 0.25252 |

[pixelseal learning curves](results/pixelseal/unified-v1/learning-curves.png).

## Interpretation and limits

Initialization/update0 is not a trained result. FP32 original baseline and the mixed-precision initialization are reported separately. The final mask metric and local-only diagnostic are distinct; a high local AUC alone does not establish successful segmentation. Fixed threshold0.5; no validation-driven threshold search.

Read per-image JSON for complete misses,AP/AUC and rectangular/irregular breakdowns. Full/empty synthetic controls and low-observability false vetoes are separate diagnostics. Synthetic interior AUC excludes cells whose RF29 touches a seam or patch padding; strips narrower than29 cannot contribute interior cells.

No new submission is produced by this screening report. Large checkpoints,full probability maps and exact uint8 pools remain on JURECA under native-localizer/triforensic-{runs,data}.

The first training launch failed before optimization due to a missing synchronized verifier module. The dependency was copied,imports checked,and training restarted. No failed-launch result is presented as a trained score.
