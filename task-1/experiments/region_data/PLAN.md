# Fixed generator comparison — 2026-11-22

Baseline is message_localizer, not the obsolete failed native RGB PS/TM model.
Both arms start from the same 400-step synthetic-only checkpoint, seed42.
200 additional updates ×32 views; AdamW encoder5e-6/new5e-5, wd.01, clip1,
FP32, frozen BN statistics, unchanged segmentation+0.1 paired ranking.
Control continues the old irregular same-content generator. Treatment uses50%
old,25%same-content COCO object supports,25%cross-image transplant/addition;
new branch minority bins35/35/20/10% over .25–1/1–5/5–20/20–50%, equal polarity,
25% object-boundary remnants. These changes form one data recipe, not isolated
causal tests. Half the batch views participate in genuine pairs.

Prepare128train and32dev extra exact embeddings/family, on unchanged source
partitions, square R side128/192/256/384. Originals and bitstrings are reused
within their partition; count these as additional realizations, not new covers.
No watermark-strength change, no post-embedding resize/flip/JPEG/blur.
COCO polygons are transformed by the original pre-embedding source crop;
marked RGB pixels undergo no interpolation. Clean-object addition and marked
transplantation have content-matched clean twins with identical visual edits.

Fixed evaluation: old64-example dev; new64-example dev (same seed/suite for
both models); all170 competition training images per family, threshold0.5,
synthetic-only. No held-out labels guide generator masks, thresholds or schedules.
Describe real geometry exploratorily, do not fit a real mask shape bank.
Compare per-image mIoU, FG IoU <=1%,1–5%,>5%, clean-hole IoU and false-positive
area. Record effective view counts after pairing; decoded bit accuracy is a
diagnostic and never a gate deleting tiny surviving examples.

If useful, use a predeclared150-step real adaptation with75%synthetic replay,
same3folds/LR/threshold as previous OOF. Keep a matched control under the same
additional synthetic exposure. Do not submit to leaderboard in this task.

## Follow-up after first synthetic-only comparison

The object recipe improved PS overall (0.55500→0.56919) but neither method
localized <=1% foreground. TM overall0.58415→0.58291; FG1–5%0.06114→0.07059.
This motivates a bounded DATA-only counterfactual pairing screen, not a new
architecture/loss or threshold sweep.

Both follow-up arms start from objects600,100updates×32views, same AdamW/LR/loss,
seed73 (fresh RNG stream, matched across arms). Control keeps the object recipe.
Treatment makes half the views equal-area position pairs on the same exact
clean/marked full-image base: identical photo and message; two disjoint masks
with identical official128-grid area at different locations. COCO mask shapes
are translated on a4px grid, but RGB pixels are NOT translated/resampled. Half
of pairs have small foreground, half small clean holes. Pair minority sampling:
70% .25–1%,30%1–5%. Remaining unpaired views retain the previous object recipe.
This removes the global clean-vs-marked class-balance shortcut in paired views.
Fixed threshold0.5 on real170/family; no real labels in training. This is an
adaptive development follow-up, not an untouched confirmatory benchmark.
