# Cached carrier fallback screen

Two fixed rules, no fitting and no threshold sweep, declared before measuring
this screen. Use the strong #3188 PixelSeal global/carrier checkpoint and its
cached local logits on the existing57 held-out images. Only completely empty
base masks are eligible; every other prediction remains unchanged.

Control: retain #3188. Candidate1: choose the164 highest local-carrier-logit
cells (the fixed1% size used by the earlier fallback, but different evidence).
Candidate2: use the predefined500,494 native rectangle grid from pixelseal_regions;
select the rectangle maximizing (mean local logit inside minus mean in a3-cell
surrounding ring)*sqrt(area), then emit that rectangle. This ranking score is
not a calibrated z-score or independent-pixel significance test. No GT-derived
box or area is an inference input. No resizing of raw images.

Report per-image delta, bootstrap CI, all minority buckets and complete misses.
A candidate needs mean gain>=0.003, positive paired-bootstrap lower bound, and
no>0.01 clean-IoU regression on WM>50% before test inference. Choose the higher
mIoU only among passing candidates. This is a reused single-fold exploratory
screen, not full OOF or an unbiased test estimate. The synthetic-only/presence
classifier from pixelseal_regions is not used; only the already trained carrier.
