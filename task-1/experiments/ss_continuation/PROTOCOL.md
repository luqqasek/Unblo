# Bounded StegaStamp continuation from the best accepted system

Declared before new training. Best public submission is #3188 (0.6736); its
StegaStamp branch is the all170 real/replay450 checkpoint from #3054.
Earlier matched OOF improvement150→450 was +0.02037. The prior1000 continuation
was cancelled while pending and produced no training updates.

Use only two A100 workers on one node: fold1 (57 validation images) and matching
all170 refit. Restore exact450 model, optimizer and RNG, then train550 further
updates to1000. Existing FP32 model/loss/LRs, batch16 with12 synthetic+4 real,
no image augmentation, no changed embedders or strengths. Verify parent fold
probabilities against archived450 maps before the first update.

Update750 is progress only. At1000 evaluate two predeclared candidates: raw1000
probabilities and the equal mean of450 and1000 probabilities. Both use the
inherited0.5 threshold and fixed top1% fallback on empty/full masks. Select the
highest mIoU candidate passing all three conditions: paired mean gain>=0.005,
image-bootstrap95% lower bound>0, and no>0.01 reduction in either combined
FG<=5% IoU or clean IoU on WM>50%. Otherwise retain450 and do not infer/submit.
Report every candidate and the reused-fold selection limitation, not full OOF.

The all170 model supplies no validation evidence. If selected, predict all340
StegaStamp test images, replace only that family in #3188 and validate the exact
1020-name binary uint8 artifact below20MB. No GT routing, TTA, threshold search
or public-score-based hyperparameter selection. Preserve previous checkpoints.

Composition update2026-11-22T06:24Z, before1000-step selection: the independent
fixed PixelSeal fallback passed its unchanged validation gate and submission
#3457 was accepted at0.6743. If StegaStamp passes its predeclared gate, preserve
PixelSeal and TrustMark from#3457 when assembling. This changes neither
StegaStamp training, validation, candidate selection nor threshold. The model
recipe retains#3188 as its historical parent; assembly records#3457 explicitly.
