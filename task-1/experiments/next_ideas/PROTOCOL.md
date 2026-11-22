# Next ideas: measured opportunities and StegaStamp adaptation

Baseline accepted submission: #2675, public mIoU0.6440. Older results remain intact.
This run is separately authorized after the user cancelled TriForensic.

## Corrections to the expert audit

- Shape is a proxy, not a verified edit type. Train metadata already distinguishes
  full-canvas watermark followed by restoration (85/family) from other examples.
- Re-embedding and gradient times high-pass were previously measured in
  message_localizer/signal_check.py; their real macro AUC was near chance.
  Repeating these unchanged is lower priority than a new StegaStamp transfer test.
- Two-class per-image losses and a frozen official TrustMark verification decoder
  are already present. C2/C5 are not wholly untested ideas.
- Every train mask may contain both classes without proving the same for hidden
  test. A forced minority prediction must be validated; tiny-area arithmetic alone
  cannot establish the location, hit probability, or expected gain.
- An oracle crop that fails does not prove no decoder can recover any crop.
  Outside-mask blur changes context and creates boundaries; it cannot uniquely
  identify the original embedding/edit process.
- Independent fair-coin binomial tails do not calibrate adaptive crop search with
  correlated decoder bits. Keep TrustMark's verified90/100 rule unchanged.
- A public-score difference must be evaluated through paired differences; the
  standard error of one score cannot establish the significance of a paired gain.
- No test self-training or held-out mask bank is used. Synthetic replay retains
  exact cached uint8 pixels; no post-embedding image augmentation is introduced.

## A1 and A5

Use label-disjoint saved maps, exact majority targets and official per-image
two-class IoU. Error contributions are weighted by n/510 and sum to1 minus the
reported510-image proxy score. PS has historical #2409 full OOF only; #2675 PS
continuation has a57-image screen, so this proxy is not called current full OOF.
TM includes the accepted frozen crop corrections; SS uses its synthetic-only model.
Two fixed fallback rules replace only empty/full predictions: top164 probability
cells or the maximum-mean13x13 box, with complementary scores for clean holes.
Neither reads GT for region placement. Report regressions as well as gains.

## A4: short, three-fold native StegaStamp adaptation

Three independent workers, one per original outer fold, share a single4-A100 node.
Each starts from the same label-independent synthetic-only raw checkpoint.
All model weights are trainable from update0. Exactly150 updates, raw weights and
threshold0.5 are declared before reading results; update50 is progress only.
Batch16=12synthetic+4real, microbatch16, FP32 and TF32off. Cached original2500
COCO encodings, original same-content generator, no additional embedding calls.
AdamW backbone3e-6/new1e-5, decay0.01 with bias/norm exclusions, gradient clip1.
Keep existing per-image BCE+dualDice and local auxiliary0.04. No new architecture.
Validate the initial maps against archived probabilities (max absolute error<1e-4)
before any optimizer step. Save optimizer/RNG state and exact sample exposure.
Predeclared success milestone: final pooled170-image OOF gain>=0.01, with paired
image bootstrap and minority regressions reported. Reused folds imply development
selection bias. Do not silently select update50 or a different threshold afterward.

The fourth GPU runs an oracle ladder with each unchanged official decoder. Crop
pixels are resized directly by the official wrapper once: PS256,TM245,SS400.
GT is used only for offline views, never proposals or production inference.
Views: full image, exact positive bbox, outside-GT box blur17, outside-GT mean,
and inside-GT blur17. Score correct and other real messages; report empirical
null resolution rather than extrapolating tiny false-positive probabilities.

## Data leakage and reproduction

Before launch a63-bit grayscale DCT pHash screen searched all510 train images
for pairs within Hamming distance4; no candidates were found. This is a limited
screen, not proof that no edited siblings exist. Existing folds are preserved.
No test pixels, labels, masks or predictions influence training or selection.
All GPU tasks run through Slurm on one node. The job releases the node when workers
finish; no fixed wall-time or unmeasured mIoU improvement is promised.

## Bounded continuation declared after the150-update result

The first endpoint improved170-image OOF from0.694962 to0.722019, with all
three folds improving and the curve still rising from update50. Before any
450-update result or new public receipt, declare exactly300 additional updates
to total450. Restore each150 checkpoint's model, AdamW and RNG state, and keep
the same deterministic data sequence, losses, LRs and effective batch16.
Three single-GPU fold workers run beside a fourth all170 continuation on the
same node. The all170 worker is not used for validation or model selection.

The450 candidate inherits the already selected150 postprocessing unchanged.
Select450 only if paired OOF mIoU improves by at least0.005 over150, the paired
bootstrap lower bound is positive, and neither FG IoU on<=5% images nor clean
IoU on>50%-watermarked images drops by more than0.01. Otherwise retain150.
This is one bounded development extension, not unbiased additional validation.
No public score enters this decision. Keep every150 checkpoint and prediction.

Initial all170 refit uses4-GPU DataParallel; its450 continuation uses1 GPU
beside the three folds. Restore the first CUDA RNG stream when reducing device
count; the model has no dropout or stochastic depth. Floating-point reduction
order can differ, while optimizer state, data exposure and batch size remain
the same. Save this distinction rather than claiming bitwise equivalence.

Production creates150 predictions first. Only a selected450 candidate produces
additional test predictions. Each artifact must pass binary/shape/name/size
validation. Upload at most once per10 minutes, only after local selection.

```sh
PYTHONPATH=.:src python -m experiments.next_ideas.audit
# Within a Slurm allocation with4 GPUs:
python -m experiments.next_ideas.job
```
