# Native RGB localizer: bounded quality pilot

**Real held-out StegaStamp mIoU: 0.69496** on 170 images, all three original fold partitions, threshold0.5.
Conditional image bootstrap95%: [0.67340, 0.71641].

The real validation images were not used to train or select the checkpoint or threshold.
The final pilot update and raw weights were declared before reading real validation scores.

| Evaluation | mIoU |
|---|---:|
| Real held-out, 170 images | 0.69496 |
| Synthetic dev, 64 unseen sources/messages | 0.71418 |
| Synthetic dev, partial masks only | 0.71461 |
| Real: all-clean constant | 0.25397 |
| Real: all-marked constant | 0.24603 |
| Real: spatially shuffled model maps | 0.44716 |
| Real: best constant per image, GT ORACLE diagnostic | 0.46011 |

FG IoU 0.68834; clean IoU 0.70158; macro FG AP 0.83594.
Training: 200 updates x batch64 = 12800 views; 2,500 distinct COCO sources.
ConvNeXt-v1 Base ImageNet-22K + learned local stem + progressive decoder; all stages trainable.
93,777,762 parameters, native512 input, native128 logits. One node, four A10040GB, FP32.

## Limits

- one family, reconstructed existing fold partitions.
- repeatedly used real development data.
- no real labels used for model training/checkpoint/threshold selection.
- simple synthetic hard composites only; full edit mixture not implemented in this pilot.
- bootstrap conditions on this fitted model and these 170 images; edited source siblings cannot be excluded.
- not a calibrated estimate of overall three-family hidden competition score.

No test predictions, leaderboard submission, full nested CV or other-family training occurred in this pilot.
The old models and outputs were preserved. This result must not be compared causally with historical
multi-fold results as though budgets, splits and generators were matched.

## Artifacts

FP32 probability/GT maps and per-image CSV reproduce the official score exactly. Training/dev curves,
source/message manifests, numerical-fidelity checks, resource profiles and failure galleries accompany this report.

## Execution and audit

The main GPU pilot took **1,087.46 seconds (18.12 minutes)** from its process start,
including numerical checks, batch profiles, generation, training and first-fold
evaluation. Code/environment setup and downloads happened earlier. Folds2/3 were
confirmed in a separate short inference allocation, reusing fold1 maps.

Training pool:2,500 encoded realizations from2,500 distinct source IDs and byte
hashes. The200 updates actually sampled2,435 distinct sources,9,600 primary
views and3,200 companion views. Exactly6,400 of12,800 views belonged to pairs.
Development pool:128 separate sources/messages; the fixed synthetic score uses
64 primary examples, with companion controls reported separately.

[Learning curves](results/final/learning_curves.png) ·
[Machine-readable summary](results/final/summary.json) ·
[Source/message/view audit](results/final/data_verification.json) ·
[Frozen actual configuration](pilot_config_frozen.json).

Reproduction from task-1 after copying the small artifacts locally:

```bash
PYTHONPATH=. .venv/bin/python -m experiments.native_localizer.report_pilot --artifacts outputs/native_localizer/artifacts --out experiments/native_localizer/results/final
```

Resumable raw/EMA/optimizer/RNG checkpoint on JURECA:
`/p/project1/training2666/pszenny1/native-localizer/runs/stegastamp/pilot-30min/last.pt`.
The selected evaluation uses its raw model weights, not EMA. No hidden-test
predictions or leaderboard submission were made.
