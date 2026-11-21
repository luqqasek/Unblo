# Frozen extractors: 3-fold CV

510 training images; each image evaluated exactly once outside training. Seed 42.
Independent heads refitted from scratch on 113/114 images per method; validation 57/56.
Backbones frozen; existing label-independent feature caches reused.
Epoch (max 50, patience 8) and threshold selected on an inner 80/20 split, then head refitted on all outer training images.

| Method | Fold 1 | Fold 2 | Fold 3 | Mean | SD (sample) | OOF |
|---|---:|---:|---:|---:|---:|---:|
| pixelseal | 0.519131 | 0.477273 | 0.527204 | 0.507869 | 0.026803 | 0.507756 |
| trustmark | 0.512931 | 0.480991 | 0.493457 | 0.495793 | 0.016098 | 0.495807 |
| stegastamp | 0.465554 | 0.341844 | 0.424390 | 0.410596 | 0.062998 | 0.410515 |
| Overall | 0.499205 | 0.433369 | 0.481684 | 0.471420 | 0.034097 | 0.471359 |

Metric: official per-image binary mIoU, averaged across images. OOF weights all images equally; fold mean weights folds equally.
SD describes variation across three data splits, not random-seed stability or a confidence interval.
No exact duplicate images found. Source grouping metadata unavailable; shared-source leakage cannot be excluded.
Architecture was previously explored on this dataset; this is not a new untouched test set.
No test data or leaderboard used. Fixed-threshold scores, inner histories, checkpoints, split names and OOF probabilities are in results/cv3/.

Reproduce from repo root: `task-1/.venv/Scripts/python.exe task-1/experiments/frozen_extractor/cross_validate.py`.
