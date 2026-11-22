# PCMV bounded screen, frozen before GPU evaluation

One PixelSeal fixed-fold screen. The production reference is submission3054
(public0.6677). First evaluate the EMA1561 TriForensic global/carrier branch
without refiner/Noiseprint inputs. Use it if development mIoU gains>0.003;
otherwise keep PS data-small150. Report all minority regressions. This follows
the user's explicit priority for the strongest overall checkpoint over perfect
historical comparability. Test images never select the rule.

## One node, two GPUs

Job15905215 completed the first base audit in85seconds, using4GPUs, before
the two-GPU restriction reached its running process. It generated no test
predictions because the earlier minority guard rejected the checkpoint.
Job15905266 reserves one4-A10040GB node; our processes are restricted to physical
devices0/1. Devices2/3 are reserved for the user's separate experiment in this
same allocation. Runtime30minutes. Cached-base test inference runs first, then replaces its process with
the two-worker PCMV feature stage, avoiding another queue wait. All image caches
are resumable. No additional node is submitted concurrently.

## One fit and final evaluation

-90 real images for scorer training,23 for action-threshold calibration,57 for
 final evaluation. Base/carrier weights saw only the original113 training
 images, so they did not train on the57 evaluation images. Their predictions
 on scorer train/calibration images are in-sample: report that limitation.
-Up to260 exact existing synthetic partial views, balanced across five pool
 buckets. Reserve64 clean source identities (including donor identities)
 before sampling. First32 sources supply64 clean/full-marked training controls;
 next32 supply64 calibration controls. This is a fast screening budget, not
 the proposed2,000–3,000-image full expansion.
-200 clean/full-marked realization pairs from128 separate dev source photos
 form the final anti-shortcut audit. Messages vary before original embedding.
 These are400 views, not400 unique photos. No weights or thresholds fit on them.
-SAM2.1-small automatic object masks, individually prompted local-contrast
 extrema, base connected components, local-contrast components and boxes;
 maximum48 masks per image. GT-free deterministic proposals. One automatic
 sampling grid plus prompted proposals replaces the proposed larger multi-grid
 bank for this bounded first test. No NRGMark is used in this PS-only screen.
-FP32 original PixelSeal decoder; isolation, ablation, ring-isolation and tight
 masked-crop probes, full-image pre-pooling masked/JND-weighted readouts;
 clean per-bit bias estimated from32 training covers,64 wrong-message nulls.
 Known-message alignment is a feature, not a formal Gaussian p-value.
-JND-quartile inside/ring carrier contrasts, sample counts, geometry, base map
 statistics and per-image ranks. No raw RGB/SAM semantic embeddings in scorer.
-Two multioutput regressors: ControlB sees unstratified base/local/geometry;
 treatment also sees counterfactual/JND evidence. Each is fitted once using
220 fixed HistGradientBoosting iterations,15 leaves, min40 samples, LR.06,
 L2=1, seed42, no early stopping. Images weighted equally per source view;
 real training views receive3x synthetic weight.
-Tune only action threshold on23 calibration images plus64 calibration controls.
 Fixed grid0,.002,.005,.01,.02,.03,.05,.08,.12,1. Up to2 non-overlapping
 add/carve toggles. Threshold1 gives no-op eligibility. Require<2% false
 polarity toggles and mean positive mIoU loss<.003 on calibration controls.
-One final57-image evaluation and400-control audit; report both class IoUs,
 minority buckets/misses, per-image changes, paired bootstrap intervals,
 wrong-message conditional swap and bounded greedy oracle. Reused development
 folds and base choice imply selection optimism; do not call this unbiased OOF.

The screen does not train a new pixel backbone, compute full3-fold CV, or use
validation GT for proposals. It does not claim that every recommendation in
the original broad PCMV proposal is implemented. Expand only after evidence.

## Reproduction

Cluster, task-1 cwd, PYTHONPATH=.:src:

```sh
.venv-native/bin/python -m experiments.pcmv.infer_cached --root /p/project1/training2666/pszenny1/native-localizer
```

This reuses the completed base check, predicts test, then runs the screen. Use
--infer-only for predictions alone. A completed base is reused. Completed feature shards resume; a completed
scorer experiment refuses silent retraining. Checkpoint, data and model-source
dependencies remain in the archived experiments; do not regenerate embeddings
with different versions/strengths.
