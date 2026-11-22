# Shared IML-ViT clean-hole adaptation

[Frozen protocol](PROTOCOL.md), [official source and checkpoint provenance](../next_ideas/VERIFIED_FORENSICS_RESEARCH.md). Fixed original fold1 across three families; repeatedly used development data. No test labels, boxes or per-image GT-selected polarity at inference. No new submission is implied by this report.

## Integration and numerical checks

Official checkpoint strict-load passed: True; 91,777,729 parameters. Raw normalization and first image projection FP32, deeper BF16. Probability difference versus FP32 on the probe image: mean0.000537, max0.052187. These are not identical predictions. Native512 versus official padded1024 mean probability difference0.003292; pixels are never resized. Cold probe latencies include initialization and are not comparable throughput benchmarks.

Single-GPU versus4-GPU replica check: `{
  "FP32_single_vs_four_replicas_max_abs_logit": 0.0,
  "tolerance": 0.001,
  "upstream_list_alias_fixed": true
}`.

The first attempt failed before any optimizer update because upstream FPN stores ordinary Python-list aliases to registered layers; DataParallel used GPU0 weights in another replica. The wrapper now resolves registered simfp modules on each replica, preserving operations and checkpoint keys. A second startup reached validation0 but stopped on JSON logging of a NumPy integer; the counter now uses a Python int. Both failures occurred before training. Prior oracle JSON row availability was GT-filtered; it is not used for routing. Full-canvas agreement is newly computed for all171 images. Accepted historical TM rectangles were audited:28 total, max area9.613%; an explicit15% assertion protects integration.

## Measured validation

| Update | Mean mIoU | Baseline | Delta | Paired95% CI | Clean IoU at WM>50% | Routed images |
|---|---:|---:|---:|---|---:|---:|
| 0 | 0.584325 | 0.659924 | -0.075599 | [-0.09728776312811056, -0.0556031846548524] | 0.078796 | 78 |
| 50 | 0.630190 | 0.659924 | -0.029734 | [-0.03966556252001428, -0.020661127380212703] | 0.243291 | 78 |
| 150 | 0.641382 | 0.659924 | -0.018542 | [-0.026185182410859764, -0.011328192254718065] | 0.286019 | 78 |

## Latest per-family and minority metrics

| Family | Group/metric | n | Baseline | Current | Complete misses before/current |
|---|---|---:|---:|---:|---|
| pixelseal | all/miou | 57 | 0.570883 | 0.560074 | — |
| pixelseal | fg_le1/fg_iou | 7 | 0.000000 | 0.000000 | 7/7 |
| pixelseal | fg_1to5/fg_iou | 9 | 0.012724 | 0.012724 | 8/8 |
| pixelseal | fg_5to20/fg_iou | 11 | 0.348457 | 0.348457 | 3/3 |
| pixelseal | fg_gt20/fg_iou | 30 | 0.881765 | 0.887937 | 0/0 |
| pixelseal | clean_le1/clean_iou | 4 | 0.000000 | 0.000000 | 4/4 |
| pixelseal | clean_1to5/clean_iou | 10 | 0.075527 | 0.025765 | 6/7 |
| pixelseal | wm_over50/clean_iou | 27 | 0.239856 | 0.187357 | 11/12 |
| trustmark | all/miou | 57 | 0.646616 | 0.616107 | — |
| trustmark | fg_le1/fg_iou | 3 | 0.000000 | 0.000000 | 3/3 |
| trustmark | fg_1to5/fg_iou | 18 | 0.525302 | 0.525302 | 5/5 |
| trustmark | fg_5to20/fg_iou | 6 | 0.277846 | 0.277846 | 1/1 |
| trustmark | fg_gt20/fg_iou | 30 | 0.884871 | 0.878557 | 0/0 |
| trustmark | clean_le1/clean_iou | 4 | 0.000000 | 0.000000 | 4/4 |
| trustmark | clean_1to5/clean_iou | 5 | 0.103828 | 0.030588 | 2/4 |
| trustmark | wm_over50/clean_iou | 26 | 0.248572 | 0.122089 | 6/11 |
| stegastamp | all/miou | 57 | 0.762273 | 0.747964 | — |
| stegastamp | fg_le1/fg_iou | 4 | 0.280969 | 0.280969 | 1/1 |
| stegastamp | fg_1to5/fg_iou | 6 | 0.451241 | 0.451241 | 1/1 |
| stegastamp | fg_5to20/fg_iou | 10 | 0.636646 | 0.636646 | 0/0 |
| stegastamp | fg_gt20/fg_iou | 37 | 0.943656 | 0.943666 | 0/0 |
| stegastamp | clean_le1/clean_iou | 3 | 0.298541 | 0.197803 | 0/0 |
| stegastamp | clean_1to5/clean_iou | 12 | 0.444761 | 0.398553 | 0/0 |
| stegastamp | wm_over50/clean_iou | 32 | 0.553446 | 0.502458 | 0/0 |

Routed non-restoration images (offline diagnostic): 0. Macro per-image AP/AUC, defined counts and full per-image changes are retained in `real-150.json`. The fixed gate uses original message agreement plus predicted baseline area; GT is used only for scoring/diagnostics. All unrouted masks are preserved exactly, including SS empty/full fallback and TM verified small rectangles.

## Decision and cost

Actual completion: `{"step": 150, "stop": "predeclared_no_gain_gate", "seconds": 114.6113254899974, "candidate": false, "minority_guard": false, "final_miou": 0.6413815361530222, "delta": {"mean": -0.01854247806526327, "ci95": [-0.026185182410859764, -0.011328192254718065]}, "test_inference": false}`. No test inference was performed by this screen. A positive flag is development evidence, not a promised leaderboard score.

Training uses170 real restoration images from fold-training only and384 hard-composited synthetic views from existing full-image encodings. Effective batch8:4 real +4 synthetic. All91.8M parameters are trainable; one AdamW, encoder LR1e-5, decoder LR3e-5, existing per-image halfBCE/halfdualDice. BN running stats fixed; no dropout, geometric/photo augmentation or new embedder calls. Source/weight hashes and runtime versions: `config.json`. Same-content and cross-cover hole edits preserve uint8 values outside the declared edit mask.
