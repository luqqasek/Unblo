# PixelSeal: fixed local-carrier fallback

Starting system: accepted submission #3188, public mIoU0.6736. This screen uses the existing EMA1561 local branch, not the rejected new region classifier. No model training, threshold sweep or GT-dependent region placement.

[Predeclared protocol](PROTOCOL.md). One reused57-image development fold, not full OOF. Only9 completely empty base predictions are eligible. The inherited global prediction remains unchanged everywhere else.

| Candidate | mIoU | Delta | Paired95% CI | Improved / worse | Selected |
|---|---:|---:|---|---:|---|
| local_top164 | 0.585511 | +0.003786 | [0.00025098005828435693, 0.009617547076214709] | 7 / 2 | True |
| local_ring_rectangle | 0.579939 | -0.001786 | [-0.004279913038587942, 0.0007935962440388908] | 2 / 7 | False |

The selected rule emits the164 highest local-carrier-logit cells when the base mask is empty. The dense rectangle rule instead maximizes interior-versus-ring contrast times square-root area; it regressed and was rejected. Neither changes raw image pixels.

The cached selected result reduces FG complete misses on ≤5% masks from16 to8. Mean FG IoU remains low:0.05452 on≤1% and0.01382 on1–5%. Most of the global gain comes from one image; this is a modest exploratory improvement, not a solution to tiny localization.

| Group | n | Baseline | Selected | FG misses before / after |
|---|---:|---:|---:|---:|
| fg_le1 | 7 | 0.000000 | 0.054517 | 7 / 3 |
| fg_1to5 | 9 | 0.000000 | 0.013820 | 9 / 5 |
| fg_5to20 | 11 | 0.392297 | 0.392297 | 0 / 0 |
| fg_gt20 | 30 | 0.881543 | 0.881543 | 0 / 0 |

Clean IoU on WM>50% stays0.267867; no clean-hole case is modified. Image-level bootstrap resamples57 image differences; it does not remove bias from repeatedly inspecting this fold or selecting between two candidates.

## Actual deployment verification

Same checkpoint hash, independently instantiated carrier/JND, same native pixels and BF16 recipe. Maximum local-logit difference from archived inference: 0.0234375. Some tied top164 rankings differ, so exact mask reproduction failed. We evaluated the actual deployment output before any test prediction, without changing the rule or acceptance gate.

Actual deployed validation mIoU **0.585528**, delta **+0.003804**, paired95% CI[0.00023208655436043681, 0.00966555714214029]; passed gate: True.

## Test predictions

All340 PixelSeal masks prepared; 46 formerly empty predictions change. Every changed mask has164 positive cells. TrustMark and StegaStamp stay bit-identical to #3188. No test GT, resampling, TTA or new payload handling.

## Server receipt

HTTP200; actual response: `{"submission_id": 3457, "status": "success", "score": 0.6743, "previous_best": 0.6736, "delta": 0.0007, "improved": true, "message": "New best on this task. Score shown is on the public split; the final ranking uses the held out subset, revealed after submissions close."}`. This is the public split only; no hidden final score or per-family public score is available.
