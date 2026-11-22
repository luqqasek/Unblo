# PCMV: one fixed-fold experiment and stronger PixelSeal deployment

Current work uses the strongest verified deployable PixelSeal baseline. TrustMark and StegaStamp remain exactly as in #3054. One PixelSeal scorer screen is planned; no full cross-validation or backbone search. GPUs0–1 only; GPUs2–3 reserved for the user’s other experiment.

## Stronger checkpoint verification

EMA1561 global/carrier branch, no refiner or Noiseprint inference inputs. Same57-image development fold: 0.570883 → **0.581724**, delta +0.010841, paired95% CI [0.0008853393920901176, 0.022832824834269724]. Selected for test: True. This checkpoint trained113 real images plus synthetic replay; no all170 refit. Reused fold and checkpoint selection create optimism; this is not a fresh held-out estimate.

The first short job completed with the earlier minority guard and produced no test predictions. Per the user’s priority to use the strongest overall model, the subsequent decision retained this checkpoint despite FG1–5% IoU falling0.012724→0. Global validation and public scores improved; the tiny-watermark problem remains unresolved. The original decision/evaluation artifacts are preserved.

| Bucket | n | Metric | Previous | Candidate |
|---|---:|---|---:|---:|
| fg_le1 | 7 | fg_iou | 0.000000 | 0.000000 |
| fg_1to5 | 9 | fg_iou | 0.012724 | 0.000000 |
| fg_5to20 | 11 | fg_iou | 0.348457 | 0.392297 |
| fg_gt20 | 30 | fg_iou | 0.881765 | 0.881543 |
| clean_le1 | 4 | clean_iou | 0.000000 | 0.000000 |
| clean_1to5 | 10 | clean_iou | 0.075527 | 0.106361 |
| wm_over50 | 27 | clean_iou | 0.239856 | 0.267867 |

## Leaderboard response

Actual HTTP status: 200; response: `{"submission_id": 3188, "status": "success", "score": 0.6736, "previous_best": 0.6677, "delta": 0.0059, "improved": true, "message": "New best on this task. Score shown is on the public split; the final ranking uses the held out subset, revealed after submissions close."}`.

PCMV fitting/evaluation has not completed. Source implementation and planned budget are not measured results.

## Scope and safeguards

See PROTOCOL.md for the frozen screening budget and SOURCES.md for pinned official implementation references. SAM2 resizing is restricted to geometry proposals. Official watermark decoder probes retain original512 coordinates, exact uint8 replacement bytes and the original decoder preprocessing. Counterfactual boundaries may introduce distribution shifts; scores are learned features, not calibrated Gaussian significance. No GT enters proposal generation or feature extraction. Prior submissions and checkpoints remain preserved.
