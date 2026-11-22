# PCMV: one fixed-fold experiment and stronger PixelSeal deployment

The stronger PixelSeal checkpoint was deployed as submission #3188 / public 0.6736. TrustMark and StegaStamp remain exactly as in #3054. The single PixelSeal PCMV screen completed with no validation improvement; no PCMV submission was made. No extended training was launched. No full cross-validation or backbone search. GPUs 0–1 only in the second job; GPUs 2–3 reserved for the user’s other experiment.

## Stronger checkpoint verification

EMA1561 global/carrier branch, no refiner or Noiseprint inference inputs. Same 57-image development fold: 0.570883 → **0.581724**, delta +0.010841, paired 95% CI [0.0008853393920901176, 0.022832824834269724]. Selected for test: True. This checkpoint trained 113 real images plus synthetic replay; no all 170 refit. Reused fold and checkpoint selection create optimism; this is not a fresh held-out estimate.

The first short job completed with the earlier minority guard and produced no test predictions. Per the user’s priority to use the strongest overall model, the subsequent decision retained this checkpoint despite FG 1–5% IoU falling 0.012724→0. Global validation and public scores improved; the tiny-watermark problem remains unresolved. The original decision/evaluation artifacts are preserved.

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

## Single scorer fit / final evaluation

Single repeatedly used PixelSeal57-image developmental fold;90 real scorer training,23 inner calibration;113-real-trained base used for all features (in-sample meta-training limitation). One fit and final evaluation; not full OOF. At most one toggle to avoid nonadditive two-action damage.

| Variant | mIoU | Delta vs strongest base | Paired95% CI |
|---|---:|---:|---|
| Base | 0.581724 | — | — |
| control_b | 0.581724 | +0.000000 | [0.0, 0.0] |
| pcmv | 0.581724 | +0.000000 | [0.0, 0.0] |

Exact one/two-disjoint-toggle oracle (GT-only diagnostic): 0.662397 / 0.673509. Deployment allows at most one toggle. Candidate for further inference: False. Full buckets, miss counts, actions and wrong-message swap are saved in evaluation.json. The wrong-message test holds the existing base/proposals fixed; it tests only new message features.

## Why there was no gain

Both scorers selected threshold 1.0 and applied zero toggles on the 57 validation images and 400 final control views. For ControlB the admissible calibration thresholds were 0.02 and above; for PCMV they were 0.08 and above. All admissible settings tied at calibration mIoU 0.737850, and the predeclared conservative tie break selected no-op. Lower thresholds violated the calibration safety limits; their validation performance was not used to retune the rule.

Zero false toggles and unchanged wrong-message scores are therefore trivial consequences of no-op, not evidence that the verifier can localize a watermark. The result rejects this trained scorer and recipe; it does not prove that all region-level evidence is unusable.

There are two distinct limitations. The proposal bank itself has limited tiny-region coverage (none of 7 tiny foreground images has a proposal reaching minority IoU 0.5). Separately, the scorer did not find an action policy that improved inner calibration while satisfying clean/full-image controls. Calibration images were seen by the base model: their 0.737850 baseline is much higher than 0.581724 on the held-out fold. This in-sample feature/calibration gap is a limitation, not an independently established causal explanation.

## Final minority metrics

| Group | n | Metric | Base | PCMV | Complete misses base / PCMV |
|---|---:|---|---:|---:|---|
| fg_le1 | 7 | fg_iou | 0.000000 | 0.000000 | 7 / 7 |
| fg_1to5 | 9 | fg_iou | 0.000000 | 0.000000 | 9 / 9 |
| fg_5to20 | 11 | fg_iou | 0.392297 | 0.392297 | 0 / 0 |
| fg_gt20 | 30 | fg_iou | 0.881543 | 0.881543 | 0 / 0 |
| clean_le1 | 4 | clean_iou | 0.000000 | 0.000000 | 4 / 4 |
| clean_1to5 | 10 | clean_iou | 0.106361 | 0.106361 | 5 / 5 |
| wm_over50 | 27 | clean_iou | 0.267867 | 0.267867 | 9 / 9 |

## Actual exposure and cost

Scorer training: 414 image views (90 real + 260 existing synthetic partial + 64 clean/full controls), 19,872 proposals, two action targets per proposal. Two predeclared variants, each with two 220-iteration regressors; no repeated endpoint search. Fitting and final evaluation took 14.88s. The GPU feature job took 21 min 21 s wall time, using only physical GPUs 0/1; base audit used an earlier 85 s job. Repairs ran concurrently and are not added again to wall time.

Feature generation covered 958 views total: 170 real, 260 synthetic partial, 128 training/calibration controls, 400 final audit controls. The final 400 controls are 200 message/cover realization pairs over 128 distinct dev photos. Full-image control embedding was repeated for the native512 supplement (528 embedding calls,264 distinct realizations); the partial images reuse exact original encodings. Original embedders, strengths, uint8 rounding and inference payloads were unchanged.

The initial CPU attempt failed before any regressor fit because metadata shadowed the numeric base map. Signed uint8 arithmetic in new real-message features was also repaired before fitting. All 170 real features were recomputed with unchanged masks/base predictions; regression checks and repair receipts are archived. The accepted #3188 prediction path always cast bits to float and was unaffected. A fit lock prevented the early CPU request and final worker callback from training twice.

No further PCMV training, test inference or leaderboard submission is justified by this screen. The best deployed pipeline remains #3188 / public 0.6736. The Slurm allocation completed and was released. [Validation gallery of the deployed PixelSeal improvement and failures](GALLERY.md).

## Scope and safeguards

See PROTOCOL.md for the frozen screening budget and SOURCES.md for pinned official implementation references. SAM2 resizing is restricted to geometry proposals. Official watermark decoder probes retain original 512 coordinates, exact uint8 replacement bytes and the original decoder preprocessing. Counterfactual boundaries may introduce distribution shifts; scores are learned features, not calibrated Gaussian significance. No GT enters proposal generation or feature extraction. Prior submissions and checkpoints remain preserved.
