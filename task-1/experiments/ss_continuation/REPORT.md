# StegaStamp: bounded450→1000 continuation

[Frozen protocol](PROTOCOL.md). One fixed57-image validation fold and one matching all170 refit, at most two A100 workers. Restore the best450-step weights, optimizer and RNG; unchanged12 synthetic +4 real views/update, LRs, loss, FP32 and pixel processing.

The earlier150→450 run improved full170-image OOF0.724414→0.744788. That motivates this extension but does not predict its gain. This experiment is a single-fold screen and must not be compared directly with the full-OOF number.

Two candidates were declared before training: raw1000 and equal mean probabilities450/1000. Threshold0.5 and the previously selected empty/full top1% fallback are fixed. Public scores do not choose the endpoint, blend or threshold.

## Final validation

Matched baseline mIoU: 0.762273. Selected: raw1000.

| Candidate | mIoU | Delta | Paired95% CI | Minority guard | Passed |
|---|---:|---:|---|---|---|
| raw1000 | 0.784104 | +0.021831 | [0.008520512565523457, 0.035988343353845396] | True | True |
| equal_probability_450_1000 | 0.778716 | +0.016443 | [0.008494600021606605, 0.025022797769286658] | True | True |

### raw1000: minority metrics

| Group | n | Metric | Before | After | Misses before / after |
|---|---:|---|---:|---:|---:|
| fg_le1 | 4 | fg_iou | 0.280969 | 0.350989 | 1 / 1 |
| fg_1to5 | 6 | fg_iou | 0.451241 | 0.469882 | 1 / 1 |
| fg_5to20 | 10 | fg_iou | 0.636646 | 0.721269 | 0 / 0 |
| fg_gt20 | 37 | fg_iou | 0.943656 | 0.943839 | 0 / 0 |
| clean_le1 | 3 | clean_iou | 0.298541 | 0.296201 | 0 / 0 |
| clean_1to5 | 12 | clean_iou | 0.444761 | 0.485976 | 0 / 0 |
| wm_over50 | 32 | clean_iou | 0.553446 | 0.585788 | 0 / 0 |

### equal_probability_450_1000: minority metrics

| Group | n | Metric | Before | After | Misses before / after |
|---|---:|---|---:|---:|---:|
| fg_le1 | 4 | fg_iou | 0.280969 | 0.350047 | 1 / 1 |
| fg_1to5 | 6 | fg_iou | 0.451241 | 0.458234 | 1 / 1 |
| fg_5to20 | 10 | fg_iou | 0.636646 | 0.687678 | 0 / 0 |
| fg_gt20 | 37 | fg_iou | 0.943656 | 0.945793 | 0 / 0 |
| clean_le1 | 3 | clean_iou | 0.298541 | 0.296911 | 0 / 0 |
| clean_1to5 | 12 | clean_iou | 0.444761 | 0.471428 | 0 / 0 |
| wm_over50 | 32 | clean_iou | 0.553446 | 0.579552 | 0 / 0 |

Completed in 19.20 minutes using two A100 workers on one node (job15905487); the sum of allocated GPU time is approximately 38.39 GPU-minutes. Training used550 additional updates per worker and8800 image views per worker; the all170 refit is not validation evidence.

Prepared all340 StegaStamp test predictions and replaced that family in submission#3457. PixelSeal and TrustMark remain bit-identical to that newer validated base. The earlier#3188 model parent is retained in recipe provenance. Submission validation passed1020 unique names, binary uint8[1020,128,128], and<20MB.

## Actual public receipt

HTTP200: `{"submission_id": 3485, "status": "success", "score": 0.6777, "previous_best": 0.6743, "delta": 0.0034, "improved": true, "message": "New best on this task. Score shown is on the public split; the final ranking uses the held out subset, revealed after submissions close."}`. Hidden final performance is unknown.
