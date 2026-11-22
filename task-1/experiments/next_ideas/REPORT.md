# Next ideas: opportunity audit and StegaStamp adaptation

Accepted public best remains #2675, mIoU0.6440 until a new server receipt is recorded. This work was explicitly authorized after the TriForensic cancellation. [Frozen screening protocol](PROTOCOL.md).

## Findings before new training

All510 training masks contain both classes. Exact train metadata identifies85 full-canvas/restoration images per family; the remaining43 rectangular and42 fragment supports are geometry proxies, not definitive edit-type labels.

| Family | Protocol | mIoU | Empty masks | Full masks | Clean IoU on85 restored canvases |
|---|---|---:|---:|---:|---:|
| pixelseal | Historical #2409 OOF | 0.585655 | 41 | 6 | 0.285853 |
| trustmark | #2675 OOF + fixed crops | 0.657046 | 16 | 9 | 0.244718 |
| stegastamp | Synthetic-only, all170 real | 0.694962 | 7 | 0 | 0.442654 |

Clean-class errors on those255 restoration images contribute0.168898 score units, or47.7% of the measured total error. This is an error budget, not an achievable promised gain. The weighted error cells sum exactly to1 minus the510-image proxy mIoU0.645888.

The510-image number mixes the historical PS OOF with current TM/SS recipes. The newer PS #2675 continuation has only a57-image screen; do not call this current full OOF or infer the hidden score from it.

## A5: forced small regions are not a universal fix

| Family | Fixed fallback on empty/full masks | mIoU | Delta | Bootstrap95% delta | Improved / worse |
|---|---|---:|---:|---|---|
| pixelseal | top1pct | 0.587861 | +0.002206 | [-0.000033,+0.005263] | 11 / 36 |
| pixelseal | best13box | 0.588424 | +0.002768 | [-0.000245,+0.006800] | 7 / 40 |
| trustmark | top1pct | 0.657914 | +0.000868 | [-0.000340,+0.002407] | 5 / 20 |
| trustmark | best13box | 0.657513 | +0.000467 | [-0.000655,+0.001999] | 3 / 22 |
| stegastamp | top1pct | 0.697615 | +0.002653 | [+0.000304,+0.005660] | 6 / 1 |
| stegastamp | best13box | 0.697985 | +0.003024 | [-0.000121,+0.007244] | 3 / 4 |

These two rules were specified before measurement. Neither establishes that hidden test masks must contain both classes. PS/TM intervals include zero. StegaStamp top1pct is a small promising signal, not a large capability jump. All intervals are exploratory after extensive reuse of these development images.

## A4: native StegaStamp real-label adaptation

Three original folds,150 fixed updates,75% exact synthetic replay /25% real training examples. Raw weights,FP32,threshold0.5; no augmentation or architecture change. Initialization must match archived probabilities within1e-4. Update50 is progress only;150 is the predeclared endpoint.

Update0: mIoU**0.694962**, FG≤1%0.189824, FG1–5%0.360077, clean≤1%0.149905, clean1–5%0.313440, clean on WM>50%0.442654.

Update50: mIoU**0.705884**, FG≤1%0.212872, FG1–5%0.370857, clean≤1%0.134430, clean1–5%0.339969, clean on WM>50%0.462953.
Paired change: `{"mean": 0.01092229313179425, "ci95": [0.006816117437090043, 0.014849310783850326], "improved": 113, "worse": 50}`.

Update150: mIoU**0.722019**, FG≤1%0.211660, FG1–5%0.382853, clean≤1%0.163369, clean1–5%0.372081, clean on WM>50%0.491067.
Paired change: `{"mean": 0.02705743546142719, "ci95": [0.018781981571411394, 0.035188036306005456], "improved": 124, "worse": 42}`.

| Validation group | n | Before | After150 | Complete misses before / after |
|---|---:|---:|---:|---:|
| all / miou | 170 | 0.694962 | 0.722019 | — |
| fg_le1 / fg_iou | 12 | 0.189824 | 0.211660 | 7 / 6 |
| fg_1to5 / fg_iou | 33 | 0.360077 | 0.382853 | 8 / 6 |
| clean_le1 / clean_iou | 12 | 0.149905 | 0.163369 | 3 / 4 |
| clean_1to5 / clean_iou | 23 | 0.313440 | 0.372081 | 0 / 0 |
| wm_over50 / clean_iou | 85 | 0.442654 | 0.491067 | 3 / 4 |

All three folds improved at the predeclared endpoint. Tiny clean-hole mean IoU improved, but complete misses increased from3 to4: average gains do not mean every failure was fixed. The ten validation gallery panels include largest gains, largest regressions and tiny-region failures; [per-image gallery index](results/gallery/index.json).

The reproduction check passed before training: largest archived-probability difference across folds was5.25e-6. This verifies this cluster run, not the separate historical RTX3070 reproduction discrepancy.

Compute: three concurrent single-GPU workers;450 optimizer updates total,7200 image views total (5400 synthetic replay +1800 real draws). Fixed2500-source synthetic pool, zero new embedding calls. Per-fold wall seconds: [346.7, 346.3, 347.0]; training seconds: [285.5, 285.3, 286.1]. Actual peak allocated model memory was13.05GB per GPU. The fourth GPU completed the three-family oracle diagnostics and was then idle until the training workers released the node.

## Frozen submission candidate

Selected150-update adaptation: True. Postprocessing: `top1pct`; fixed1% fallback affects only completely empty/full predictions. OOF mIoU before fallback0.722019, after fallback0.724414. The fallback adds+0.002395; paired95% interval[0.00010245551590245343, 0.0055763488466821845]. This remains exploratory selection on reused development folds.

Final refit uses all170 StegaStamp real images with the same150-update replay policy. It has no held-out validation score. Four-GPU DataParallel preserves effective batch16 and the optimizer-update count; minor floating-point reduction differences from the single-GPU folds are possible. Keep PixelSeal and TrustMark test masks from#2675 bit-for-bit. No new threshold tuning, TTA or test-label use. [Frozen recipe](results/selected_recipe.json).

## One bounded extension to450 updates

The150-update model is preserved and is the first submission candidate. A fixed450 endpoint was declared before its results: restore150 model/optimizer/RNG, keep the same replay and learning rates, train300 more updates. Three fold workers plus one all170 worker share one4-A100 node. Select only for paired OOF gain>=0.005, positive bootstrap lower bound and no>0.01 regression in FG<=5% or clean IoU on WM>50%. The all170 worker supplies no validation evidence. Public scores do not enter selection. [Predeclared extension](results/extension_protocol.json).

The450 continuation has not produced its final validation result.

## A2: oracle ladder

GT defines diagnostic views only. Successful crop decoding does not prove the original edit type; failed decoding does not prove physical signal absence. Outside-GT blur/mean replacements change context and produce boundaries. The official wrapper performs one native resize (PS256/TM245/SS400), without an extra intermediate256 resize.

### pixelseal

| Group | n | Full ≥90% | GT bbox ≥90% | Outside GT blurred ≥90% | Outside GT mean ≥90% |
|---|---:|---:|---:|---:|---:|
| fg_le1 | 17 | 0 | 0 | 0 | 0 |
| fg_1to5 | 26 | 0 | 3 | 0 | 0 |
| holes | 85 | 81 | 81 | 80 | 82 |

### trustmark

| Group | n | Full ≥90% | GT bbox ≥90% | Outside GT blurred ≥90% | Outside GT mean ≥90% |
|---|---:|---:|---:|---:|---:|
| fg_le1 | 9 | 0 | 2 | 0 | 0 |
| fg_1to5 | 36 | 0 | 19 | 0 | 0 |
| holes | 85 | 80 | 80 | 78 | 79 |

### stegastamp

| Group | n | Full ≥90% | GT bbox ≥90% | Outside GT blurred ≥90% | Outside GT mean ≥90% |
|---|---:|---:|---:|---:|---:|
| fg_le1 | 12 | 0 | 0 | 0 | 0 |
| fg_1to5 | 33 | 0 | 8 | 0 | 0 |
| holes | 85 | 75 | 74 | 74 | 82 |

At the unchanged90% gate, exact GT boxes recover3/43 small PixelSeal,21/45 small TrustMark and8/45 small StegaStamp cases. Both PS and SS are below the proposed20% small-region crop-search milestone. Outside-mask blur/mean recovers zero small cases at this gate for all three families. Therefore this audit does not justify a large new PS/SS crop search or promise that masking background will recover small fragments. It also does not prove there is no local presence signal. Existing TrustMark verification is preserved.

## What the audit overstated

Re-embedding and gradient×HPF were already measured in message_localizer: real re-embedding macro AUC0.49812 PS /0.51902 TM. Two-class per-image losses and an unchanged frozen verifier already exist. A verified payload is not a pixel-perfect pseudo-mask, particularly when clean holes remain inside a decodable crop. The independent fair-coin null is not sufficient to lower a search verifier threshold after many correlated/adaptive crops.

A63-bit grayscale DCT pHash search found no pairs within Hamming distance4 among510 training images. This limited check does not prove absence of edited siblings. There is no test self-training and no use of held-out real masks in a synthetic mask bank.
