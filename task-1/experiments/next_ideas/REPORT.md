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

**Training has not yet produced a complete150-update OOF result. No adaptation gain is claimed.**

## A2: oracle ladder

GT defines diagnostic views only. Successful crop decoding does not prove the original edit type; failed decoding does not prove physical signal absence. Outside-GT blur/mean replacements change context and produce boundaries. The official wrapper performs one native resize (PS256/TM245/SS400), without an extra intermediate256 resize.

## What the audit overstated

Re-embedding and gradient×HPF were already measured in message_localizer: real re-embedding macro AUC0.49812 PS /0.51902 TM. Two-class per-image losses and an unchanged frozen verifier already exist. A verified payload is not a pixel-perfect pseudo-mask, particularly when clean holes remain inside a decodable crop. The independent fair-coin null is not sufficient to lower a search verifier threshold after many correlated/adaptive crops.

A63-bit grayscale DCT pHash search found no pairs within Hamming distance4 among510 training images. This limited check does not prove absence of edited siblings. There is no test self-training and no use of held-out real masks in a synthetic mask bank.
