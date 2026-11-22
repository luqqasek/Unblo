# RTX 3070: all-information controlled experiment

Frozen before new validation results, 2026-11-22. User explicitly requires every available data type and all official labeled images, not one family or a small real subset.

## Data contract

- All 510 official train images, 170/family, original three folds. Each appears exactly once in outer validation and twice in outer training; all outer-training records visited every epoch.
- All 4,000 train sources/family in the existing synthetic_scaling master pools (12,000 method/source realizations). The 200 source-held-out examples/family stay development checks; their labels never enter optimization. Historical overlapping experiment derivatives are not independent extra sources. Source IDs/hashes enforce identity and disjointness.
- RGB512 is an inference input; original pretrained decoder spatial features are inference inputs. Method chooses a family-specific model. All known message bits and signed decoder logits are retained individually for every family, including StegaStamp.
- Official mask128 is the primary loss. Mask512 supplies a subpixel auxiliary segmentation target and exact boundaries. Connected regions supply center/distance geometry supervision. Coverage is an auxiliary regression target. `full_watermark` is an auxiliary regime classification target, never an inference input.
- Exact synthetic clean counterparts supply a residual-energy auxiliary target and matched clean negative examples for the local RGB path. Synthetic mode/source/geometry metadata supplies target provenance and split controls, never privileged inference inputs.
- Image names are bookkeeping identities, never predictive features. Official test images have no GT; they are not training covers or tuning data. No submission in this experiment.

## Model and comparison

RTX-compatible multi-task decoder with native RGB/high-pass path, frozen original watermark-decoder pyramid features, per-bit evidence and known-message conditioning. Native weak-signal input operations use FP32. Full512 masks are predicted via a subpixel auxiliary head; primary inference remains128.

This is a new compact model, not full backpropagation through the best large submitted localizer. Compare it both with a matched compact control and stored best-system validation predictions. A gain against the control alone does not establish a gain over #2675.

Control and all-information arms use identical sources, folds, seeds, sample order, core capacity and optimizer-update counts. Control uses RGB, method, frozen decoder features, three aggregated message-evidence maps and mask128 supervision. The full arm additionally uses the individual bit evidence, explicit message vector and all auxiliary targets above. Both see every synthetic training realization and every eligible real training image; no replacement sampling that accidentally omits examples.

Fixed endpoints: two synthetic epochs, then20 real epochs/fold with synthetic replay. AdamW, seed42, effective batch8, microbatch selected by measured memory; full and control use identical microbatch. Constant threshold0.5, no outer checkpoint/threshold selection. Report all three folds, all families, paired changes, image-bootstrap intervals conditional on selected training runs, small-FG/clean metrics, memory, wall time and coverage counts. Development folds have already been inspected historically, so this is not a new independent holdout.

## Reproduction and limits

Prepare exact current inputs on the RTX host in a new experiment directory; preserve all old artifacts. Verify reconstructed synthetic masks against existing128 labels, source hashes and message strings. Newly generated RGB must use the original encoder/default strength/uint8 rules; record any pixel mismatch against historical hashes and recompute decoder features for those pixels rather than pairing stale features with a different image. No claims of a physical signal ceiling or a guaranteed public score.

Pre-validation implementation check: PixelSeal individual bit maps retain the larger native spatial resolution of both decoder scales. The measured maximum is16×16 for this original decoder, so the existing16 maps were already at its native maximum; a separate cache reconstruction verifies that no extra spatial downsampling is applied. This check was made before any training validation result. TrustMark retains its native high-scale16 maps; StegaStamp has a global100-bit decoder output. Cached maps are signed/tanh-normalized FP16; raw RGB remains uint8 at512 and the trainable path uses FP32.

Execution scheduling: family training starts as soon as its cache is complete. Preparation of later families may overlap on the same RTX3070/CPU after the memory preflight showed sufficient headroom. No other GPU is used. Record actual pipeline wall time; do not interpret per-arm wall-time differences as an isolated GPU performance benchmark while preparation overlaps.
