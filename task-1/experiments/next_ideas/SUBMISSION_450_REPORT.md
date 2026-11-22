# StegaStamp 450-step continuation: submission #3054

Accepted public mIoU: **0.6677**. Previous best reported by the server: **0.6597**. Server improvement flag: **True**.

The artifact changes only StegaStamp. All PixelSeal and TrustMark masks are identical to submission #2675. Hidden final-set performance and per-family public scores are unavailable.

## Validation and selection

StegaStamp OOF mIoU: **0.744788**, including the fixed empty/full-mask fallback. This is three-fold development validation on 170 real images, one seed, with repeatedly inspected folds. The all-170 refit has no held-out validation score. Public scores were not used to choose the training endpoint, threshold or fallback.

Raw FP32 native 512-pixel model, threshold 0.5, no TTA. All backbone, local stem and decoder weights are trainable. AdamW uses backbone LR 3e-6 and new-module LR 1e-5; every effective batch contains 12 exact synthetic and 4 real training views. The original 2,500-source synthetic pool is reused without new embedding calls or post-quantization augmentation.

The fixed fallback changes only completely empty/full predictions, selecting the 164 highest minority-probability cells. It does not use GT at inference and does not establish that all hidden images contain both classes.

## Artifact checks

- 1,020 unique official test names, binary uint8 masks of shape [1020, 128, 128].
- File size: 16,734,774 bytes, below 20 MB.
- SHA-256: `e8abdf5626b4dd5af067cd8266adeb95a01f1fa19ecf03c9e5df82090e887714`.
- Changed images relative to #2675: `{"pixelseal": 0, "trustmark": 0, "stegastamp": 340}`.
- Receipt timestamp: 2026-11-22T04:15:16.761187+00:00.

Exact configuration, checkpoint hash and shard hashes are in `assembly_validation.json` and the frozen `selected_recipe.json`. Credentials were read from stdin and are absent from these files. Prior models and submissions are preserved.
