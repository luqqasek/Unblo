# PIPE screening from the current best PixelSeal model

Declared before new training results. Keep the accepted EMA1561 global/carrier
model from#3188/#3457 and its existing57-image validation split. Use at most
two A100s on one node. This is a reused development fold, not independent OOF.

Prepare512 new source-disjoint training photos,64dev and64locked from the
pinned PIPE shard. Exclude every existing COCO source assignment. Screen decoded
originals against510 competition training images and retained new photos using
exact pixel SHA256 and pHash<=4. No competition test pixels used. Follow the
author-confirmed stretch-to512 for edit masks, then threshold>=128; preserve
all image512 uint8 pixels. All transformations precede embedding.

Generate one exact full-image PixelSeal embedding per train/dev original with
random raw256bits using the unchanged starter wrapper. Cache clean, edited,
marked, edit support and provenance. Never resize/compress/blur marked pixels.
For a clean hole, overwrite declared support with either the same clean cover
or PIPE inpainted pixels. Keep an equal number of both mechanisms. For a
surviving marked object, copy original marked object pixels onto its inpainted
background, restricting watermark support to the declared object region.

One fixed200-update adaptation, two parallel arms from identical EMA1561
weights and fresh identical AdamW states. Effective batch16,12synthetic+4real
training images, matched real draws. A uses existing replay; B replaces6of12
synthetic views with new PIPE constructions. Remaining old replay preserves
small crops, remnants, holes and controls. Of new views50%are clean holes,
25%marked fragments,25%full clean/marked controls. This tests source/content/
geometry together, not a pure effect of dataset size. A later same-source
control can separate inpainting content if the mixture helps.

Train the existing global extractor/decoder/skips with encoderLR2e-6 and
new-moduleLR2e-5, AdamW weight decay0.01, clip1.0. Keep the previously trained
carrier fixed in both arms; no new carrier/refiner loss. Loss is existing
per-image0.5BCE+0.5dualDice in both arms. No new architecture, augmentation,
threshold search or payload convention. FP32 pixels, filters, normalization,
first projections and loss; BF16 deeper layers as in parent inference.

Evaluate initialization, update50(progress only), and200(fixed endpoint).
Reproduce parent predictions before training. Compare official mIoU, FG IoU
<=1%/1-5%, clean IoU<=1%/1-5%/WM>50%, misses and paired image bootstrap.
Apply the fixed#3457 top164 local-cell fallback only to empty masks in both
arms. Keep all candidate results. PIPE must improve mIoU by>=0.005 over both
initialization and matched old replay, with positive paired CI lower bound
and no>0.01 regression in combinedFG<=5% or cleanIoUWM>50% before scaling.
Do not submit from a failed or incomplete screen. Preserve all old checkpoints.
