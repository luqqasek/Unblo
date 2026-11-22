# Watermark localization — current research handoff, 2026-11-22

**Best accepted public submission: #2409, mIoU 0.6369**, up from #2163 (0.6145).
Exactly one new submission was made after freezing the new validation recipe.
The hidden final score and per-family public scores are unknown.

Start with [small-region report](small_zoom/REPORT.md),
[generator comparison](region_data/REPORT.md), and
[RTX3070 diagnostic evidence](small_region_diagnosis/REPORT.md).
[Prior #2163 handoff](RESEARCH_HANDOFF_PRE_2409.md) is historical, not the current baseline.

## Subsequent local-presence screens (production unchanged)

Read [local_presence/REPORT.md](local_presence/REPORT.md) before proposing new work.
The bounded empty-mask TrustMark crop scan recovers two more ≤1% images without
GT proposals: reused OOF 0.652655→0.657046; tiny FG IoU0.080586→0.245392,
misses8/9→6/9. Other168 masks unchanged. Paired-bootstrap delta CI95[0,0.01107].
The gain is concentrated in two development images; no new submission was made.

PixelSeal paired local pretraining gives tiny-fold AUC0.74975 vs base0.52268,
but naive integration loses0.03602mIoU. Spatial correction reduces false positives:
fold1 0.557485→0.560130 (+0.002645), only1/7 tiny recoveries,1–5% FG IoU still0.
Do not compare this57-image screen directly with production170-image0.585655.
Further500-update local continuation + matched spatial correction fails to beat it;
real-patch adaptation hurts TM by0.01539. Clean-hole veto also regresses.

Three sequential4GPU/one-node jobs completed in14m17s allocated node time.
New checkpoints under local-presence-runs, exact encodings under local-presence-data;
SHA256/provenance and negative endpoints are saved. Regenerable caches6.44GB removed.
Next research should distinguish presence ranking from spatial extent/calibration,
and improve tiny-box proposals without expanding expensive exhaustive search blindly.
Use the exact existing TM AR≤2/raw100bit/native245 wrapper. No90%-gate requirement
for irregular remnants; no claim of physical undetectability from negative pilots.

## Current evidence and production

| Family | Development validation | Protocol / model |
|---|---:|---|
| PixelSeal | 0.585655 | Original 3-fold OOF, message_localizer, synthetic control600 +150 real/replay updates |
| TrustMark | 0.652655 | Original 3-fold OOF base0.593332 + GT-free NRGMark proposals/crop verification |
| StegaStamp | 0.694962 | Unchanged synthetic-only native RGB model,170 real validation images |

These are different development protocols, not one unbiased CV estimate. Folds
have been reused and proposal scales were tested adaptively. No test GT or
leaderboard threshold tuning was used. The accepted public score is separate.

TrustMark: frozen Adobe NRGMark-Q detector, source commit
4c60b9fdb4125fca3001f0803e6890a6b55371ac, proposes at512/1536/2560/4096.
Crops always use ORIGINAL512 pixels and the unchanged competition TrustMark0.8.0
Q/raw100-bit decoder. Refine fixed crop-size/offset grid; accept best candidate
at >=90% bit agreement and area<=10%; union with base threshold0.5 prediction.
No GT, edit type, GT area, clean counterpart or residual is an inference input.
Wrong-message control adds no corrections on170 validation images.

TM FG IoU1–5%:0.07164→0.50398 (36images). All19 rectangular cases improve from
FG IoU0 to0.81917 mean;17 irregular cases stay at0.15171. FG IoU<=1%:0→0.08059,
but only1/9 images recovered. Clean-hole metrics unchanged. This solves a
specific crop-geometry failure, not all small-region localization.

## What did not solve PixelSeal

New object/remnant generator after matched adaptation: PS0.58548 vs old-generator
control0.58566; TM0.59333 vs control0.58761. Better bbox fill did not solve tiny FG.

New512 train/64dev exact tiny-crop realizations, then200extra updates x32views:
PS synthetic-only real mIoU control0.56324, tiny data0.55957, tiny+Y residual0.55959.
A SINGLE fixed fold1 screen (57images), per user request:0.55706/0.55958/0.55958.
FG<=1% and1–5% remain0 in that screening fold. Do not call this full OOF.
The residual head's nonzero-residual MAE1.378549 vs zero predictor1.378558 shows
near-zero collapse under this particular L1 design and budget. No proof of a
fundamental ceiling. These experimental models were NOT used for #2409.

Latest base PS<=1% AUC0.4959, AP0.0057; oracle bucket threshold FG IoU0.0064.
TM<=1% AUC0.6415, AP0.0361; oracle FG IoU0.0195. Mere threshold/hysteresis is
not the main fix. Current architecture already has native stride4 skips and
stride1 local stem; don't repeat the false claim that all paths have stride32.

## Artifact locations and reproduction

Cluster root `/p/project1/training2666/pszenny1/native-localizer`:
- Production PS/TM: `small-zoom-production/{family}/last.pt`, config.json,
  inference.json, family-subset.pt; fixed150 updates,24synthetic+8real, LR2e-6/2e-5.
- PS synthetic initialization: `region-runs/pixelseal/control/last.pt`.
- TM synthetic initialization: `region-runs/trustmark/objects/last.pt`.
- NRGMark source/weights: `checkpoints/nrgmark`; fetch reproducibly with
  `experiments.small_zoom.fetch_nrgmark --root ROOT`, no package upgrade.
- Baseline OOF: `region-runs/{family}/{control,objects}-adapt/fold{1,2,3}`.
- New proposals: `small-zoom-finalscale/trustmark/proposals.json` (GT-free).
- Negative PS screening: `tiny-runs/{control,tiny,tiny_aux}`; real adaptation onlyfold1.
- Exact tiny encodings: `tiny-rect-data/pixelseal/{train,dev}`.
- StegaStamp unchanged: `runs/stegastamp/pilot-30min/last.pt`.

Final artifact: `experiments/small_zoom/results/submission/submission-small-zoom.pt`,
1020 unique names, binary uint8[1020,128,128],16,734,690bytes, SHA256
4221278af7cdd670059752305fa340225c12a04bf3547dd9766665c63ce14003.
Receipt and frozen recipe are alongside it. Large weights/maps remain outside Git.
Windows project has reports/dashboard and final submission; do not overwrite its
other agents' uncommitted work. Both dashboard languages must stay current.

In a Slurm allocation: `python -m experiments.small_zoom.production --root ROOT
--family trustmark` (or pixelseal). This preserves completed production outputs.
`small_zoom.assemble` combines the subsets with unchanged StegaStamp; upload is
separate and requires user authorization. User authorized #2409 this turn.

## Next goals and operating constraints

Prioritize PixelSeal representation and the still-missing smallest TM rectangles;
keep irregular remnants separate from newly embedded small crops. GT-bbox decoding
is an oracle diagnostic, never proposal input. Preserve exact default watermark
strength, raw bits, PNG/uint8 arithmetic, source splits and majority downsampling.
Locked synthetic source suite remains unopened. Do not use test images as covers.

User wants feedback within~30minutes, real validation mIoU and explicit small-FG
metrics, with ETA updates. A single fixed-fold screening estimate is now explicitly
acceptable; full CV is not required for every iteration. Substantial validation
improvement may be turned into a new authorized submission. Don't tune repeatedly
against leaderboard results. Keep public vs validation evidence separate.

A10040GB: micro16 / effective32 fit PS at24.45GB; micro32 OOM before first step.
FP32 and frozen BN; microbatch gradient equivalence tested. GPU only inside Slurm.
General unreserved jobs are user-authorized outside the reserved one-node limit.
Scheduler actually allocates whole4GPU nodes; pack workers when possible.
Do not log/store API keys. Old checkpoints and accepted outputs are preserved.
