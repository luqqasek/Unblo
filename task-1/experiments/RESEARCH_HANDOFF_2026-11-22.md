# Watermark localization — current research handoff, 2026-11-22

**Best accepted public submission: #3188, mIoU 0.6736**, up from #3054 (0.6677).
Only PixelSeal changed to the EMA1561 global/carrier branch; TrustMark and
StegaStamp remain unchanged. Same 57-image PixelSeal development fold:
0.570883 → 0.581724; tiny foreground remains unresolved. See
[PCMV and submission receipt](pcmv/REPORT.md). Public per-family scores and
the hidden final score are unknown.

Start with [latest submission](next_ideas/SUBMISSION_450_REPORT.md), [latest experiments](next_ideas/REPORT.md), [previous experiments](local_presence/REPORT.md), [previous small-region report](small_zoom/REPORT.md),
[generator comparison](region_data/REPORT.md), and
[RTX3070 diagnostic evidence](small_region_diagnosis/REPORT.md).
[Prior #2163 handoff](RESEARCH_HANDOFF_PRE_2409.md) is historical, not the current baseline.

## Next ideas: active opportunity audit

A new user task explicitly authorizes further experiments and submission after
TriForensic was stopped. See [next_ideas](next_ideas/REPORT.md) and protocol.
A1 weighted error accounting attributes47.7% of measured error to clean-class
errors in255 restoration examples. Baseline audit uses historical PS #2409 OOF,
current corrected TM OOF and unchanged synthetic-only SS. All510 GT masks have
both classes, but this is not proof about hidden test. Fixed top1% fallback
improves SS0.694962→0.697615; PS/TM intervals include zero.

Three-fold SS adaptation completed:150 predeclared updates,75% synthetic replay
plus25% real, backbone LR3e-6/new1e-5, existing FP32 architecture/loss, raw
weights and threshold0.5. OOF0.694962→0.722019; paired delta0.027057,
CI95[0.018782,0.035188];124 improved,42 worse. All three folds improved.
FG<=1%0.189824→0.211660; FG1–5%0.360077→0.382853; clean on WM>50%
0.442654→0.491067. Tiny clean IoU improves0.149905→0.163369 but misses
increase3→4. Fixed empty/full1% fallback adds0.002395 to0.724414.
Three GPUs trained folds; fourth did oracle views; job wall6m26s on one node.
Initial probabilities reproduced within5.25e-6 before training. The all170
refit and340 test predictions completed;#3005/public0.6597 is the actual receipt.
All PS/TM test masks remain bit-identical to#2675. Public per-family scores are
unavailable. The predeclared450 continuation has completed; no node remains allocated for it.

Because the curve improved from50 to150, one extension to450 was declared
before its results: restore150 model/optimizer/RNG, same replay/LRs, three
fold workers and one all170 worker on one4-GPU node. Keep150 unless450 adds
>=0.005 paired OOF with positive CI and no>0.01 tiny-FG/clean-hole regression.
Public scores do not enter selection. These are reused development folds.

450 passed: raw mIoU0.743835, inherited fallback0.744788 versus0.724414,
delta+0.020373 CI95[0.014373,0.026550],127 improve/43 worsen. Clean IoU on
WM>50%0.491693→0.538708; FG<=1%0.220095→0.239436. Macro ROC-AUC slightly
regresses0.945884→0.943751, despite improved mask/AP scores. All170 refit and
340 test predictions completed; actual#3054 public0.6677. PS/TM unchanged.

Verified next research: [IML-ViT/SAFIRE and prior frozen-model negatives](next_ideas/VERIFIED_FORENSICS_RESEARCH.md).
Official IML-ViT checkpoint downloaded and SHA256 matches the owner's published
7631fe852e139d904ad96fd766a37120d703c966cb80dfad70ffae4ff426cce4.
No IML validation or training result yet. Frozen full TruFor/Mesorch/DetectiveSAM
were already tested in pretrained_forensics; do not repeat frozen fusion.

The oracle GT-box90% gate recovers3/43 small PS,21/45 TM and8/45 SS; outside-GT
blur/mean recovers0 small cases. This does not justify broad PS/SS crop search.
Re-embedding and dual-class losses are not new; their previous negative evidence
is retained. No test self-training or held-out mask bank.

## TriForensic unified screening update

See [TriForensic report](triforensic_wm/REPORT.md) and [protocol](triforensic_wm/PROTOCOL.md).
PS unified full+paired loop stopped at1561 at user request: fixed fold1 mIoU
0.581635 vs original0.557485 and newer #2675 screen0.570883. All16 FG≤5%
cases remain complete misses, as do all4 clean≤1% cases. At1000, local tiny
AUC0.73533/AP0.05701 does NOT propagate to the final map: final tiny AUC0.47172;
base logits−5.30 and correction+0.009 on tiny GT. Refiner contributes only
+0.000042 overall mIoU over its own base. Detached carrier skips also modify
that base, so this does not isolate the benefit of backbone continuation.

10k exact composites per family reuse2474 total source photos, not10k unique
photos. Train/dev source overlap0. Four GPUs on one node; no staged training.
Disabling activation checkpointing was slower despite higher VRAM use and was
reverted. **User then cancelled training for a different task.** TM last logged
update300; saved/evaluated checkpoint200 mIoU0.644154 vs0.646616 baseline.
The1000-update gate was never reached. All-real PS refit was cancelled before
starting. No new predictions or submission; all checkpoints/results preserved.

## Subsequent local-presence screens (production unchanged)

Read [local_presence/REPORT.md](local_presence/REPORT.md) before proposing new work.
The bounded empty-mask TrustMark crop scan recovers two more ≤1% images without
GT proposals: reused OOF 0.652655→0.657046; tiny FG IoU0.080586→0.245392,
misses8/9→6/9. Other168 masks unchanged. Paired-bootstrap delta CI95[0,0.01107].
The gain is concentrated in two development images; it was subsequently included in #2675.

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

## Full-model data-use follow-up

Full MessageLocalizer continuation is stronger than frozen corrections for PS:
fold1 0.557485→0.570883 after150 updates, fixed128 replay,25% real data,
all localizer weights trainable, original JND/BN running statistics frozen.
Paired image-bootstrap delta CI95[0.007745,0.019303], but this is reused development
validation. Tiny FG IoU remains0;1–5% FG IoU0.012724. More online replay diversity
(1212 primary source IDs actually sampled) gives0.565243. TM fixed replay0.640957,
online0.646204 vs base0.646616. Do not claim that more data cannot work:150updates
and changed mixture confound data scale with optimization and distribution.
The online arm adds no new public source identities or embedder calls; it draws
from existing2500 encodings,128 extras and512 thin-crop realizations.
The frozen PS continuation and TM empty-mask scan were submitted once as #2675,
public mIoU0.6440. PS changed279 test masks, TM8, SS0. Final hidden score unknown.
See local_presence/SUBMISSION_REPORT.md and the server receipt.

## Previous #2409 evidence and production

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

## Follow-up: continuation and shared clean-hole screen rejected

PS/TM450 continuation completed on the same57-image fold: PS fixed0.570883→0.568250,
PS online0.565243→0.564761, TM fixed0.640957→0.631045, TM online0.646204→0.645036.
No arm passed the predeclared selection gate; none was submitted. TM screen
uses capped original rectangles, without the later empty-mask scan. See
[next_ideas/MESSAGE_EXTENSION_REPORT.md](next_ideas/MESSAGE_EXTENSION_REPORT.md).

The new [IML-ViT clean-hole experiment](iml_holes/PROTOCOL.md) adapts the full
pretrained model across families on fold-training restoration images. Native512
input, explicit original1024 positional-parameter adaptation, no image resize,
FP32 first projection and deeper BF16. Target is clean regions; GT-free full
message agreement plus baseline predicted area gates a fixed equal blend.
All171 validation images receive newly computed original-decoder scores; prior
GT-filtered oracle rows are NOT used for routing. Official weights strict-load
and SHA256 checks passed. The150-update early-stop gate rejected this model: mean fixed-fold mIoU
0.659924→0.641382 (delta−0.018542); all three families regressed. No test
predictions or submission were made. See [iml_holes/REPORT.md](iml_holes/REPORT.md).

Completed: [PCMV](pcmv/REPORT.md), one PixelSeal scorer fit/evaluation.
The stronger EMA1561 global/carrier branch was deployed as #3188; TrustMark and
StegaStamp remain unchanged from #3054. Both PCMV and the proposal-only control
preserve the strongest base's 0.581724 mIoU on the fixed 57-image fold: calibration
selected no action, so neither made test predictions or a submission. The job
used GPUs 0–1, leaving GPUs 2–3 for the user's experiment, and has completed.
SS1000 was canceled while pending on user steering, with zero new updates.

## New best accepted: #3188 (2026-11-22)

Public mIoU **0.6736**, +0.0059 over #3054. Only PixelSeal changed to EMA1561
global/carrier branch; TrustMark and StegaStamp masks remain byte-identical.
Same57-image PS validation0.570883→0.581724. No all170 PS refit; trained113 real
images plus synthetic data. Small FG<=5% remains unresolved; see
[pcmv/REPORT.md](pcmv/REPORT.md) for exact buckets, artifact hash and receipt.
PCMV completed without gain (0.581724 → 0.581724); no duplicate submission or extended training. Tiny foreground and clean-hole failures remain unresolved.

## Native PixelSeal region screen: rejected

[Report](pixelseal_regions/REPORT.md): exact native thin-crop pairs from public
photos and GT-clean regions of real training images; new95,610-parameter region
classifier and500,494-box search, one fixed57-image development fold. The
predeclared geometry covers all16 small cases at IoU>=0.5 (oracle coverage only),
but recognition remains chance after400 updates: held-out AUC0.5027. FullFP32
audit AUC0.5016 confirms readout quantization alone does not rescue this checkpoint.
Inner calibration selects no changes; validation0.581724→0.581724, all16 small
foreground cases still missed. No test inference or submission. Two A100 workers,
143s allocated node time; node released. Best accepted remains#3188/public0.6736.
