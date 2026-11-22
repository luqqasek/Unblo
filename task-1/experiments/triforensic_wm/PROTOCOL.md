# TriForensic-WM: fixed-fold unified screening

The approved experiment starts with PixelSeal, then TrustMark, on the existing
fold1 (113 real training images,57 validation images per family). These are reused
development folds. Reported scores are engineering estimates, not unbiased test
predictions. The public best #2675=0.6440 remains unchanged by this experiment.
StegaStamp is preserved and is not retrained in this PS/TM screening.

Source inspection confirms the official PixelSeal `convnext_tiny` factory builds
`ConvNeXtV2` blocks with GRN, depths[3,3,9,3], widths[96,192,384,768]. The earlier
native RGB experiment used ConvNeXt-v1 Base; these are different existing models.
The current Stream3 reuses the official watermark extractor, without substitution.

## One training loop

One AdamW optimizer and one OneCycleLR schedule (`pct_start=.05`,3500 requested
updates) jointly optimize the existing message localizer, the new local carrier,
detached local-to-global skips and the observability refiner. Every update uses
16 full images (12 synthetic,4 real) and64 clean/marked patch pairs (128 views).
Losses are summed before one backward. There is no patch-pretraining phase,
freezing/unfreezing schedule or intermediate model handoff. Original pretrained
extractor+decoder weights come from the existing fold1 checkpoint, never from
a refit that saw the validation images. Noiseprint++ and exact JND are frozen
throughout by design. TrustMark BatchNorm running statistics remain frozen.

Max learning rates: existing model2e-5, new modules4e-4; weight decay.01 excluding
bias and normalization parameters; clip1; EMA.995. Full-image module dropout25%
and Noiseprint channel dropout15% apply only during training. Threshold.5 is
fixed. Observed uint8 RGB, filtering, JND, first image projections and losses use
FP32; deeper convolutions use BF16. A paired fidelity audit precedes updates.
The target hardware is ONE node with4 A10040GB. DataParallel splits each full and
paired batch across all four devices and reduces into one optimizer.

The3500-step/30-minute estimate is an unverified proposal, not a performance
claim. Actual100–200-update throughput determines ETA. A30-minute allocation has
a1650-second work budget, then saves resumable model/EMA/optimizer/scheduler/RNG
state and evaluates. Resumption continues the same loop/schedule.

Measured first200 updates:206.3 seconds of training,1.032 seconds/update,
peak14.61GB allocated on GPU0. The original30-minute estimate for3500 updates
was therefore too optimistic. A carrier-only memory benchmark (not another
model training) measured3.45GB with activation recomputation versus6.84GB
without it for one512 image; timings0.311 versus0.247 seconds while sharing the
allocated node. Continuation may disable carrier activation recomputation to
use more VRAM. Forward/gradient equivalence is tested. An OOM retries identical
views/RNG with recomputation before any optimizer update. No learning stages,
new optimizer or schedule reset are introduced. A tested alternative fused
normalization kernel was slower and was NOT adopted.

## Locality and explicitly resolved contradictions

User approved: strictly local Stream1, explicit contextual inputs to the refiner.
Standard spatial GroupNorm would violate locality. Instead normalization uses
eight channel groups independently at each spatial position. The carrier's
90 RGB SRM outputs (30 filters applied separately per color),8 Bayar filters,
HPF RGB,Y and2 observability channels feed a160-channel stem, four two-convolution
residual blocks and two stride2 convolutions. Actual trainable parameters:
2,127,329. RF:5→7→23→25→29 original pixels. An input-gradient test verifies this.
Bayar centers are-1 per input kernel; surrounding weights sum+1. No InstanceNorm
or signed L1 residual prediction is used.

The refiner's three3x3 convolutions have RF7 at128, but ring radius9, frozen
Noiseprint RF35 and global spatial median expand its effective context. It is
NOT described as a strictly local head. Global decoder skips receive
`f_loc.detach()`. The refiner may backpropagate to the carrier; this does not
change the carrier's forward receptive field.

Existing internal512+256 PixelSeal/512+245 TrustMark feature scales and original
crop-verifier resizing are retained as model operations. No additional image
augmentation resizes observed real or quantized synthetic images. The synthetic
request's blur/denoise/JPEG modes conflict with its hard constraint, so they are
omitted. Clean overwrite modes are exact cover restoration, clean natural crops
and flat patches. No compression stratum is added without measured justification.

## Exact synthetic data

10,000 training composites and640 source-disjoint development composites per
family are requested. Source photographs reuse the current COCO source manifests;
the realization count is NOT a claim of10,000 unique photographs. Embedding calls,
unique sources, background/donor identities, messages, initial rectangles,
integer translations, final supports, perturbation RMS and unchanged-pixel
fractions are recorded independently. Original competition wrappers/weights,
strength, random raw bitstrings and uint8 rounding are unchanged. Masks come
from provenance;4x4 targets include ties at8/16. Clean twins undergo identical
compositing with clean donor pixels. Cross-image fragments use different photos.

Pool proportions:20% standard,20% direct tiny-crop embedding,20% surviving large-
source fragments,25% clean holes/intact low-observability hard negatives,15%
clean/full/disconnected controls. Full-image sampling is uniform over five
buckets, as specified separately by the user. Effective view counts include128
paired crops in addition to16 full images, rather than claiming a batch of16.

PixelSeal tiny rectangles have shorter side12–32 and initial AR<=16. TrustMark
retains the official initial AR<=2, with shorter side16–96 to cover small areas;
AR16 would violate the exact task conventions. Fragment support can be thin or
irregular regardless of initial rectangle AR. No geometry is learned from
validation masks. Dev and train source/donor/background identities are checked.

Noiseprint++ is the original standalone pretrained DnCNN17 checkpoint, not a
newly trained model or a claim of the full TruFor segmentation system. Source
commit, hashes and license are retained in `noiseprint_provenance.json` and the
fetch script. Its official RGB/256 input is used. All170 real images may receive
label-independent cached Noiseprint features; only113 training labels are used
for updates. Synthetic composite Noiseprint maps are also cached.

## Reporting and gate

Evaluate the original FP32 #2409 model on exactly the same57 images, then the
new initialized model, update100,1000,2000,3000 and3500 (or an earlier saved wall
budget endpoint). Report official mean per-image mIoU, both IoUs, macro AP and
ROC-AUC, FG<=1%/1–5%/5–20%/>20%, clean<=1%/1–5%, clean IoU on WM>50%, and miss
counts. TrustMark1–5% is split by rectangular versus irregular shape for offline
diagnostics. Absent positive classes are excluded from AP; official empty union
IoU stays1. No threshold tuning on the validation masks.

At update1000 PixelSeal must have real tiny local-map AUC>.55 AND synthetic
thin-strip interior paired AUC>=.65. Failure stops optimization for diagnosis.
The matched fold1 old local-map AUC is approximately.52268 on7 tiny images; the
research brief's.4959 referred to a different170-image aggregate.

An interior score requires the entire RF29 to lie inside support AND inside the
64px crop. Strips narrower than29 cannot have such cells. They are excluded from
that diagnostic with explicit counts, not relabeled interior. Boundary-inclusive
and JND-quartile diagnostics are reported separately. They remain training data.

## TrustMark verifier preservation

Use union of existing multi-scale proposals,45 edge-covering tiles (9 of256px,
36 of160px;25 tiles would leave edges uncovered), and top8 local connected
components. Soft-NMS selects16 seeds. Three coordinate-search rounds independently
modify x,y,width,height and optimize signed tanh-logit agreement. Original uint8
pixels are recropped for the exact unchanged raw100-bit decoder. Acceptance is
integer correct-bit count>=90, avoiding float32 `0.9` comparison ambiguity.

Previously accepted boxes remain eligible. Accepted boxes must occupy<=15%,
and added support excludes `delta_clean>2` (predeclared screening veto threshold).
Large rectangles cannot erase clean holes. GT boxes are used only AFTER
inference in a separate four-way offline audit. The new verifier is not claimed
validated or production-ready until its measured results are saved.

## Commands (inside an allocated4-GPU node)

```sh
python -m experiments.triforensic_wm.fetch_noiseprint
python -m experiments.triforensic_wm.tests
python -m experiments.triforensic_wm.prepare_job --family pixelseal
python -m experiments.triforensic_wm.train --root ROOT --family pixelseal
# If the same loop was paused at its wall budget:
python -m experiments.triforensic_wm.train --root ROOT --family pixelseal --resume
```

Repeat preparation/training for TrustMark after the PixelSeal screening. Large
pools/checkpoints remain on project storage; reports/configs/metrics and
provenance are versioned. No new leaderboard submission is implied by a screen.

Primary implementation sources: [TruFor](https://github.com/grip-unina/TruFor),
[OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html).
