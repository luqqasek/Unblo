# Local evidence and tiled TrustMark proposals

Single node, four A100 40GB workers, seed42. Fixed fold1 screening for learned
branches; existing170-image OOF for the inexpensive frozen proposal addition.
No test inference or leaderboard selection in this screen. Baseline #2409 retained.

Workers: PixelSeal paired local presence; TrustMark paired local presence;
PixelSeal one-sided clean-hole correction; TrustMark tiled proposals/refinement.

Local encoder: fixed30 SRM filters on Y (pinned source/hash in srm_source.json),
8 trainable constrained RGB filters, high-pass RGB, Y and local high-pass energy.
No spatial or batch normalization. FP32 first projection, filters and all layers;
TF32 off. 372737 parameters, analytical and gradient-verified RF27, stride4 output.
This is a bounded local-evidence pilot, not a reproduction of full SRNet/TruFor.
Original JND remains in the existing main model; this auxiliary uses a texture
proxy. Noiseprint++, ring contrasts and new boundary losses are not in this pilot.

Exact new embedding: up to512 distinct train and128 distinct dev sources from
existing disjoint manifests. PixelSeal short12–32px and aspect1–16; TrustMark
short16–64px and aspect1–2. Native source pixels, exact default wrappers and PNG
quantization. Separate full-source remnants and same-content clean restorations.
No post-embedding JPEG/blur/resampling. Pair bank2048train/256dev,64x64 observed
patches, original4-grid alignment, provenance masks with >=8/16 majority rule.
New rectangular embeddings and remnants/hole patches each receive one third of
presence-bank entries; holes worker uses full marked/twin-restored pairs.

Pretrain1000 updates x128 views, AdamW3e-4, clipping1.0. Objective .5 per-image
class-balanced BCE +.5 unweighted BCE +.1 changed-cell ranking +.05 patch presence.
Patch presence is a log-mean-exp of logits, not an additional inference input.
Report local synthetic and real fold1 metrics at250 and1000; endpoint fixed1000.
The patch bank fits in host memory; no GPU feature extraction on login nodes.

Integration freezes the already adapted fold1 baseline and the newly trained
local encoder. Fit only a zero-initialized1x1 correction to base logits, bounded
to +/-8 (presence) or [-8,0] (holes), using64 local features and base logit.
300updates,lr1e-3,8real fold-training+24synthetic full-image examples. Both
features and actual baseline logits are precomputed for this bounded calibration
test; the trainable main pipeline is preserved. Targets: class-balanced BCE and
both-class per-image softIoU. Fixed threshold.5, endpoint300;100 is diagnostic.
No validation image participates in optimizer updates. Corrected TrustMark
retains accepted old rectangles after the learned correction.

Tiled inference: 256stride128 (9tiles),160stride80 with edge coverage (36tiles),
detector-only resize1024, batches4. Original512 pixels for native245 decoding.
Deduplicate at boxIoU.8,top16 seeds; three rounds of independent translation and
axis-size changes8/4/2px. Search by signed tanh(logit/2); final hard>=.9 verifier.
Keep area<=10%,minside8. Preserve existing accepted rectangles and add at most
one newly accepted rectangle. Wrong-query control scored on identical searched
candidates; not a formal false-positive guarantee. GT is used only after inference
for evaluation and proposal-recall diagnostics.

Research corrections: exact conditional-median collapse is not established by
marginal residual sparsity; negative pilot does not establish physical
undetectability. TrustMark initial aspect must<=2, not10. RF31 is inconsistent
with an additional48px surround. Existing OR already has area<=10 and did not
change measured clean-hole cohort. Proposal unions do not guarantee no metric
regression: additional false-positive pixels remain possible.

Primary references checked: SRNet paper https://ws.binghamton.edu/fridrich/Research/SRNet.pdf ;
NRGMark https://collomosse.com/pubs/Agarwal-WACV-2026.pdf ;
constrained-filter author code https://github.com/LuminolT/constrained-conv-TIFS2018 .
These motivate experiments, not expected competition scores or wall-clock promises.

## Follow-up after first measured endpoints

First screen did not justify deployment: PS presence -0.03602mIoU, PS holes
-0.03312, TM presence -0.01969 on fold1. Tiled TM +0.00062 over170 images,
no new<=1% recovery. PS local tiny ROC-AUC rose0.52268→0.74975 on7fold1images,
but the integrated map added too many false positives. These findings motivate
the second screen; they do not establish physical undetectability.

Second single-node4GPU batch: two PS spatial refiners, two halves of a bounded
TM scan. PS refiners use the same fixed local1000 checkpoint, zero-initialized
three3x3 layers with dilation1/2/4, effective RF83, delta bounded+/-8.
600updates x16views (12synthetic+4real),lr2e-4, seed42, fixed fold1/threshold.5.
Compare ordinary pixel BCE + both-class Dice vs BCE + both-class softIoU,
removing the first screen's strong per-class BCE balance. Real and synthetic
feature caches in this follow-up remain FP32 (first screen's deep feature cache
was FP16; raw inputs and feature extraction were FP32). Base and local encoder
remain frozen.200updates diagnostic;600 endpoint fixed before this screen.

TM scan applies only if the original #2409 full pipeline predicts an empty
mask; existing successful rectangles cannot change under this rule. Eligibility
uses no GT. Search w/h24..80 in8px steps,aspect<=2,positions16px apart plus
frame endpoints; divide eligible images across2GPUs. Batch256 bilinear ROI
screening, top32 soft-match candidates, exact original native245 resize for
refinement4/2/1px and final>=90% hard verifier. One wrong message per image is
evaluated on the same candidates. Full old masks preserved; only an accepted
rectangle can be added to an empty mask. Fixed grid, no GT-coordinate seeds.
This is an adaptive development follow-up, not fresh validation or new OOF.

## Matched continuation after spatial screening

Third one-node/four-GPU batch: PS synthetic-only, PS synthetic+real patches,
TM synthetic-only, TM synthetic+real patches. Start from each local1000 model;
500 updates, seed73, LR1e-4,128 observed views/update, same local loss.
Control uses64 synthetic pairs; real variant48 synthetic pairs+32 real patches.
Real patches come only from113 outer-training images,4 native64px crops per
present class per image, centered with TRAINING GT, no geometric resampling.
This matches updates/views, not synthetic-only exposure: replacing32 views
with real data deliberately changes their source distribution.
Both then use the same frozen-base600-update FP32 spatial-IoU correction,
seed42,12synthetic+4real views. Fixed endpoints; no validation threshold search.
All TM evaluations retain the old #2409 verified rectangles; the new empty-mask
scan is evaluated separately and not silently combined with learned corrections.

Measured: PS control0.55803, real0.55937, both below earlier spatial0.56013.
TM control0.64795 (+0.00133 vs same-fold base), real0.63122 (-0.01539).
Real patch adaptation does not justify promotion. Full metrics and image
bootstrap intervals are in completed.json. These short negative runs are not
proof that real adaptation or carrier detection cannot work at a different budget.

Three Slurm jobs completed:15904587 (4m46s),15904621 (6m52s),15904639 (2m39s).
Total allocated node time14m17s; each one node/four GPUs, sequential allocations.
Approximately0.952 allocated GPU-hours, not measured active utilization.

## Full-model data-use screen requested during the work

User asked whether real training data and available synthetic data were being
used adequately. The preceding screens froze the full adapted base model and
used128 fixed synthetic compositions to train correction heads. The main
historical model had already used2500 synthetic training source rows and full
real/replay adaptation; these quantities must not be confused.

Fourth one-node/four-GPU batch: each family's full MessageLocalizer starts from
the unchanged fold1 production-baseline checkpoint. ALL encoder/decoder/context
weights are trainable; original JND and ResNet BN running statistics stay frozen.
Compare128 fixed full_examples replay compositions vs online draws from existing
2500 original encodings +128 extra encodings and512 exact thin-crop realizations.
No new independent public source identities or embedding calls are added by this
screen; it expands the subset and composition diversity used in this continuation.

Online mixture:50% existing RegionData sampler (which itself retains50% legacy
examples),25% exact newly-small-crop realizations,25% large-source remnants or
same-content clean holes. RegionData includes COCO objects and cross-image edits.
This is a combined replay-volume/distribution test, not an isolated data-size law.
150updates x16views (12synthetic+4real), identical real RNG/exposure, encoderLR2e-6,
new modulesLR2e-5, seed4242. Native FP32, no JPEG/blur/new resize. Batch16 when it
fits; fallback8x2 accumulation if first backward OOM. Loss unchanged both-class
Dice+BCE. No new paired auxiliary. Validate0/50/150; endpoint150 fixed.
The new TM empty-mask scan is not applied in this comparison. All old #2409
rectangles remain. Same fold reused; report developmental optimism explicitly.
