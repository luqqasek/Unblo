# Native RGB localizer — live execution record

This is a new experiment. Existing models, predictions and task wrappers are preserved.
No localization improvement or completed training result is claimed at initialization.

## Grounding

- Read `Watermark Localization.pdf` (all four pages), GPU slides and official starter.
- Earlier code/reports were absent from local HEAD `712350d`; fetched GitHub commit
  `7e811ae3be46a8bb7b4ef549bb645c8557e0df72` and inspected an isolated archive at
  `/private/tmp/native-localizer-reference`. A fast-forward was blocked by existing
  untracked task-2 files; no files were overwritten and no merge was performed.
- Read experiment index, research handoff, synthetic_scaling plan, original fold
  construction, STN-alignment runner and 2k reproduction guide from that commit.
- Best **reported** accepted public submission: #1496, 0.5235. The raw server receipt
  and ignored model/results files are not in this clone. Synthetic_scaling has no
  committed completed-results report. Do not call either report a new measurement.
- Verified unchanged wrapper: PixelSeal 256 raw bits, default strength and uint8
  rounding; TrustMark 0.8.0 Q, raw 100-bit binary, no ECC; StegaStamp 100 bits,
  WAVES residual, channel-mean subtraction, native resize and truncation.
- Official 4x4 majority conversion uses >=50%, including ties. Official per-image
  two-class IoU gives 1 for an empty union. No STN belongs in the new RGB model.

## Implemented and locally verified

- ConvNeXt-v1 Base, original 512 RGB, native 128/64/32/16 feature resolutions;
  pretrained safetensors loaded strictly after removing classification head.
- 256-channel progressive decoder, local FP32 learned stem, 128-grid main/local heads.
  Final trainable parameter count **93,777,762**; local receptive field **27x27**.
- Six meaningful tests pass: native shape/full-backbone gradient checks, normalization
  locality, empty/full loss stability, pair ranking, provenance with zero residual,
  majority ties and integer aspect-ratio constraints. Five existing tests also pass.
- Generator initial stage uses unchanged official embedders, uint8 PNG clean/marked
  caches, raw messages, rectangular embedding records and procedural hard supports.
  Local-crop embedding and surviving fragments of full-image embedding are distinct.
- Initial generator supports same-content debug data only. COCO object transfers,
  documented generated edits and full 60/25/15 mixture remain to be implemented and
  validated. Do not describe the debug generator as the full requested generator.
- Per-image both-class objectives, weak local auxiliary and changed-cell pair ranking.
  Main-head unchanged-pixel consistency remains disabled conservatively: a 27px
  local neighborhood is insufficient to cover the global branch's receptive field.
- Resumable AdamW/warmup/cosine/EMA/RNG training, exact batch-view/pair accounting,
  per-image AP/IoU/AUC diagnostics, internal boundaries, maps and failure galleries.

## Cluster

- User: pszenny1; account training2666; active reservation cispahack ends
  2026-11-22 14:00 CET. User explicitly chose to retain the empty-passphrase key
  after being informed this conflicts with the JSC published access requirement.
- Scratch creation failed with `Disk quota exceeded`; isolated directory instead:
  `/p/project1/training2666/pszenny1/native-localizer/`.
- Independent Python 3.11.18 environment installed with Torch 2.5.1+cu124,
  timm 1.0.31, ORT GPU 1.23.2. Exact environment freeze saved on cluster.
- First allocation 15903433 exposed four physical A100-SXM4-40GB devices, but stopped
  before the benchmark due to an incomplete wrapper upload. No training results.
- GPU jobs request exactly one node. Submission refuses if the user already has a
  queued/running job. A shared nonblocking lock also prevents concurrent experiments.
- COCO source split reserves 50,000 train / 2,000 dev / 2,000 locked source IDs before
  generation. Official image host has a mismatched TLS certificate; verified official
  path-style S3 endpoint returns 200 without disabling TLS verification.

## Still required

Actual encoded debug shard and fidelity checks; 100–200-step GPU profile and cost
estimate; fixed-batch overfit; unseen-source localization; validated full generator;
large corpus and three synthetic runs; matched controls; real nested transfer/CV
with existing manifests; seed confirmation; frozen final recipe and valid submission.
Ignored prior checkpoints/OOF/split files must be recovered or reconstructed exactly
from the recorded original code before attributing matched comparisons.

## Updated scope: fast quality pilot (user steering)

The user explicitly replaced the multi-hour initial run with a **maximum 30-minute
iteration to obtain quality information**, prioritizing validation mIoU as an
estimate of test generalization. Do not start the full 50k-update runs after the
pilot without incorporating that constraint.

- Exact StegaStamp smoke data: 64 train and 32 dev sources, source-disjoint.
- Completed fixed-batch diagnostic: 200 FP32 updates, batch16, 0.9993974 train mIoU.
  This is **memorization/capacity only**, not a validation result.
- Measured one-A100 FP32 batch2: 0.245294 s/microstep, 2.97 GB peak.
- Measured one-A100 FP32 batch16: 1.885719 s/update, 13.09 GB peak.
- Measured four-A100 FP32 batch16: approximately 0.57 s/update.
- Measured four-A100 deeper-BF16 batch16: 0.481491 s/update, 4.46 GB GPU0 peak;
  input normalization, first projection and entire local stem remain FP32.
- One-pair precision check was insufficient; added 16-pair, crop-scale-stratified
  response/gradient checks. AMP remains conditional on those checks.
- Entire official COCO train2017 archive downloaded (19,336,861,798 bytes) and
  extracted. Source IDs reserve 50k train / 2k dev / 2k locked; the fast pilot uses
  2,500 distinct training sources and a fixed separate synthetic development set.
- Original real outer folds reconstructed from the exact original algorithms and
  official metadata order. First real fold has 113 train / 57 validation images.
  No exact or DCT pHash<=4 sibling candidates found among the 510 real images.
  Ignored original manifests still unavailable; reconstruction is labelled.
- Current pilot is **synthetic-only StegaStamp**, fixed threshold0.5, raw final
  weights on real outer fold1 once. Real labels never train or select this pilot.
- Queued job15903579 had an estimated hour-long wait and was cancelled while still
  pending. Development partition is authorized for this user and had idle nodes.
  Pilot job **15903610** runs on `dc-gpu-devel`, one node/four A100, capped30min.
  It chooses batch64/128 by measured image throughput and <90% VRAM, not by labels.
- A generic login intermittently lacked the expected mounts. Pin SSH operations to
  `jureca03.fz-juelich.de` with HostKeyAlias=jureca.fz-juelich.de and strict verification.

## Completed pilot result

- Main experiment job15903610: **1,087.46s (18.12min)**, one node/four A100,
  batch64, FP32,200 updates /12,800 views. Additional-fold inference job15903703
  completed; no training or tuning in that job. No GPU allocations remain running.
- Source-disjoint AND message-disjoint synthetic dev mIoU:
  50 updates0.594261;100 updates0.671756;150 updates0.706671;200 updates**0.714179**.
- Real validation: **0.694961 mIoU on all170 StegaStamp images**, same frozen raw
  checkpoint and fixed threshold0.5. No real labels trained or selected the model.
- Fold1 **0.712742**, fold2 **0.664132**, fold3 **0.708244**; pooled per image,
  accounting for fold sizes57/57/56. These are partitions of one synthetic-only
  checkpoint's held-out real evaluation, not three separately fine-tuned models.
- Both-class metrics: FG IoU0.688344; clean IoU0.701580. Macro per-image FG AP0.835944.
  Conditional image bootstrap95% mIoU[0.673397,0.716414].
- GT<=1%: FG IoU0.189824,n12; GT1–5%:0.360077. Large-GT clean IoU0.442654.
  The smallest regions remain a material weakness despite the overall gain.
- Spatial shuffle0.447164; GT-oracle best constant per image0.460106. The actual
  model score is not explained by selecting a dominant-class constant alone.
- Independently recomputed official scores from all170 saved FP32 maps match the
  CSV/summary exactly. Verified2500 distinct train source IDs AND byte hashes,
 128 separate dev sources, no source/message overlap,100 raw bits and aspect<=16.
- Training actually sampled2435 distinct source IDs:9600 primary views+3200
  companion views,6400 views in matched pairs. These are views/recipes, not a claim
  of12,800 distinct RGB images. Encoded training pool2500+dev128 realizations.
- [REPORT.md](REPORT.md), frozen pilot config, metrics/maps/galleries in
  `task-1/outputs/native_localizer/artifacts/`; reproducible checkpoint remains at
  `/p/project1/training2666/pszenny1/native-localizer/runs/stegastamp/pilot-30min/last.pt`.
- This validates the short StegaStamp hypothesis. It is not an overall three-family
  hidden-test score, full nested real adaptation, final submission or completed
 100k-realization/50k-update study. Do not restart the multi-hour plan automatically.

## Three-family submission follow-up

User explicitly requested training PixelSeal and TrustMark with the same method
as StegaStamp, then predictions for all families and leaderboard upload.
`family_pilot.py` replicates the frozen 200-update, batch-64 FP32 recipe, using
2500 train / 128 dev exact-embedder realizations and the same source partitions.
The final raw checkpoint and threshold 0.5 are fixed before seeing test results.
`real_eval.py` now supports all three families with no change to its metrics.
Reserved PixelSeal job: 15903774. Unreserved development TrustMark: 15903778.
The user explicitly authorized unreserved jobs outside the reserved one-node
limit; those use per-job locks. Reserved jobs retain the shared one-node lock.
No API key is stored in this project. No upload has occurred at this stage.

Final follow-up: new PixelSeal pilot real170 mIoU=0.25302950 (FG IoU=0),
TrustMark=0.50601117; both rejected in favor of best legacy synthetic_scaling
recipes at the user's request. New StegaStamp retained (0.69496171).
Jobs15903778 and15903795 completed. Test-copy incompleteness caused first SG
inference job15903849 to fail; after syncing/checking all1020 files,15903864
completed. Planned all-native inference15903838 was canceled before execution.
Hybrid submission #2008 accepted at public mIoU0.5993 (previous0.5235).
See ../native_hybrid_submission/REPORT.md for measured OOF ensembles, full refits,
artifact hashes and server receipt. No leaderboard-based tuning performed.
