# Current research handoff — 2026-11-22

## Start here

**Best accepted public submission: #2163, mIoU 0.6145**, up from #2008 (0.5993) and #1496 (0.5235). The final hidden score is unknown. The public server returns one combined score, not measured per-family test scores.

| Family | Current model | Real development mIoU | Protocol |
|---|---|---:|---|
| PixelSeal | Original extractor, two scales, message-conditioned progressive decoder | 0.571916 | 170 images, 3-fold OOF, seed42, fixed 150-step adaptation |
| TrustMark | Original extractor, two scales, message-conditioned progressive decoder | 0.578721 | Same protocol |
| StegaStamp | Native RGB ConvNeXt-v1 Base + learned local stem | 0.694962 | Synthetic-only, all170 real images, fixed threshold0.5 |

Do not average these as a uniform CV result. These are repeatedly used development images. The new PS/TM recipe was selected among several development candidates; bootstrap intervals are conditional on trained models and do not account for model selection or all source siblings.

Machine-readable evidence: [comparison.json](message_localizer/results/comparison.json), [submission receipt](message_localizer/results/submission/server_receipt.json), [frozen test recipe](message_localizer/results/submission/frozen_recipe.json). Full reports: [new PS/TM](message_localizer/REPORT.md), [native RGB pilot](native_localizer/REPORT.md), [previous hybrid](native_hybrid_submission/REPORT.md), [legacy scaling](synthetic_scaling/REPORT.md).

## What actually ran

1. Native RGB model: ImageNet-22K ConvNeXt-v1 Base, learned FP32 local stem, progressive128 output; all stages trainable. Exact original frozen embedders supplied the synthetic supervision. 2500 encoded sources/family,128 dev sources,200 updates ×64 views. StegaStamp transferred well; PS collapsed to clean (mIoU0.25303, FG IoU0); TM0.50601. This was a bounded pilot, not the requested full100k-example/50k-update program.
2. Submission#2008 combined native StegaStamp with the best old frozen-extractor heads. Old PS n4k_s2500 and TM n4k_s5000 each ensemble seeds42/7/123. Their OOF ensemble scores were0.558993 and0.547453. Mean individual-seed scores are different:0.553600 and0.544068.
3. Zero-training message diagnostics on170 images/family measured dense dot/cosine/soft-accuracy maps at original/384/512 scales, sliding windows, occlusion, gradient×HPF and re-embedding, including wrong-message controls.
4. New PS/TM model: original pretrained extractor shared at512 and native decoder scale (PS256, **TM245**); projected multi-level feature fusion, message maps and local RGB/Y/HPF/JND context. PS source implementation is ConvNeXtV2; do not silently substitute another ConvNeXt based on its name. TM BN running statistics stay frozen. FP32 throughout.
5. Synthetic training:400 updates ×32 views, first50 with frozen extractor, then350 joint. LR2e-5 encoder/3e-4 new modules,20-step warmup/cosine. Per-image BCE + both-class Dice and weak paired ranking. Procedural irregular masks/holes enabled immediately. No dense all-bits-per-cell auxiliary, COCO object-transplant generator or LaMa edits in this run.
6. Synthetic-only real transfer: PS0.519584, TM0.559286. Adaptation:150 fixed updates/fold, batch32=24 synthetic+8 outer-training real examples, LR2e-6/2e-5, threshold0.5. No outer-fold checkpoint/threshold tuning. All170 predictions joined by unique image name with identical baseline GT.
7. Final refit for#2163: identical150-step schedule using all170 real training images per family; same synthetic checkpoint initialization, threshold0.5, no TTA. StegaStamp predictions unchanged from#2008. Test file has1020 unique names, binary uint8[1020,128,128],16,734,789 bytes.

## Lessons supported by evidence

- **Exact extractor initialization plus the overall new recipe is useful**, particularly for TM. It does not isolate the causal effect of message conditioning, unfreezing, masks or architecture: those changed together.
- Adaptation with synthetic replay materially helps this recipe: PS0.51958→0.57192; TM0.55929→0.57872. This is not evidence to revert to aggressive real-only learning.
- PS improvement vs old ensemble is+0.01292 with paired95% CI[-0.00044,+0.02702]. TM+0.03127 with CI[+0.01325,+0.05022]. Fixed50/50 old+new fusion gives PS0.57084 and TM0.57151; neither beats the new adapted model's point estimate. PS fusion has narrower positive paired CI, but was not used in#2163.
- Known-message evidence exists, but it is not the claimed guaranteed16×/10× SNR gain. Dense cosine AUC: PS256 correct0.63352 vs wrong0.47581; TM245 correct0.60127 vs wrong0.49571. AUC alone cannot guarantee0.70–0.80 mIoU.
- Gradient×HPF was near random (~0.50); re-embedding was near random for PS and weak for TM. Neither entered the new model. Re-embedding a post-edited image does not establish pixel-aligned original residuals; source crop geometry matters.
- A parallel saved audit [known_message_audit](known_message_audit/REPORT.md) also found no large zero-training gains. Its Keep/Drop, integrated-gradient and learned-fusion protocols differ; do not mix their values with this experiment.
- Native scale is checkpoint-dependent. TrustMark wrapper uses245, not the assumed256. PS dense-to-global agreement max error4.77e-6, TM5.72e-6. Always verify actual wrapper/config and linear-head ordering.
- Original embedders, raw payload conventions and quantization were preserved. No stronger watermark, ECC/text conversion, WAM or MaskWM replacement was used.
- More total mIoU does **not** imply tiny-region localization works. After adaptation, FG IoU for GT≤1% remains0 in both families. For1–5% it is PS0.02435/TM0.06082. Overall minority IoU is PS0.22502/TM0.23678. Prioritize this failure rather than hiding it behind mean AUC.
- StegaStamp's stronger measured perturbations are a plausible explanation for the fast RGB pilot, not a proved physical cause of every performance difference. Failed200-step PS RGB training does not prove a fundamental RGB ceiling.

## Next useful work

Keep#2163 as the accepted baseline. Improve small remnants and clean holes with source-disjoint, exact-embedder examples of large-source fragments, local crops and irregular object masks; compare both minority polarities. Preserve clean/marked same-content twins to prevent editing shortcuts. Current source splits are retained; the locked source suite has not been opened.

Run bounded experiments giving first generalization feedback within about30 minutes; report real validation mIoU, FG/clean/minority metrics and elapsed/remaining time. Use actual throughput to choose batch size; count views and optimizer updates separately. A100 has40GB. Do not promise a0.70 score or run a large broken generator. Full three-seed confirmation and the complete proposed generator remain unexecuted.

Known message can be an inference input, but clean twins, residuals, GT-derived geometry and source provenance cannot. A wrong query message does not make the physical watermark mask empty. Do not force full local payload recovery from tiny surviving remnants. Geometric post-embedding augmentation/TTA requires separate validation.

## Reproduction and access

JURECA root: `/p/project1/training2666/pszenny1/native-localizer`.
- StegaStamp production: `runs/stegastamp/pilot-30min/last.pt`, raw model (not EMA), threshold0.5.
- PS/TM synthetic bases: `message-runs/{family}/matched-v1/last.pt`.
- PS/TM OOF: `message-runs/{family}/matched-v1-adapt/fold{1,2,3}/last.pt` and maps/metrics.
- PS/TM production refits: `message-runs/{family}/matched-v1-full-refit/last.pt`.
- Exact encoded training/dev data: `encoded/{family}/pilot-{train,dev}`; retain PNG bases/manifests, not frozen feature bottlenecks.
- Old heads and legacy caches remain on Windows: `L:/3 - Projekty pythonowe/European-AI-Hackathon/task-1/experiments/synthetic_scaling`.

Pinned dependencies and checkpoint/source hashes are in each experiment's provenance files and `native_localizer/pretrained.json`. Run GPU work only inside Slurm; never on login nodes. User authorized ordinary unreserved jobs; no need to wait exclusively for cispahack. No active GPU job is needed for further analysis.

Use `message_localizer.refit_job` for full refit+inference; `--inference-only` preserves completed refits. Ensure `PYTHONPATH=.:src` and the `src/watermark_localization/submission.py` module are deployed: its absence caused one failed inference launch, resolved without retraining. Code and small evidence are in Git; large tensors are intentionally excluded. Upload keys must never be committed. A public submission is an explicit user-authorized action, not an automatic model-selection loop.

See [cluster cleanup manifest](cluster_cleanup/REPORT.md) for retained/deleted files and measured recovered bytes. Deleted failed/debug weights cannot be recovered without rerunning; their measurements and configs remain.

Source-byte hashes in execution provenance describe the original host snapshots. Windows text copied into this handoff is normalized to LF in Git; semantic code is unchanged by that conversion. Legacy synthetic_scaling code in main is historical; its latest measured report/summary is archived here, while additional uncommitted Windows experiments are not overwritten or represented as rerun.
