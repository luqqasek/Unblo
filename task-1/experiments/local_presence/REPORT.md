# Local presence: small watermark experiments

Development screens on one Slurm node at a time, four NVIDIA A10040GB workers. New accepted submission#2675 has public mIoU0.6440, above#2409/public0.6369; all previous artifacts are preserved. Learned branches use fixed fold1 (57 validation images per family); frozen proposal screens reuse170-image OOF. These repeatedly inspected development estimates are not independent hidden-test estimates.

| Run | Endpoint | n | mIoU | Delta | FG ≤1% | FG 1–5% | FG 5–20% | FG >20% | Clean ≤1% | Clean 1–5% | Clean in >50% WM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pixelseal-presence | baseline | 57 | 0.55748 | 0.00000 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.00000 | 0.08307 | 0.23200 |
| pixelseal-presence | final | 57 | 0.52147 | -0.03602 | 0.03418 | 0.04219 | 0.27636 | 0.85977 | 0.00051 | 0.05367 | 0.14018 |
| trustmark-presence | baseline | 57 | 0.64662 | 0.00000 | 0.00000 | 0.52530 | 0.27785 | 0.88487 | 0.00000 | 0.10383 | 0.24857 |
| trustmark-presence | final | 57 | 0.62692 | -0.01969 | 0.00000 | 0.47876 | 0.36298 | 0.88013 | 0.00000 | 0.10169 | 0.20257 |
| pixelseal-holes | baseline | 57 | 0.55748 | 0.00000 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.00000 | 0.08307 | 0.23200 |
| pixelseal-holes | final | 57 | 0.52437 | -0.03312 | 0.00000 | 0.00000 | 0.25114 | 0.77059 | 0.00112 | 0.05690 | 0.21164 |
| tm-tiled | baseline | 170 | 0.65265 | 0.00000 | 0.08059 | 0.50398 | 0.52614 | 0.88561 | 0.00568 | 0.14776 | 0.24472 |
| tm-tiled | final | 170 | 0.65328 | 0.00062 | 0.08059 | 0.50015 | 0.53897 | 0.88561 | 0.00568 | 0.14776 | 0.24472 |
| ps-spatial-dice | baseline | 57 | 0.55748 | 0.00000 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.00000 | 0.08307 | 0.23200 |
| ps-spatial-dice | final | 57 | 0.55973 | 0.00224 | 0.04800 | 0.00000 | 0.30187 | 0.85896 | 0.00088 | 0.07881 | 0.23075 |
| ps-spatial-iou | baseline | 57 | 0.55748 | 0.00000 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.00000 | 0.08307 | 0.23200 |
| ps-spatial-iou | final | 57 | 0.56013 | 0.00265 | 0.03734 | 0.00000 | 0.30397 | 0.86116 | 0.00120 | 0.07746 | 0.23151 |
| tm-scan | baseline | 170 | 0.65265 | 0.00000 | 0.08059 | 0.50398 | 0.52614 | 0.88561 | 0.00568 | 0.14776 | 0.24472 |
| tm-scan | final | 170 | 0.65705 | 0.00439 | 0.24539 | 0.50398 | 0.52614 | 0.88561 | 0.00568 | 0.14776 | 0.24472 |
| ps-adapt-control-spatial | baseline | 57 | 0.55748 | 0.00000 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.00000 | 0.08307 | 0.23200 |
| ps-adapt-control-spatial | final | 57 | 0.55803 | 0.00054 | 0.00977 | 0.00000 | 0.30110 | 0.86137 | 0.00000 | 0.07833 | 0.23115 |
| ps-adapt-real-spatial | baseline | 57 | 0.55748 | 0.00000 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.00000 | 0.08307 | 0.23200 |
| ps-adapt-real-spatial | final | 57 | 0.55937 | 0.00188 | 0.00000 | 0.00000 | 0.30841 | 0.86546 | 0.00174 | 0.07572 | 0.23126 |
| tm-adapt-control-spatial | baseline | 57 | 0.64662 | 0.00000 | 0.00000 | 0.52530 | 0.27785 | 0.88487 | 0.00000 | 0.10383 | 0.24857 |
| tm-adapt-control-spatial | final | 57 | 0.64795 | 0.00133 | 0.00000 | 0.53003 | 0.29233 | 0.88253 | 0.00000 | 0.11199 | 0.25019 |
| tm-adapt-real-spatial | baseline | 57 | 0.64662 | 0.00000 | 0.00000 | 0.52530 | 0.27785 | 0.88487 | 0.00000 | 0.10383 | 0.24857 |
| tm-adapt-real-spatial | final | 57 | 0.63122 | -0.01539 | 0.00000 | 0.46847 | 0.27557 | 0.88120 | 0.00000 | 0.10850 | 0.24920 |
| ps-data-small | baseline | 57 | 0.55748 | 0.00000 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.00000 | 0.08307 | 0.23200 |
| ps-data-small | final | 57 | 0.57088 | 0.01340 | 0.00000 | 0.01272 | 0.34846 | 0.88177 | 0.00000 | 0.07553 | 0.23986 |
| ps-data-online | baseline | 57 | 0.55748 | 0.00000 | 0.00000 | 0.00000 | 0.30093 | 0.85988 | 0.00000 | 0.08307 | 0.23200 |
| ps-data-online | final | 57 | 0.56524 | 0.00776 | 0.00000 | 0.00611 | 0.34499 | 0.88747 | 0.00000 | 0.06151 | 0.21516 |
| tm-data-small | baseline | 57 | 0.64662 | 0.00000 | 0.00000 | 0.52530 | 0.27785 | 0.88487 | 0.00000 | 0.10383 | 0.24857 |
| tm-data-small | final | 57 | 0.64096 | -0.00566 | 0.00000 | 0.51644 | 0.33905 | 0.86493 | 0.00000 | 0.11164 | 0.24798 |
| tm-data-online | baseline | 57 | 0.64662 | 0.00000 | 0.00000 | 0.52530 | 0.27785 | 0.88487 | 0.00000 | 0.10383 | 0.24857 |
| tm-data-online | final | 57 | 0.64620 | -0.00041 | 0.00000 | 0.53593 | 0.27414 | 0.86972 | 0.00000 | 0.13563 | 0.25938 |

## What changed and what did not

The initial 1x1 corrections were rejected: PS presence -0.03602, PS clean-hole veto -0.03312, TM presence -0.01969 mIoU. Class-balanced integration produced too many false positives or false clean holes. At1000 updates the standalone PS local encoder has tiny-region macro AUC0.74975/AP0.05317 versus base0.52268/0.00678 on the same7 images. That ranking improvement is not a usable segmentation improvement by itself.

Spatial aggregation plus ordinary BCE reduced overcorrection: PS Dice +0.00224 and soft-IoU +0.00265 mIoU. Both recover only1/7 tiny masks; all9 masks in1–5% remain missed. This is small and validation-adaptive, not a robust breakthrough.

Tiled NRGMark adds only +0.00062 mIoU with13 improvements/9 regressions and no new tiny recovery. The bounded empty-mask scan instead recovers2 previously missed tiny TrustMark rectangles: overall0.65265→0.65705, tiny FG IoU0.08059→0.24539, misses8/9→6/9. The remaining168 masks are unchanged, including clean holes and1–5% rectangles. Both exact final crops pass90% raw known-message agreement. One wrong query per image yields0/18 acceptances over the searched candidates; this is a limited control, not a universal false-positive guarantee. Paired image-bootstrap95% interval for mean delta is[0,0.01107], reflecting only two changed cases.

Clean-hole localization remains unresolved. Do not deploy the negative one-sided veto or infer a physical detection limit from this short screen. The next matched screen continues each local encoder for500 updates:128 synthetic views versus96 synthetic+32 real outer-training patches, followed by the same600-update spatial correction. Measured PS control0.55803/real0.55937 both trail the earlier spatial0.56013. TM control0.64795 is +0.00133 over its same-fold base; TM real0.63122 is -0.01539. Real-patch adaptation is rejected in this screen.

After the full-model data screen, one production candidate was frozen: additional PS continuation plus TM empty-mask scanning; SS unchanged. See the production section for the actual receipt. Preserve#2409 and all old masks.

## Compute

- pixelseal-presence: 200.9s worker elapsed; peak PyTorch allocated 2.35GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- trustmark-presence: 223.9s worker elapsed; peak PyTorch allocated 2.35GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- pixelseal-holes: 187.3s worker elapsed; peak PyTorch allocated 2.34GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- tm-tiled: 250.8s worker elapsed; peak PyTorch allocated 2.29GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- ps-spatial-dice: 39.4s worker elapsed; peak PyTorch allocated 1.61GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- ps-spatial-iou: 39.4s worker elapsed; peak PyTorch allocated 1.60GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- tm-scan: 372.6s worker elapsed; peak PyTorch allocated 3.95GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- ps-adapt-control-spatial: 35.9s worker elapsed; peak PyTorch allocated 1.61GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- ps-adapt-real-spatial: 35.8s worker elapsed; peak PyTorch allocated 1.61GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- tm-adapt-control-spatial: 33.7s worker elapsed; peak PyTorch allocated 1.60GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- tm-adapt-real-spatial: 33.7s worker elapsed; peak PyTorch allocated 1.60GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- ps-data-small: 128.0s worker elapsed; peak PyTorch allocated 24.16GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- ps-data-online: 162.5s worker elapsed; peak PyTorch allocated 24.16GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- tm-data-small: 98.3s worker elapsed; peak PyTorch allocated 14.19GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.
- tm-data-online: 134.4s worker elapsed; peak PyTorch allocated 14.19GB. Includes data/preparation where recorded, excludes queue. This is not node elapsed.

The first three successful Slurm allocations total14m17s node time; the full-model data screen adds3m31s and production7m54s, for25m42s allocated node time across five sequential4GPU jobs (approximately1.713 allocated GPU-hours). These are allocated node durations, excluding queues and local preparation. Slurm accounting is recorded separately. Four independent workers share one node; short spatial workers finish before the longer scan workers. Do not add parallel worker durations and call that wall time.

## Protocol

New local encoder:372737parameters, analytical/gradient-checked RF27, fixed30 SRM(Y) filters plus8 constrained RGB filters, high-pass RGB, luminance and texture observability proxy; no spatial normalization. Original PS JND stays in the unchanged base. All raw pixels and local layers areFP32, TF32off. First integration usedFP16 storage for extracted deep features; later spatial screens useFP32 caches.

Exact default embedders and PNG quantization; up to512 train/128 dev source identities for new encodings. PixelSeal thin rectangles short12–32px/AR≤16; TrustMark short16–64px/AR≤2. Separate small-crop embeddings and full-source remnants/clean restorations; matched64px RGB twins, no post-embedding resize/JPEG/blur. A dedicated strict interior-only/JND-quartile transfer test has not yet been run; patch banks include boundary-straddling examples.

Local pretraining1000updates×128 views. Initial correction300updates×32views; spatial correction600updates×16views,75%synthetic replay. Base model frozen, validation never enters optimizer batches, threshold0.5 and final update fixed per screen. Later experiments were proposed after seeing previous validation results, so this is development selection. Spatial local-feature pathRF83; the full predictor also receives contextual base logits, so its total RF is not bounded83.

TrustMark proposal-only tiles:9×256 and36×160, resized1024 only for detection. Verification always original image crops and native245 decoder, raw100bits, noECC. Empty-mask scan:18 eligible images selected by baseline output,46,642 crops/image (w/h24–80step8,AR≤2,position stride16), top32 soft agreement proposals, exact refinement4/2/1px, final hard agreement≥90%. GT boxes never seed inference. GPU batched crop resize agrees with exact resize within1e-4; final verification bypasses the approximate batched path.

Metrics: official per-image mean of clean/foreground IoU, empty union=1; AP uses sklearn average_precision_score per image, absent positive classes excluded; ROC-AUC excludes single-class targets. Buckets and complete misses are saved in each completed.json; full per-image outputs are preserved. Clean-hole buckets mean clean fraction≤1%,1–5%, and all images with watermark coverage>50%.

Not run: Noiseprint++, annular features, boundary head, RF intervention, full end-to-end refit or new full CV. The research claims about guaranteed0.70+, exact conditional-median collapse, and zero-risk proposal unions are hypotheses, not established conclusions.

## Full-model fine-tuning and data use

The historical main models already used full encoder/decoder fine-tuning with real-label replay; only the new correction screens above froze those bases. A subsequent four-worker screen compares full MessageLocalizer continuation using128 fixed synthetic composites against online sampling across2500 original encoded training sources,128 extra realizations and512 thin-crop encodings. The latter two pools overlap existing source identities. No additional public source identities or embedding calls are created in this screen. This changes replay diversity and geometry mixture jointly, not only dataset size.

Both arms:150updates×16views,12synthetic+4real per update,113real training images,57validation images, same real sampling RNG, all localizer weights trainable except original JND; ResNet BN statistics frozen; encoderLR2e-6,otherLR2e-5. The existing TM rectangle verifier is preserved; the new empty-mask scan is evaluated separately. Validation0/50/150 is diagnostic, fixed endpoint150. Completed endpoints appear in the main table only after completion.

Measured PS fixed replay0.570883 (+0.013398 vs base; paired-bootstrap CI95[0.007745,0.019303]); online0.565243 (+0.007758). Both still have tiny FG IoU0; fixed replay1–5% FG IoU0.012724. Most gain is outside the smallest-region bucket. TM fixed replay0.640957 (-0.005659), online0.646204 (-0.000412). Online sampled1212 PS/1218 TM primary source identities across1800 synthetic views; fixed replay visited all128 stored compositions. The fixed arm records indices rather than source IDs, so its raw exposure counter0 is not zero training sources. A150-update screen cannot establish a data-scaling ceiling, and the online arm changes both diversity and mixture.

Bootstrap intervals resample image-level paired changes and condition on the chosen run. They do not correct for repeated validation inspection or adaptive experiment selection; unrecognized source siblings are not newly grouped in this screening analysis.

## Reproduction and sources

Run each stage inside one Slurm allocation: `python -m experiments.local_presence.job`, then `job2`, then `job3`; the full-model data screen uses `job4`. Scripts preserve completed outputs; use separate output directories for reruns. Cluster root and baseline paths are recorded in code/configs. CPU aggregation: `python -m experiments.local_presence.aggregate_scan`; report: `python experiments/local_presence/report.py`. Use task-1 cwd and PYTHONPATH=.:src. Pinned environment and unchanged base checkpoint provenance remain in message_localizer/region_data.

[SRNet](https://ws.binghamton.edu/fridrich/Research/SRNet.pdf), [NRGMark](https://collomosse.com/pubs/Agarwal-WACV-2026.pdf), [constrained-filter author code](https://github.com/LuminolT/constrained-conv-TIFS2018). Fixed-filter source commit and SHA256 are in srm_source.json. Detailed changes in PLAN.md.

## Production candidate

PixelSeal: continue original full170-data #2409 model for150 updates with the fixed replay policy selected above, then predict340 test masks. TrustMark: preserve old340 test masks; scan only38 empty masks using the unchanged validation grid/verifier. StegaStamp: preserve old340 masks exactly. No test labels, no TTA, no leaderboard hyperparameter search. Frozen recipe in results/submission/frozen_recipe.json. Implementation correction before upload: use integer correct-bit count>=90/100, because float32 mean90/100 becomes0.899999976. This preserves the declared threshold and leaves scan validation metrics unchanged (no90-bit boundary cases). One boundary case was observed during test execution; no test labels were available. The original #2409 mask is still preserved outside the new empty-mask scan. Details in implementation_correction.json.

Server receipt: {"submission_id": 2675, "status": "success", "score": 0.644, "previous_best": 0.6369, "delta": 0.007, "improved": true, "message": "New best on this task. Score shown is on the public split; the final ranking uses the held out subset, revealed after submissions close."}
