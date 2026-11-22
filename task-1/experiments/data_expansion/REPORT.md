# PIPE: new photographs and realistic clean holes

This is a data preparation result, not a model-quality gain. See
[source research](RESEARCH.md), [raw audit](results/raw_audit.json),
[prepared pool](results/prepared_pool.json), and the
[training protocol](PROTOCOL.md).

The pinned803MB image/mask download passed both published SHA256 checks.
The image shard contains5483 edit pairs from5460 photographs:1414COCO and
4046Open Images. Of these4807source IDs are outside existing COCO assignments.
The separate mask shard contains24duplicated join keys; only unambiguous,
byte-identical mask variants are eligible. A64-pair raw audit found no
conflicting variants in that inspected subset.

Raw RGB/mask files are not exact provenance targets. All23inspected examples
with same-size masks have changed background pixels outside the mask. The new
generator embeds the clean original first, then copies edited pixels ONLY
inside the declared binary region. Untouched marked pixels stay exactly equal.

The author confirms torchvision stretch to512×512, not center crop. Masks are
resized before embedding using PIL bilinear and thresholded at128. No real
competition image or quantized watermarked image is geometrically transformed.
Four Open Images overlays were visually inspected and align with the removed
objects. [Author clarification](https://huggingface.co/datasets/paint-by-inpaint/PIPE_Masks/discussions/4).

The prepared smoke pool contains512train,64dev and64locked photographs,
including528Open Images and112COCO. Photos and all derivative edits are split
by source identity before embedding. No source belongs to an existing COCO
assignment. Exact decoded-pixel and pHash screens against510 competition train
images and within the new pool passed; an additional pHash screen against6500
downloaded existing photos found no candidates. This does not prove absence
of every transformed/cropped duplicate. Competition test pixels were not read.

The first strict same-size-only preparation yielded640COCO photographs; it is
preserved separately as a preparation diagnostic and is NOT the training pool.
The author-verified version uses `pipe-prepared-v2` and includes Open Images.

Running comparison: the current best PixelSeal global/carrier checkpoint,
two matched200-update adaptation arms, one with current replay and one with
half of synthetic views replaced by PIPE. It runs as shared Slurm step15905562.2
on physical GPUs2/3 of jrc0287, while the separate unlearning task uses0/1.
No second node is allocated. Exact original embedders, payloads and uint8
arithmetic are unchanged. Measured training results are recorded below.
PIPE filters small objects; retain tiny-crop/remnant replay rather than
claiming this dataset directly solves tiny PixelSeal localization.

Reproduction on JURECA:

```bash
data-expansion/.venv-audit/bin/python data-expansion/audit_pipe.py --root "$EXPERIMENT_ROOT" --inspect 64
data-expansion/.venv-audit/bin/python data-expansion/prepare_pipe_v2.py --root "$EXPERIMENT_ROOT"
data-expansion/.venv-audit/bin/python data-expansion/verify_prepared.py --root "$EXPERIMENT_ROOT"
# From task-1 after preparation and duplicate audit:
sbatch experiments/data_expansion/run.sh
```

The audit environment pins pyarrow19.0.1, numpy1.26.4, Pillow11.3.0 and
scipy1.15.3 separately from the training environment. Large photographs,
Parquet shards, encoded pixels and weights are not committed to Git.

## Measured screen

| Arm | mIoU | Delta vs initial |
|---|---:|---:|
| baseline | 0.585459 | +0.000000 |
| old_replay | 0.587910 | +0.002451 |
| pipe | 0.581785 | -0.003674 |

PIPE vs initial: `{"delta": -0.0036740340751068327, "ci95": [-0.008530018427658067, 0.0008055329767275847], "minority_guard": true, "passed": false, "improved": 14, "worse": 28}`.
PIPE vs matched old replay: `{"delta": -0.006124970220146966, "ci95": [-0.01430439591042968, 0.0007445617660600326], "minority_guard": false, "passed": false, "improved": 23, "worse": 24}`.

Passed predeclared gate: **False**. These are reused-fold developmental measurements, not full OOF or public scores.

| Group | n | Metric | Initial | Old replay | PIPE | Initial/old/PIPE misses |
|---|---:|---|---:|---:|---:|---|
| fg_le1 | 7 | fg_iou | 0.054517 | 0.066321 | 0.055115 | 3/3/2 |
| fg_1to5 | 9 | fg_iou | 0.013820 | 0.042969 | 0.023187 | 5/6/1 |
| fg_5to20 | 11 | fg_iou | 0.391923 | 0.387845 | 0.356810 | 0/0/0 |
| fg_gt20 | 30 | fg_iou | 0.881536 | 0.881736 | 0.879093 | 0/0/0 |
| clean_le1 | 4 | clean_iou | 0.000000 | 0.007384 | 0.000000 | 4/3/4 |
| clean_1to5 | 10 | clean_iou | 0.106361 | 0.082186 | 0.095544 | 5/5/4 |
| wm_over50 | 27 | clean_iou | 0.267810 | 0.265447 | 0.267494 | 9/8/8 |

Shared allocation15905562, two GPUs; encode + train + evaluate wall time5.63 minutes.200optimizer updates and3200training image views per arm.576exact embedding calls shared across arms (512train,64dev);64locked sources remain unencoded and unevaluated.

No test inference or leaderboard submission uses this failed data screen. This result does not establish that PIPE is useless: it rejects this bounded mixture/adaptation recipe. It also does not justify scaling the corpus without a new diagnosed hypothesis.
