# Native-resolution localizer

Execution status and limitations: [STATUS.md](STATUS.md). This directory is separate
from all previous methods. A prepared script is not a completed experiment.

**Completed short pilot:** [REPORT.md](REPORT.md). StegaStamp validation mIoU
**0.69496 on all170 real images**, with no real-label training/selection, fixed
threshold0.5. Synthetic dev0.71418.200 updates x batch64; main pilot18.1minutes.
PixelSeal/TrustMark later ran the same pilot and underperformed; see `../native_hybrid_submission/REPORT.md`. New extractor-initialized PS/TM results and current submission are in `../RESEARCH_HANDOFF_2026-11-22.md`. The full original large-budget recipe remains unexecuted.

For a fresh bounded run on the cluster, from task-1:

```bash
.venv-native/bin/python -m experiments.native_localizer.submit_cluster --development --minutes 30 --gpus 4 -- experiments.native_localizer.pilot --root /p/project1/training2666/pszenny1/native-localizer --gpus 4 --run-name pilot-next
```

Existing runs are protected from overwriting. The same predeclared raw last
checkpoint can be assessed on the remaining real folds without further training:

```bash
.venv-native/bin/python -m experiments.native_localizer.submit_cluster --development --minutes 5 --gpus 1 -- experiments.native_localizer.real_eval --checkpoint ../runs/stegastamp/pilot-next/last.pt --data data --run ../runs/stegastamp/pilot-next --folds 2 3
```

The development partition uses no cispahack reservation. The user additionally
authorized ordinary unreserved queue jobs outside the reserved one-node limit;
the completed pilot used one node at a time throughout.

Run modules from `task-1` with `PYTHONPATH=.:src` and the independent Python environment.

## Reproduce the first stages

```bash
bash experiments/native_localizer/bootstrap_cluster.sh
python -m experiments.native_localizer.download_coco --out ../sources/coco2017/bulk
python -m experiments.native_localizer.prepare_sources --out ../sources/coco2017 --images-directory ../sources/coco2017/bulk/train2017 --download-limit 50000
python -m experiments.native_localizer.submit_cluster --minutes 30 --gpus 4 -- experiments.native_localizer.debug_run --root /p/project1/training2666/pszenny1/native-localizer --family stegastamp --gpus 4
```

The downloader fixes 50,000 training / 2,000 dev / 2,000 locked **source IDs** before
generation. All donors/backgrounds must come from that same source partition.
It checks all 510 training-image byte hashes and DCT pHash candidates; it never reads
competition test pixels. This is a heuristic duplicate screen, not proof against
every strongly transformed sibling. The locked source suite is not evaluated yet.

The first debug job creates only 64 train and 32 dev exact encoded realizations,
checks bit conventions/residuals, profiles FP32 and deeper AMP and performs 200
fixed-batch FP32 updates. Its warmup is shortened ONLY to diagnose capacity.
Neither overfit scores nor random-RGB profiling loss show generalization.

## Full synthetic-stage command

Generate and validate a 2,500-realization shard first, using distinct train sources:

```bash
python -m experiments.native_localizer.generator --sources ../sources/coco2017/train.jsonl --partition train --method stegastamp --count 2500 --models checkpoints --out ../encoded/stegastamp/debug-train
python -m experiments.native_localizer.generator --sources ../sources/coco2017/dev.jsonl --partition dev --method stegastamp --count 256 --models checkpoints --out ../encoded/stegastamp/debug-dev
python -m experiments.native_localizer.train --family stegastamp --train ../encoded/stegastamp/debug-train/encoded.jsonl --dev ../encoded/stegastamp/debug-dev/encoded.jsonl --backbone checkpoints/convnext_base.fb_in22k/model.safetensors --out ../runs/stegastamp/seed42 --stop-after 1000
```

Generation and training commands above must run **inside a Slurm allocation**.
Use `submit_cluster` for a single module or `debug_run` for the initial sequential
diagnostic. Do not run them directly on login nodes. Downloads and source indexing
run on login nodes with bounded workers/threads.

The initial generator implements **same-content debug composites only**. It does
not yet implement/validate the full cross-image and LaMa mixture. Do not scale this
debug distribution to the final 50,000-update experiment and call it the full recipe.

Resume `train` with `--resume <out>/last.pt`, retaining the same config. `--stop-after`
limits this invocation without altering the 50k-step warmup/cosine schedule. Training
is FP32 until a documented fidelity decision enables `--amp-validated`. Multiple
GPUs via `--gpus 4` remain within a single node. No second queued/running job is
submitted while an existing user job is present.

## Provenance and outputs

- `pretrained.json`: pinned ImageNet-22K repository/revision/SHA256/normalization.
- `encoded.jsonl`: exact call count, source ID/hash, message, crop, R, embedding
  regime, quantization, perturbation norms and paths to original PNG clean/marked bases.
- `view_recipes.jsonl`: every optimizer update's source/realization IDs, recipe seed,
  pair relationships and realized areas. Masks can be reproduced procedurally.
- `train.jsonl`: loss components, gradient norms, view counts, proportions, timing.
- `last.pt` / `best.pt`: model, EMA, AdamW, scheduler, Python/NumPy/Torch/CUDA RNG.
- `dev/<update>`: official score, both-class IoU, per-image AP/AUC with undefined
  counts, size/regime strata, movement pairs, constant/shuffle controls, maps/gallery.

Final nested real adaptation, controls, OOF/seed confirmation and submission have
not yet run. Prior ignored checkpoints, probability maps and split JSON files are
not included in the Git handoff; do not invent matched baseline results.

`infer` requires a frozen recipe, reads original 512 RGB, selects the family model
directly, uses no STN/TTA, preserves name ordering and safely reloads the saved
binary uint8 submission to validate it under 20 MB. It does not upload anything.
