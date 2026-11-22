#!/usr/bin/env bash
# Reproduce E50's recorded configuration. CUDA recommended. Does not upload.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
mkdir -p outputs/eda outputs/runs
if [ -e outputs/runs/E50_feature10.pt ]; then
  echo "outputs/runs/E50_feature10.pt already exists; refusing to overwrite it." >&2
  exit 1
fi
"$PYTHON" src/eda1.py
"$PYTHON" src/probe_real.py
"$PYTHON" src/run_real.py E50_feature10 tv=2 steps=2500 bs=32 anchor_bs=32 \
  lr=6e-5 mu=-3 s=6.5 lam_delta=2 holdout=0 save=1 seed=0 alt_shift=1 \
  alt_rmix=0 class_count_weight=0 lam_anchor=5 feat_kind=cls \
  freeze_absent_head=0 lam_feat=10 anchor_bank_fraction=0
"$PYTHON" scripts/verify_submission.py outputs/runs/E50_feature10.pt
