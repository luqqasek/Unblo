#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="$PWD/.cache/pip"
export HF_HOME="$PWD/.cache/huggingface"
if [[ ! -x .venv/bin/python ]]; then python3.11 -m venv .venv; fi
if [[ "$(uname)" == Darwin ]]; then
    .venv/bin/python -m pip install -r requirements-macos.txt -e '.[dev]'
else
    .venv/bin/python -m pip install -r requirements-juelich.txt -e '.[dev]'
fi
bash scripts/download_resources.sh
.venv/bin/python download_models.py --models_dir checkpoints
.venv/bin/python -m pytest
.venv/bin/python scripts/verify_setup.py
