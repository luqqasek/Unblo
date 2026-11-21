#!/bin/bash
set -euo pipefail
# Run from the isolated task-1 directory on a LOGIN node (networked setup only).
TASK_ROOT=$(pwd)
EXPERIMENT_ROOT=$(dirname "$TASK_ROOT")
export UV_INSTALL_DIR="$EXPERIMENT_ROOT/tools"
export UV_NO_MODIFY_PATH=1
export UV_PYTHON_INSTALL_DIR="$EXPERIMENT_ROOT/python"
export UV_CACHE_DIR="$EXPERIMENT_ROOT/cache/uv"
export HF_HOME="$EXPERIMENT_ROOT/cache/huggingface"
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_BUILDS=2
export UV_CONCURRENT_INSTALLS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$UV_INSTALL_DIR" "$EXPERIMENT_ROOT/logs"
if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh -o "$EXPERIMENT_ROOT/uv-install.sh"
  sh "$EXPERIMENT_ROOT/uv-install.sh"
fi
if [ ! -x .venv-native/bin/python ]; then
  "$UV_INSTALL_DIR/uv" venv --python 3.11 .venv-native
fi
"$UV_INSTALL_DIR/uv" pip install --python .venv-native/bin/python -r experiments/native_localizer/requirements-linux.txt
.venv-native/bin/python -m pip --version 2>/dev/null || true
"$UV_INSTALL_DIR/uv" pip freeze --python .venv-native/bin/python > "$EXPERIMENT_ROOT/environment.freeze.txt"
.venv-native/bin/python -c 'import torch, timm, onnxruntime; print(torch.__version__, timm.__version__, onnxruntime.get_available_providers())'
