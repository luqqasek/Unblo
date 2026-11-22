#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --job-name=ss1000-screen
#SBATCH --output=../logs/ss1000-screen-%j.out
#SBATCH --error=../logs/ss1000-screen-%j.err
set -euo pipefail
TASK_ROOT="${SLURM_SUBMIT_DIR:?}"
EXPERIMENT_ROOT=$(dirname "$TASK_ROOT")
cd "$TASK_ROOT"
exec 9>"$EXPERIMENT_ROOT/single-node.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/src"
export HF_HOME="$EXPERIMENT_ROOT/cache/huggingface"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTORCH_KERNEL_CACHE_PATH="$EXPERIMENT_ROOT/cache/torch-kernels"
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
srun .venv-native/bin/python -u -m experiments.ss_continuation.job
