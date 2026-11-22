#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --job-name=pipe-screen
#SBATCH --output=../logs/pipe-screen-%j.out
#SBATCH --error=../logs/pipe-screen-%j.err
set -euo pipefail
TASK_ROOT="${SLURM_SUBMIT_DIR:?}"
EXPERIMENT_ROOT=$(dirname "$TASK_ROOT")
cd "$TASK_ROOT"
exec 9>"$EXPERIMENT_ROOT/single-node.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/src"
export HF_HOME="$EXPERIMENT_ROOT/cache/huggingface"
export PYTHONUNBUFFERED=1
export PYTORCH_KERNEL_CACHE_PATH="$EXPERIMENT_ROOT/cache/torch-kernels"
srun .venv-native/bin/python -u -m experiments.data_expansion.job
