#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=native-wmloc
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
set -euo pipefail
TASK_ROOT="${SLURM_SUBMIT_DIR:?Submit from task-1}"
EXPERIMENT_ROOT=$(dirname "$TASK_ROOT")
cd "$TASK_ROOT"
# Reserved jobs retain the shared one-node lock. User explicitly authorized
# additional unreserved jobs outside that limit; each gets its own lock.
LOCK_PATH="$EXPERIMENT_ROOT/single-node.lock"
if [ "${NATIVE_UNRESERVED_AUTHORIZED:-0}" = 1 ]; then
  LOCK_PATH="$EXPERIMENT_ROOT/unreserved-${SLURM_JOB_ID}.lock"
fi
exec 9>"$LOCK_PATH"
flock -n 9 || { echo 'Another experiment owns the one-node lock'; exit 1; }
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HOME="$EXPERIMENT_ROOT/cache/huggingface"
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/src"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TIMM_FUSED_ATTN=0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$TASK_ROOT"
if [ "$#" -eq 0 ]; then
  echo 'Pass a module and its arguments to submit_cluster.py'; exit 2
fi
srun .venv-native/bin/python -m "$@"
