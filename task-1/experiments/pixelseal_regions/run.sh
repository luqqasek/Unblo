#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --job-name=ps-native-region
#SBATCH --output=../logs/ps-native-region-%j.out
#SBATCH --error=../logs/ps-native-region-%j.err
set -euo pipefail
TASK_ROOT="${SLURM_SUBMIT_DIR:?}"
EXPERIMENT_ROOT=$(dirname "$TASK_ROOT")
cd "$TASK_ROOT"
exec 9>"$EXPERIMENT_ROOT/single-node.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/src"
export HF_HOME="$EXPERIMENT_ROOT/cache/huggingface"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv
srun .venv-native/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 -m experiments.pixelseal_regions.train --root "$EXPERIMENT_ROOT" --steps 800 --max-train-seconds 1080
