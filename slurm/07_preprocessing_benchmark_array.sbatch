#!/bin/bash
#SBATCH --job-name=sq_pre_eval
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
VARIANTS=(raw peak filter peak_filter)
PROTOCOLS=(R_R S_S R_S S_R)
task="$SLURM_ARRAY_TASK_ID"
protocol_index=$((task % 4))
variant_index=$((task / 4))
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
  --results-root "$PROJECT/results/P1_preprocessing" \
  --representations classical --classifiers svm_rbf \
  --protocols "${PROTOCOLS[$protocol_index]}" --preprocessing "${VARIANTS[$variant_index]}"
