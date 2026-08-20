#!/bin/bash
#SBATCH --job-name=sq_ab_top3
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-23%6
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
CLASSIFIERS=(random_forest xgboost svm_rbf mlp)
PROTOCOLS=(R_S S_R)
task="$SLURM_ARRAY_TASK_ID"
protocol_index=$((task % 2))
task=$((task / 2))
classifier_index=$((task % 4))
rank=$((task / 4))
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_ablation_top3_task.py --rank "$rank" --classifier "${CLASSIFIERS[$classifier_index]}" --protocol "${PROTOCOLS[$protocol_index]}"
