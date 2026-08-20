#!/bin/bash
#SBATCH --job-name=sq_ab_top3
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export PYTHONPATH="$PROJECT/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
for rank in 0 1 2; do
  for classifier in random_forest xgboost svm_rbf mlp; do
    for protocol in R_S S_R; do
      "$PROJECT/.venv/bin/python" scripts/run_ablation_top3_task.py --rank "$rank" --classifier "$classifier" --protocol "$protocol"
    done
  done
done
