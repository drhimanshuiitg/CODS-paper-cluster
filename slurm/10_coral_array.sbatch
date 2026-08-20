#!/bin/bash
#SBATCH --job-name=sq_coral
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-29%2
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
REPRESENTATIONS=(BEST_SINGLE_PLACEHOLDER data2vec_fusion full_fusion)
PROTOCOLS=(R_S S_R)
task="$SLURM_ARRAY_TASK_ID"
fold=$((task % 5))
task=$((task / 5))
protocol_index=$((task % 2))
representation_index=$((task / 2))
representation="${REPRESENTATIONS[$representation_index]}"
if [[ "$representation" == "BEST_SINGLE_PLACEHOLDER" ]]; then
  representation=$("$PROJECT/.venv/bin/python" -c 'import json; print(json.load(open("/home/pkdas/IEEE_healthcomm_workshop/results/P0_statistics/selection.json"))["best_single_encoder"])')
fi
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_coral.py --representation "$representation" --protocol "${PROTOCOLS[$protocol_index]}" --fold "$fold"
