#!/bin/bash
#SBATCH --job-name=sq_ablation
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
REPRESENTATIONS=(full_minus_hubert full_minus_wavlm full_minus_wav2vec2 full_minus_data2vec_audio full_minus_data2vec_spectrogram full_minus_data2vec)
PROTOCOLS=(R_S S_R)
task="$SLURM_ARRAY_TASK_ID"
protocol_index=$((task % 2))
representation_index=$((task / 2))
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
  --results-root "$PROJECT/results/P0_ablation" \
  --representations "${REPRESENTATIONS[$representation_index]}" \
  --classifiers svm_rbf --protocols "${PROTOCOLS[$protocol_index]}"
