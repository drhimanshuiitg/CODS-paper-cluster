#!/bin/bash
#SBATCH --job-name=sq_main
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-127%8
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
REPRESENTATIONS=(classical hubert wavlm wav2vec2 data2vec_audio data2vec_spectrogram data2vec_fusion full_fusion)
CLASSIFIERS=(random_forest xgboost svm_rbf mlp)
PROTOCOLS=(R_R S_S R_S S_R)
task="$SLURM_ARRAY_TASK_ID"
protocol_index=$((task % 4))
task=$((task / 4))
classifier_index=$((task % 4))
representation_index=$((task / 4))
REPRESENTATION="${REPRESENTATIONS[$representation_index]}"
CLASSIFIER="${CLASSIFIERS[$classifier_index]}"
PROTOCOL="${PROTOCOLS[$protocol_index]}"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
  --representations "$REPRESENTATION" --classifiers "$CLASSIFIER" --protocols "$PROTOCOL"
