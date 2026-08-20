#!/bin/bash
#SBATCH --job-name=sq_ssl_feat
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
FEATURES=(hubert wavlm wav2vec2 data2vec_audio data2vec_spectrogram)
FEATURE="${FEATURES[$SLURM_ARRAY_TASK_ID]}"
export HF_HOME="$PROJECT/checkpoints/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT/.cache/numba"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$PROJECT/logs" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/extract_features.py --feature "$FEATURE"
