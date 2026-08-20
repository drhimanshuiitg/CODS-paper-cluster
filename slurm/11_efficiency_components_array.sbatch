#!/bin/bash
#SBATCH --job-name=sq_efficiency
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
FEATURES=(classical hubert wavlm wav2vec2 data2vec_audio data2vec_spectrogram)
export HF_HOME="$PROJECT/checkpoints/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT/src"
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT/.cache/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/benchmark_efficiency.py --feature "${FEATURES[$SLURM_ARRAY_TASK_ID]}"
