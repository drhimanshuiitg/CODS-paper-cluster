#!/bin/bash
#SBATCH --job-name=sq_eff_all
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
export HF_HOME="$PROJECT/checkpoints/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT/src" MPLCONFIGDIR="$PROJECT/.cache/matplotlib" NUMBA_CACHE_DIR="$PROJECT/.cache/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
for feature in classical hubert wavlm wav2vec2 data2vec_audio data2vec_spectrogram; do "$PROJECT/.venv/bin/python" scripts/benchmark_efficiency.py --feature "$feature"; done
for representation in data2vec_fusion full_fusion; do "$PROJECT/.venv/bin/python" scripts/benchmark_fusion_efficiency.py --representation "$representation"; done
