#!/bin/bash
#SBATCH --job-name=sq_ssl_d2va
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
export HF_HOME="$PROJECT/checkpoints/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT/.cache/numba"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/extract_features.py --feature data2vec_audio
