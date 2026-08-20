#!/bin/bash
#SBATCH --job-name=sq_phase1_feat
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
export HF_HOME="$PROJECT/checkpoints/huggingface"
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT/.cache/numba"
export PYTHONPATH="$PROJECT/src"
mkdir -p "$PROJECT/logs" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/validate_splits.py
"$PROJECT/.venv/bin/python" scripts/extract_features.py --feature classical
