#!/bin/bash
#SBATCH --job-name=sq_pre_feat
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
VARIANTS=(raw peak filter)
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT/.cache/numba"
export PYTHONPATH="$PROJECT/src"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/extract_features.py --feature classical --preprocessing "${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
