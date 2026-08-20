#!/bin/bash
#SBATCH --job-name=sq_pre_feat
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib" NUMBA_CACHE_DIR="$PROJECT/.cache/numba" PYTHONPATH="$PROJECT/src"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
for variant in raw peak filter; do "$PROJECT/.venv/bin/python" scripts/extract_features.py --feature classical --preprocessing "$variant"; done
