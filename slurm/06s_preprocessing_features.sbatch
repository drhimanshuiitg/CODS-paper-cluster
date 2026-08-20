#!/bin/bash
#SBATCH --job-name=sq_pre_feat
#SBATCH --partition=gpu_small
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib" NUMBA_CACHE_DIR="$PROJECT/.cache/numba" PYTHONPATH="$PROJECT/src"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
for variant in raw peak filter; do "$PROJECT/.venv/bin/python" scripts/extract_features.py --feature classical --preprocessing "$variant"; done
