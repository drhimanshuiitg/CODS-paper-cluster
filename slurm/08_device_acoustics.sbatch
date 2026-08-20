#!/bin/bash
#SBATCH --job-name=sq_acoustics
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
export PYTHONPATH="$PROJECT/src"
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT/.cache/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/analyze_device_acoustics.py --workers "$SLURM_CPUS_PER_TASK"
