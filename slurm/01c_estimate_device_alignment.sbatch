#!/bin/bash
#SBATCH --job-name=sq_align_fit
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
export PYTHONPATH="$PROJECT/src"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/estimate_device_alignment.py
