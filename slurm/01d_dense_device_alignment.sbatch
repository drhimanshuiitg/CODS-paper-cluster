#!/bin/bash
#SBATCH --job-name=sq_align_dense
#SBATCH --partition=gpu_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export PYTHONPATH="$PROJECT/src"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/estimate_device_alignment_dense.py
