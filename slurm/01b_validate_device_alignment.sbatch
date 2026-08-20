#!/bin/bash
#SBATCH --job-name=sq_alignment
#SBATCH --partition=gpu_small
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export PYTHONPATH="$PROJECT/src"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/validate_device_alignment.py
