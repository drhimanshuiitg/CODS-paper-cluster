#!/bin/bash
#SBATCH --job-name=ppn_preflight
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export PYTHONPATH="$PROJECT/src"
cd "$PROJECT"
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  GPU-ids: ${CUDA_VISIBLE_DEVICES:-unset}"
"$PROJECT/.venv/bin/python" paired_physio_device/scripts/gpu_preflight.py
