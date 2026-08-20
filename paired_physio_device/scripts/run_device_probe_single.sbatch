#!/bin/bash
#SBATCH --job-name=ppn_probe
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export PYTHONPATH="$PROJECT/src"
cd "$PROJECT"
echo "Job $SLURM_JOB_ID: representation=$REPRESENTATION fold=$FOLD"
"$PROJECT/.venv/bin/python" paired_physio_device/scripts/run_device_probe.py \
  --representation "$REPRESENTATION" --fold "$FOLD"
