#!/bin/bash
#SBATCH --job-name=ppn_unittest
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export PYTHONPATH="$PROJECT/src"
export HF_HOME="/scratch/pkdas/IEEE_healthcomm_workshop/cache/huggingface"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" paired_physio_device/scripts/test_pairphysnet_forward.py
