#!/bin/bash
#SBATCH --job-name=ppn_bench
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export PYTHONPATH="$PROJECT/src"
export HF_HOME="/scratch/pkdas/IEEE_healthcomm_workshop/cache/huggingface"
export PYTHONUNBUFFERED=1
cd "$PROJECT"
"$PROJECT/.venv/bin/python" -u paired_physio_device/scripts/benchmark_checkpoint_vs_offload.py
