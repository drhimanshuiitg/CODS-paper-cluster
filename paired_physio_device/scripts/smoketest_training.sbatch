#!/bin/bash
#SBATCH --job-name=ppn_train_smoketest
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
"$PROJECT/.venv/bin/python" paired_physio_device/scripts/run_pairphysnet_training.py \
  --variant A1 --fold 0 --epochs 1 --batch_size 8 --n_unfrozen_layers 2
