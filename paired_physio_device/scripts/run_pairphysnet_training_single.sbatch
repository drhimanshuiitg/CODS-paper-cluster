#!/bin/bash
# PREPARED, NOT YET SUBMITTED for the real A1-A5 matrix (Stage 6).
# Usage: sbatch --export=ALL,VARIANT=A2,FOLD=0 run_pairphysnet_training_single.sbatch
#SBATCH --job-name=ppn_train
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
export PYTHONPATH="$PROJECT/src"
export HF_HOME="/scratch/pkdas/IEEE_healthcomm_workshop/cache/huggingface"
# Fix (2026-08-20): stdout is fully block-buffered when redirected to a file
# (not a TTY), so without this, SLURM's .out log does not update until the
# buffer fills or the process exits -- "live" monitoring silently sees a
# frozen file for many minutes even though training is genuinely progressing.
export PYTHONUNBUFFERED=1
cd "$PROJECT"
echo "Job $SLURM_JOB_ID: variant=$VARIANT fold=$FOLD"
# batch_size=8 + n_unfrozen_layers=2 (grad_accum_steps left at its script
# default of 2, same as the smoke test) is the EXACT configuration validated
# OOM-safe in job 1656 (ran 9m42s with gradient checkpointing + AMP, well
# past where job 1655 crashed within seconds at these same settings minus
# the OOM fixes). n_unfrozen_layers=4 (this script's original value) was
# NEVER actually tested and is not used here -- Section: Failure policy
# "record the deviation."
"$PROJECT/.venv/bin/python" -u paired_physio_device/scripts/run_pairphysnet_training.py \
  --variant "$VARIANT" --fold "$FOLD" --epochs 15 --batch_size 8 --n_unfrozen_layers 2
