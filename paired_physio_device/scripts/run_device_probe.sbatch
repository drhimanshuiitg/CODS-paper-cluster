#!/bin/bash
#SBATCH --job-name=ppn_device_probe
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-34%4
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%A_%a.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/slurm_%x_%A_%a.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
REPRESENTATIONS=(hubert wavlm wavlm_large wav2vec2 data2vec_audio data2vec_spectrogram hear)
FOLDS=(0 1 2 3 4)

task="$SLURM_ARRAY_TASK_ID"
fold_index=$((task % 5))
rep_index=$((task / 5))
representation="${REPRESENTATIONS[$rep_index]}"
fold="${FOLDS[$fold_index]}"

export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
echo "Job $SLURM_JOB_ID task $task: representation=$representation fold=$fold GPU=${CUDA_VISIBLE_DEVICES:-unset}"
"$PROJECT/.venv/bin/python" paired_physio_device/scripts/run_device_probe.py \
  --representation "$representation" --fold "$fold"
