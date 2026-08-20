#!/bin/bash
#SBATCH --job-name=sq_combined
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
classifier=$("$PROJECT/.venv/bin/python" -c 'import json; print(json.load(open("/home/pkdas/IEEE_healthcomm_workshop/results/P0_statistics/selection.json"))["primary_classifier"])')
export PYTHONPATH="$PROJECT/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
  --representations classical,hubert,wavlm,wav2vec2,data2vec_audio,data2vec_spectrogram,data2vec_fusion,full_fusion \
  --classifiers "$classifier" --protocols RS_RS
