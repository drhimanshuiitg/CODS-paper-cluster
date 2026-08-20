#!/bin/bash
#SBATCH --job-name=sq_ablation
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
# 4 concurrent fold-run workers x 2 threads/worker = 8, part of the equal
# 8/8/8/8 CPU split across all 4 running jobs, saturating the account's
# 32-CPU QOS cap. 60 combos here are all independent svm_rbf fits on large
# (~3000-4600d) leave-one-encoder-out fusion vectors -- more workers matters
# far more than more threads/fit for this embarrassingly-parallel workload.
# --mem bumped for 4 concurrent svm_rbf cache_size=4096 fits.
export PYTHONPATH="$PROJECT/src" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py --results-root "$PROJECT/results/P0_ablation" \
  --representations full_minus_hubert,full_minus_wavlm,full_minus_wav2vec2,full_minus_data2vec_audio,full_minus_data2vec_spectrogram,full_minus_data2vec \
  --classifiers svm_rbf --protocols R_S,S_R --workers 4
