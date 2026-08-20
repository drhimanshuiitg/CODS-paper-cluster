#!/bin/bash
#SBATCH --job-name=sq_main_ready
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
# 2 concurrent fold-run workers x 4 threads/worker (matches classifiers.*.n_jobs=4
# in configs/base.yaml) = 8 threads, matching --cpus-per-task above. Fixed at 4
# regardless of the total allocation so BLAS/joblib threading isn't oversubscribed
# across workers. Part of an equal 8/8/8/8 CPU split across all 4 running jobs
# (see 04s_main_benchmark_c / 09s_dimension_control / 05s_ablation_sequential),
# saturating the account's 32-CPU QOS cap.
export PYTHONPATH="$PROJECT/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
  --representations hubert,wavlm,wav2vec2,data2vec_spectrogram \
  --classifiers random_forest,xgboost,svm_rbf,mlp --protocols R_R,S_S,R_S,S_R \
  --workers 2
