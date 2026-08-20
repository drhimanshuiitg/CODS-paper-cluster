#!/bin/bash
#SBATCH --job-name=sq_main_c
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=7-00:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
# 2 concurrent fold-run workers x 4 threads/worker = 8 threads, matching
# --cpus-per-task above. Fewer workers than 04p because data2vec_fusion
# (1536-d) / full_fusion (3840-d) feature arrays are larger and svm_rbf's
# cache_size=4096 (4GB kernel cache per fit) makes concurrent SVM fits the
# binding memory cost, not CPU -- --mem is bumped accordingly. Trimmed from
# 3 to 2 workers to make room for P1_dimension_control / P0_ablation.
export PYTHONPATH="$PROJECT/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py --representations full_fusion_v2,data2vec_fusion,full_fusion --classifiers random_forest,xgboost,svm_rbf,mlp --protocols R_R,S_S,R_S,S_R \
  --workers 2
