#!/bin/bash
#SBATCH --job-name=sq_phase1_base
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
  --representations classical \
  --classifiers svm_rbf \
  --protocols R_R,S_S,R_S,S_R \
  --folds 0,1,2,3,4
