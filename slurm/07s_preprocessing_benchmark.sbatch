#!/bin/bash
#SBATCH --job-name=sq_pre_eval
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
# Sized down to fit the account's QOSMaxCpuPerUserLimit alongside the other
# concurrently running jobs (see slurm/04p_*, slurm/04s_main_benchmark_c.sbatch).
export PYTHONPATH="$PROJECT/src" OMP_NUM_THREADS=2
cd "$PROJECT"
for variant in raw peak filter peak_filter; do
  "$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py --results-root "$PROJECT/results/P1_preprocessing" --representations classical --classifiers svm_rbf --protocols R_R,S_S,R_S,S_R --preprocessing "$variant" --workers 1
done
