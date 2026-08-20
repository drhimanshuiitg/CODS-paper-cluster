#!/bin/bash
#SBATCH --job-name=sq_pre_eval
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
VARIANTS=(raw peak filter peak_filter)
PROTOCOLS=(R_R S_S R_S S_R)
task="$SLURM_ARRAY_TASK_ID"
protocol_index=$((task % 4))
variant_index=$((task / 4))
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
  --results-root "$PROJECT/results/P1_preprocessing" \
  --representations classical --classifiers svm_rbf \
  --protocols "${PROTOCOLS[$protocol_index]}" --preprocessing "${VARIANTS[$variant_index]}"
