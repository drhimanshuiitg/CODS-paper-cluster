#!/bin/bash
#SBATCH --job-name=sq_pca
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
PROTOCOLS=(R_S S_R)
task="$SLURM_ARRAY_TASK_ID"
fold=$((task % 5))
protocol_index=$((task / 5))
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/run_dimension_control.py --protocol "${PROTOCOLS[$protocol_index]}" --fold "$fold"
