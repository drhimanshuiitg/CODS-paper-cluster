#!/bin/bash
#SBATCH --job-name=sq_eff_fusion
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%A_%a.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
REPRESENTATIONS=(data2vec_fusion full_fusion)
export HF_HOME="$PROJECT/checkpoints/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT/src"
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT/.cache/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" scripts/benchmark_fusion_efficiency.py --representation "${REPRESENTATIONS[$SLURM_ARRAY_TASK_ID]}"
