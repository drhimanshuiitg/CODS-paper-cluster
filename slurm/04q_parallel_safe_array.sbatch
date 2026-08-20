#!/bin/bash
#SBATCH --job-name=sq_main_quick
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%A_%a.err

# Resume-safe quick-classifier work deliberately chosen to stay ahead of the
# three existing sequential benchmark workers.  No GPU is requested because
# these cached-feature classifiers are CPU-bound.  The %2 cap limits added
# pressure while gpu01 has ample CPU but little immediately free RAM.

set -euo pipefail
cd "$TMPDIR" || exit 1

PROJECT=/userhome/phd/h.sharma/CODS-paper
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$PROJECT"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    # Nearest future MLP blocks first; each is disjoint from current SVM/RF work.
    "$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
      --representations data2vec_audio,hubert,data2vec_fusion,wavlm,wav2vec2,data2vec_spectrogram,full_fusion \
      --classifiers mlp --protocols R_R,S_S,R_S,S_R
    ;;
  1)
    # The individual-representation worker will reach this representation last.
    "$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
      --representations data2vec_spectrogram \
      --classifiers random_forest,xgboost --protocols R_R,S_S,R_S,S_R
    ;;
  2)
    # The fusion worker must finish Data2Vec-fusion SVM before reaching this block.
    "$PROJECT/.venv/bin/python" scripts/run_main_benchmark.py \
      --representations full_fusion \
      --classifiers random_forest,xgboost --protocols R_R,S_S,R_S,S_R
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac
