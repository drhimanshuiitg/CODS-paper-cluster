#!/bin/bash
#SBATCH --job-name=sq_pca
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
# run_dimension_control.py handles one (protocol, fold) combo per invocation
# with no internal --workers flag, so parallelism here is at the shell level:
# 2 concurrent combos (xargs -P 2) x 4 threads/combo (OMP/MKL) = 8, matching
# --cpus-per-task. Each combo's own PCA+SVM fit is the expensive, mostly-
# sequential part (~130 min observed), so this trades a *few* combos running
# genuinely faster (more BLAS threads) for real cross-combo concurrency,
# rather than 1 combo at a time as before.
export PYTHONPATH="$PROJECT/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$PROJECT"
# full_fusion_v2 (hubert+wavlm+wav2vec2+data2vec_audio, dropping
# data2vec_spectrogram per the error/branch-level analysis) runs alongside
# the original full_fusion -- same protocol/fold grid, doubled to 20 combos.
for representation in full_fusion_v2 full_fusion; do
  for protocol in R_S S_R; do for fold in 0 1 2 3 4; do printf '%s %s %s\n' "$representation" "$protocol" "$fold"; done; done
done | xargs -P 2 -n 3 bash -c '"$0/.venv/bin/python" "$0/scripts/run_dimension_control.py" --representation "$1" --protocol "$2" --fold "$3"' "$PROJECT"
