#!/bin/bash
#SBATCH --job-name=sq_coral
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
# results/P0_statistics/selection.json doesn't exist yet (P0_statistics hasn't
# run -- it needs P0_device_gap fully complete first, which it isn't: full_fusion/
# xgboost, full_fusion_v2/svm_rbf, full_fusion_v2/mlp still have real gaps despite
# show_progress.py reporting 98%+, inflated by duplicate completions from earlier
# job resubmissions). Starting with data2vec_fusion + full_fusion only, which don't
# need that selection; best_single_encoder can be added as a follow-up sbatch once
# P0_statistics produces selection.json.
# 2 concurrent combos (xargs -P 2) x 4 threads/combo = 8, matching --cpus-per-task,
# same sizing convention as 09s_dimension_control.sbatch.
export PYTHONPATH="$PROJECT/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$PROJECT"
for representation in data2vec_fusion full_fusion; do
  for protocol in R_S S_R; do for fold in 0 1 2 3 4; do printf '%s %s %s\n' "$representation" "$protocol" "$fold"; done; done
done | xargs -P 2 -n 3 bash -c '"$0/.venv/bin/python" "$0/scripts/run_coral.py" --representation "$1" --protocol "$2" --fold "$3"' "$PROJECT"
