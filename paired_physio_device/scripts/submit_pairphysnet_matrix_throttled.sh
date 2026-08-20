#!/bin/bash
# Stage 6: submits the full A1-A5 x 5-fold PairPhysNet training matrix (25
# jobs), respecting the cluster's empirically-discovered
# QOSMaxSubmitJobPerUserLimit (~4 concurrent pending+running jobs). Each job
# is genuinely multi-hour (15 epochs over a full fold) -- this driver just
# submits and throttles; it does NOT wait for full completion before
# returning control (that would be many hours). Logs progress to
# paired_physio_device/logs/submit_pairphysnet_matrix_throttled.log.
set -uo pipefail
cd /home/pkdas/IEEE_healthcomm_workshop
SBATCH_SCRIPT=paired_physio_device/scripts/run_pairphysnet_training_single.sbatch
LOG=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/submit_pairphysnet_matrix_throttled.log
LOGDIR=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs
RESULTS=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/results/event
MAX_CONCURRENT=4

VARIANTS=(A1 A2 A3 A4 A5)
FOLDS=(0 1 2 3 4)

# Idempotent restart support (2026-08-20): this driver may be restarted
# after a session hiccup while some combos are already deep into a real,
# multi-hour run. Blindly re-iterating the fixed VARIANTS x FOLDS list would
# submit a duplicate job for a combo that is already running or already
# finished. Skip a combo if either is true: (a) it already has a
# completion.json under results/event/, or (b) a currently queued/running
# job's log header already claims that exact (variant, fold).
already_done() {
  ls "$RESULTS/${1}_fold${2}_"*/completion.json >/dev/null 2>&1
}
already_running() {
  local v="$1" f="$2" jid header
  for jid in $(squeue -u "$USER" -h -o "%i" 2>/dev/null); do
    header=$(head -1 "$LOGDIR/slurm_ppn_train_${jid}.out" 2>/dev/null)
    if echo "$header" | grep -q "variant=${v} fold=${f}$"; then
      return 0
    fi
  done
  return 1
}

echo "$(date) starting/resuming throttled submission of $((${#VARIANTS[@]} * ${#FOLDS[@]})) A1-A5 training jobs" >> "$LOG"

for variant in "${VARIANTS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    if already_done "$variant" "$fold"; then
      echo "$(date) skip variant=$variant fold=$fold -> already has a completion.json" >> "$LOG"
      continue
    fi
    if already_running "$variant" "$fold"; then
      echo "$(date) skip variant=$variant fold=$fold -> already queued/running" >> "$LOG"
      continue
    fi
    while true; do
      n_active=$(squeue -u "$USER" -h | wc -l)
      if [ "$n_active" -lt "$MAX_CONCURRENT" ]; then
        break
      fi
      sleep 30
      # re-check: another combo may have finished while we waited
      if already_done "$variant" "$fold"; then
        break
      fi
    done
    if already_done "$variant" "$fold"; then
      echo "$(date) skip variant=$variant fold=$fold -> completed while waiting for a slot" >> "$LOG"
      continue
    fi
    out=$(sbatch --export=ALL,VARIANT="$variant",FOLD="$fold" "$SBATCH_SCRIPT" 2>&1)
    echo "$(date) submit variant=$variant fold=$fold -> $out" >> "$LOG"
  done
done

echo "$(date) all 25 combos accounted for (submitted, already running, or already done)." >> "$LOG"
