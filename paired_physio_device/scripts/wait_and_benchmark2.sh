#!/bin/bash
# Waits for the next currently-running training job to finish naturally,
# then (before the driver can grab that slot for the next queued combo):
# pauses the driver, runs the no-checkpointing benchmark, reports the
# result, and resumes the driver. Zero disruption to still-running jobs.
set -uo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
LOG=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/wait_and_benchmark2.log

echo "$(date) watching for the next job to finish naturally" > "$LOG"

while true; do
  n_active=$(squeue -u "$USER" -h | wc -l)
  if [ "$n_active" -lt 4 ]; then
    echo "$(date) a slot is free (n_active=$n_active)" >> "$LOG"

    echo "$(date) pausing submission driver" >> "$LOG"
    pkill -9 -f "submit_pairphysnet_matrix_throttled.sh" 2>>"$LOG" || true
    sleep 3

    echo "$(date) submitting no-checkpointing benchmark job" >> "$LOG"
    cd "$PROJECT"
    out=$(sbatch paired_physio_device/scripts/benchmark_checkpoint_vs_offload.sbatch 2>&1)
    echo "$(date) $out" >> "$LOG"
    bench_jid=$(echo "$out" | grep -oP 'Submitted batch job \K[0-9]+')

    if [ -n "$bench_jid" ]; then
      echo "$(date) waiting for benchmark job $bench_jid" >> "$LOG"
      while squeue -j "$bench_jid" -h 2>/dev/null | grep -q .; do
        sleep 10
      done
      echo "$(date) benchmark job $bench_jid finished. Output:" >> "$LOG"
      cat "$PROJECT/paired_physio_device/logs/slurm_ppn_bench_${bench_jid}.out" >> "$LOG" 2>&1
    fi

    echo "$(date) resuming submission driver" >> "$LOG"
    cd "$PROJECT" && nohup "paired_physio_device/scripts/submit_pairphysnet_matrix_throttled.sh" >> "$LOG" 2>&1 &
    echo "$(date) done -- driver resumed, this watcher exiting" >> "$LOG"
    exit 0
  fi
  sleep 20
done
