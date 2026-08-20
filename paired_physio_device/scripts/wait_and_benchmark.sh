#!/bin/bash
# Waits for one of the 4 currently-running A1 jobs (1694-1697) to finish,
# then: pauses the submission driver (so it doesn't grab the freed slot for
# A2), runs the checkpointing-vs-offload benchmark on that freed slot,
# reports the result, and resumes the driver.
set -uo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
LOG=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/wait_and_benchmark.log
WATCH_JOBS=(1694 1695 1696 1697)

echo "$(date) watching for one of ${WATCH_JOBS[*]} to finish" > "$LOG"

while true; do
  for jid in "${WATCH_JOBS[@]}"; do
    if ! squeue -j "$jid" -h 2>/dev/null | grep -q .; then
      echo "$(date) job $jid left the queue -- a slot is free" >> "$LOG"

      echo "$(date) pausing submission driver so it doesn't grab this slot" >> "$LOG"
      pkill -9 -f "submit_pairphysnet_matrix_throttled.sh" 2>>"$LOG" || true
      sleep 3

      echo "$(date) submitting benchmark job" >> "$LOG"
      cd "$PROJECT"
      out=$(sbatch paired_physio_device/scripts/benchmark_checkpoint_vs_offload.sbatch 2>&1)
      echo "$(date) $out" >> "$LOG"
      bench_jid=$(echo "$out" | grep -oP 'Submitted batch job \K[0-9]+')

      if [ -n "$bench_jid" ]; then
        echo "$(date) waiting for benchmark job $bench_jid to complete" >> "$LOG"
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
  done
  sleep 20
done
