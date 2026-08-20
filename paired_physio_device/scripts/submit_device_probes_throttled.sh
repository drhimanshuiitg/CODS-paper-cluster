#!/bin/bash
# Submits the remaining device-probe combinations, respecting the cluster's
# real QOSMaxSubmitJobPerUserLimit=4 (discovered empirically this session --
# concurrent pending+running jobs, not a cumulative historical count).
# Polls squeue and tops up to 4 concurrent jobs until the whole list is submitted.
set -uo pipefail
cd /home/pkdas/IEEE_healthcomm_workshop
SBATCH_SCRIPT=paired_physio_device/scripts/run_device_probe_single.sbatch
LOG=/home/pkdas/IEEE_healthcomm_workshop/paired_physio_device/logs/submit_device_probes_throttled.log
MAX_CONCURRENT=4

# remaining combos: everything except the 4 already submitted (wavlm f0-3)
# and hubert fold0 (already complete from the smoke test)
COMBOS=(
  "wavlm 4"
  "wavlm_large 0" "wavlm_large 1" "wavlm_large 2" "wavlm_large 3" "wavlm_large 4"
  "wav2vec2 0" "wav2vec2 1" "wav2vec2 2" "wav2vec2 3" "wav2vec2 4"
  "data2vec_audio 0" "data2vec_audio 1" "data2vec_audio 2" "data2vec_audio 3" "data2vec_audio 4"
  "data2vec_spectrogram 0" "data2vec_spectrogram 1" "data2vec_spectrogram 2" "data2vec_spectrogram 3" "data2vec_spectrogram 4"
  "hear 0" "hear 1" "hear 2" "hear 3" "hear 4"
  "hubert 1" "hubert 2" "hubert 3" "hubert 4"
)

echo "$(date) starting throttled submission of ${#COMBOS[@]} remaining jobs" > "$LOG"

for combo in "${COMBOS[@]}"; do
  rep=$(echo "$combo" | cut -d' ' -f1)
  fold=$(echo "$combo" | cut -d' ' -f2)
  while true; do
    n_active=$(squeue -u "$USER" -h | wc -l)
    if [ "$n_active" -lt "$MAX_CONCURRENT" ]; then
      break
    fi
    sleep 15
  done
  out=$(sbatch --export=ALL,REPRESENTATION="$rep",FOLD="$fold" "$SBATCH_SCRIPT" 2>&1)
  echo "$(date) submit $rep fold=$fold -> $out" >> "$LOG"
done

echo "$(date) all ${#COMBOS[@]} jobs submitted; waiting for final drain" >> "$LOG"
while true; do
  n_active=$(squeue -u "$USER" -h | wc -l)
  if [ "$n_active" -eq 0 ]; then
    break
  fi
  sleep 15
done
echo "$(date) DONE -- queue drained" >> "$LOG"
