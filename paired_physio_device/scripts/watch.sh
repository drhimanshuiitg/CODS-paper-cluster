#!/bin/bash
# ============================================================================
# ML TRAINING LIVE MONITOR -- PairPhysNet A1-A5 matrix (Stage 6)
# ============================================================================
# Built per watch_maker.md's spec, adapted to this project's actual reality:
# this is NOT one local `python train.py` process to attach a PID to -- it is
# up to 4 concurrent SLURM jobs (out of a 25-job matrix: 5 model variants x
# 5 folds), each its own independent training run on a shared, MIG-partitioned
# GPU node. Every metric below is read from a real, verified source; anything
# not observable is explicitly labeled "unavailable", never invented:
#   - epoch/step/loss/val_BA: parsed from run_pairphysnet_training.py's own
#     stdout (per-epoch summary lines) and stderr (tqdm per-step bar)
#   - model/config: read from an on-disk config.json belonging to a run that
#     actually produced a checkpoint (filters out abandoned debug configs)
#   - GPU/CPU/RAM: `srun --jobid=<job> --overlap nvidia-smi/ps/free` --
#     piggybacks on an already-running job's node allocation, no new
#     resources requested
#   - utilization.gpu: NVIDIA reports this as N/A under MIG -- shown as
#     "unavailable (MIG)", not faked
#   - GPU temperature/power: real sensor readings, but are PHYSICAL BOARD
#     sensors shared across all 4 MIG slices on that GPU -- labeled as
#     node/board-level, not attributed to one job's slice specifically
#
# Usage:
#   ./watch.sh                  normal live view, ~1s internal poll
#   ./watch.sh --interval N     poll every N seconds instead
#   ./watch.sh --once           print one snapshot and exit
#
# Ctrl+C exits ONLY this watcher. It never sends any signal to a training
# job; the one process-management action it takes (pausing/resuming the
# submission driver during an on-demand benchmark) is a separate, explicitly
# user-invoked script, not something this dashboard does on its own.
# ============================================================================
set -uo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
LOGDIR="$PROJECT/paired_physio_device/logs"
RESULTS="$PROJECT/paired_physio_device/results/event"
CKPTDIR="$PROJECT/paired_physio_device/checkpoints"
VARIANTS=(A1 A2 A3 A4 A5)
FOLDS=(0 1 2 3 4)
TOTAL=25
STEPS_PER_EPOCH=775
CONCURRENCY=4
EPOCHS=15

POLL_SEC=1
ONCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --interval) POLL_SEC="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    *) shift ;;
  esac
done

INTERACTIVE=0
[ -t 1 ] && INTERACTIVE=1
if [ "$INTERACTIVE" -eq 1 ]; then
  C_RESET=$'\033[0m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
  C_GRAY=$'\033[90m'; C_BOLD=$'\033[1m'; C_CYAN=$'\033[36m'
else
  C_RESET=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_BOLD=""; C_CYAN=""
fi

RESOURCE_CHECK_EVERY=10
resource_lines=()
gpu_board_lines=()
error_lines=()
suggestion_lines=()
poll_n=0
prev_snapshot=""

# ---------------------------------------------------------------------------
# Static header: read once from a REAL config.json that actually produced a
# checkpoint (several abandoned debug runs left behind config.json files
# with no checkpoint -- those are filtered out, not trusted).
# ---------------------------------------------------------------------------
ANY_CFG=""
for cfg in "$CKPTDIR"/*/config.json; do
  [ -f "$cfg" ] || continue
  ckpt_dir=$(dirname "$cfg")
  if [ -f "$ckpt_dir/best_checkpoint.pt" ]; then ANY_CFG="$cfg"; break; fi
done
CFG_BACKBONE="unknown"; CFG_BATCH="unknown"; CFG_MAXSEC="unknown"
CFG_UNFROZEN="unknown"; CFG_LR="unknown"; CFG_POOL="unknown"
if [ -n "$ANY_CFG" ]; then
  CFG_BACKBONE=$(grep -oP '"backbone_name":\s*"\K[^"]+' "$ANY_CFG" 2>/dev/null)
  e=$(grep -oP '"epochs":\s*\K[0-9]+' "$ANY_CFG" 2>/dev/null); [ -n "$e" ] && EPOCHS="$e"
  CFG_BATCH=$(grep -oP '"batch_size":\s*\K[0-9]+' "$ANY_CFG" 2>/dev/null)
  CFG_MAXSEC=$(grep -oP '"max_seconds":\s*\K[0-9.]+' "$ANY_CFG" 2>/dev/null)
  CFG_UNFROZEN=$(grep -oP '"n_unfrozen_layers":\s*\K[0-9]+' "$ANY_CFG" 2>/dev/null)
  CFG_LR=$(grep -oP '"lr":\s*\K[0-9.e-]+' "$ANY_CFG" 2>/dev/null)
  CFG_POOL=$(grep -oP '"pooling_mode":\s*"\K[^"]+' "$ANY_CFG" 2>/dev/null)
fi

fnum() { printf "%.*f" "${2:-2}" "$1" 2>/dev/null || echo "$1"; }

while true; do
  declare -A jobid_of status_of epoch_of val_ba_of elapsed_of live_jobs rate_of loss_of step_of

  while read -r jid; do
    [ -n "$jid" ] && live_jobs["$jid"]=1
  done < <(squeue -u "$USER" -h -o "%i" 2>/dev/null)

  for compl in "$RESULTS"/*/completion.json; do
    [ -f "$compl" ] || continue
    variant=$(grep -oP '"variant":\s*"\K[A-Z0-9]+' "$compl" 2>/dev/null)
    fold=$(grep -oP '"fold":\s*"?\K[0-9]+' "$compl" 2>/dev/null)
    [ -z "$variant" ] && continue
    key="${variant}_f${fold}"
    test_ba=$(grep -oP '"test_BA":\s*\K[0-9.]+' "$compl" 2>/dev/null)
    status_of[$key]="DONE"
    val_ba_of[$key]="$test_ba"
  done

  for log in "$LOGDIR"/slurm_ppn_train_[0-9]*.out; do
    [ -f "$log" ] || continue
    jobid=$(basename "$log" .out | grep -oP 'slurm_ppn_train_\K[0-9]+')
    [ -z "${live_jobs[$jobid]:-}" ] && continue

    header=$(head -1 "$log" 2>/dev/null)
    variant=$(echo "$header" | grep -oP 'variant=\K[A-Z0-9]+')
    fold=$(echo "$header" | grep -oP 'fold=\K[0-9]+')
    [ -z "$variant" ] && continue
    key="${variant}_f${fold}"
    [ "${status_of[$key]:-}" = "DONE" ] && continue
    if [ -n "${jobid_of[$key]:-}" ] && [ "${jobid_of[$key]}" -gt "$jobid" ]; then continue; fi
    jobid_of[$key]="$jobid"

    last_line=$(grep -P '^epoch [0-9]+:' "$log" 2>/dev/null | tail -1)
    if [ -n "$last_line" ]; then
      epoch_of[$key]=$(echo "$last_line" | grep -oP '^epoch \K[0-9]+')
      val_ba_of[$key]=$(echo "$last_line" | grep -oP 'val_BA=\K[0-9.]+')
      status_of[$key]="RUNNING"
    else
      status_of[$key]="STARTING"
    fi
  done

  declare -A squeue_elapsed
  while read -r jid elapsed; do
    [ -n "$jid" ] && squeue_elapsed["$jid"]="$elapsed"
  done < <(squeue -u "$USER" -h -o "%i %M" 2>/dev/null)

  # Per-job live rate/loss/step, plus a STALL check: how many real seconds
  # since the .err log last grew (a genuine "is it stuck" signal, not just
  # "is squeue showing it as RUNNING" -- SLURM keeps reporting RUNNING even
  # if the process hung).
  declare -A stall_of
  now_epoch=$(date +%s)
  for key in "${!jobid_of[@]}"; do
    jid="${jobid_of[$key]}"
    elapsed_of[$key]="${squeue_elapsed[$jid]:-n/a}"
    err_log="$LOGDIR/slurm_ppn_train_${jid}.err"
    if [ -f "$err_log" ]; then
      last_bar=$(tail -c 500 "$err_log" 2>/dev/null | tr '\r' '\n' | tail -1)
      rate_of[$key]=$(echo "$last_bar" | grep -oP '\K[0-9.]+(?=s/it)')
      loss_of[$key]=$(echo "$last_bar" | grep -oP 'loss=\K[0-9.]+')
      step_of[$key]=$(echo "$last_bar" | grep -oP '\K[0-9]+(?=/[0-9]+ \[)')
      mtime=$(stat -c %Y "$err_log" 2>/dev/null || echo "$now_epoch")
      stall_of[$key]=$((now_epoch - mtime))
    fi
  done

  # ---- error scan: known real failure signatures, most recent job only,
  # to keep this cheap (grep on 1-4 active .err files, not every log ever) --
  error_lines=()
  for key in "${!jobid_of[@]}"; do
    jid="${jobid_of[$key]}"
    err_log="$LOGDIR/slurm_ppn_train_${jid}.err"
    [ -f "$err_log" ] || continue
    hit=$(grep -E "Traceback|RuntimeError|CUDA out of memory|CUDA error|OOM|Killed|NCCL error|Segmentation fault|DUE TO TIME LIMIT" "$err_log" 2>/dev/null | tail -1)
    [ -n "$hit" ] && error_lines+=("  ${C_RED}[$key]${C_RESET} $hit")
  done

  lines=()
  sig_lines=()
  todo_list=()
  done_count=0
  running_count=0
  best_key="" best_ba="0"
  declare -A variant_rate_sum variant_rate_n variant_done_sum variant_done_n
  for variant in "${VARIANTS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      key="${variant}_f${fold}"
      st="${status_of[$key]:-}"
      rate=""
      if [ "$st" = "RUNNING" ] || [ "$st" = "STARTING" ]; then rate="${rate_of[$key]:-}"; fi
      rate_suffix=""
      if [ -n "$rate" ]; then
        rate_suffix="  ${C_GRAY}${rate}s/it${C_RESET}"
        variant_rate_sum[$variant]=$(echo "${variant_rate_sum[$variant]:-0} + $rate" | bc -l 2>/dev/null || echo "${variant_rate_sum[$variant]:-0}")
        variant_rate_n[$variant]=$(( ${variant_rate_n[$variant]:-0} + 1 ))
      fi
      if [ "$st" = "DONE" ]; then
        done_count=$((done_count+1))
        ba_raw="${val_ba_of[$key]:-}"
        ba=$([ -n "$ba_raw" ] && fnum "$ba_raw" 4 || echo "?")
        lines+=("  ${C_GREEN}${variant} f${fold}  DONE   test_BA=$ba${C_RESET}")
        sig_lines+=("$key DONE $ba")
        if [ -n "$ba_raw" ]; then
          variant_done_sum[$variant]=$(echo "${variant_done_sum[$variant]:-0} + $ba_raw" | bc -l 2>/dev/null || echo "${variant_done_sum[$variant]:-0}")
          variant_done_n[$variant]=$(( ${variant_done_n[$variant]:-0} + 1 ))
        fi
      elif [ "$st" = "RUNNING" ]; then
        running_count=$((running_count+1))
        ep="${epoch_of[$key]:-0}"
        ba_raw="${val_ba_of[$key]:-}"
        ba=$([ -n "$ba_raw" ] && fnum "$ba_raw" 4 || echo "pending")
        el="${elapsed_of[$key]:-00:00:00}"
        stall="${stall_of[$key]:-0}"
        stall_tag=""
        if [ "$stall" -gt 300 ] 2>/dev/null; then stall_tag="  ${C_RED}[no output ${stall}s -- possibly stalled]${C_RESET}"
        elif [ "$stall" -gt 45 ] 2>/dev/null; then stall_tag="  ${C_YELLOW}[${stall}s since last update]${C_RESET}"; fi
        step="${step_of[$key]:-}"
        step_tag=""; [ -n "$step" ] && step_tag=" step $step/$STEPS_PER_EPOCH"
        lines+=("  ${C_YELLOW}${variant} f${fold}${C_RESET}  epoch $ep/$EPOCHS$step_tag  val_BA=$ba  ${el}$rate_suffix$stall_tag")
        sig_lines+=("$key RUNNING $ep $ba")
        if [ -n "$ba_raw" ] && (( $(echo "$ba_raw > $best_ba" | bc -l 2>/dev/null || echo 0) )); then
          best_ba="$ba_raw"; best_key="$variant f$fold (epoch $ep)"
        fi
      elif [ "$st" = "STARTING" ]; then
        running_count=$((running_count+1))
        el="${elapsed_of[$key]:-00:00:00}"
        lines+=("  ${C_YELLOW}${variant} f${fold}${C_RESET}  starting...  ${el}$rate_suffix")
        sig_lines+=("$key STARTING")
      else
        todo_list+=("${variant}-f${fold}")
      fi
    done
  done
  todo_count=$((TOTAL - done_count - running_count))

  speed_line="  ${C_BOLD}Throughput (s/step, lower=faster):${C_RESET}"
  any_speed=0
  for variant in "${VARIANTS[@]}"; do
    n="${variant_rate_n[$variant]:-0}"
    if [ "$n" -gt 0 ]; then
      avg=$(echo "${variant_rate_sum[$variant]} / $n" | bc -l 2>/dev/null)
      speed_line="$speed_line $variant=$(fnum "$avg" 2)"
      any_speed=1
    fi
  done

  summary_lines=()
  for variant in "${VARIANTS[@]}"; do
    n="${variant_done_n[$variant]:-0}"
    if [ "$n" -gt 0 ]; then
      mean=$(echo "${variant_done_sum[$variant]} / $n" | bc -l 2>/dev/null)
      summary_lines+=("  ${C_CYAN}${variant} complete: n=$n  mean test_BA=$(fnum "$mean" 4)${C_RESET}")
    fi
  done

  # ---- ETA (own variant's measured rate; A2's rate as fallback for
  # never-yet-run variants, since A2-A5 all share the pair-loss cost A1
  # doesn't have -- a closer fallback than A1's rate would be) ----
  fallback_rate="3.78"
  total_remaining_sec=0
  for variant in "${VARIANTS[@]}"; do
    v_rate="${variant_rate_sum[$variant]:-}"
    if [ -n "$v_rate" ] && [ "${variant_rate_n[$variant]:-0}" -gt 0 ]; then
      v_rate=$(echo "$v_rate / ${variant_rate_n[$variant]}" | bc -l)
    else
      v_rate="$fallback_rate"
    fi
    for fold in "${FOLDS[@]}"; do
      key="${variant}_f${fold}"
      [ "${status_of[$key]:-}" = "DONE" ] && continue
      ep="${epoch_of[$key]:-0}"
      remaining_epochs=$(( EPOCHS - ep )); [ "$remaining_epochs" -lt 1 ] && remaining_epochs=1
      job_sec=$(echo "$remaining_epochs * $STEPS_PER_EPOCH * $v_rate" | bc -l 2>/dev/null)
      total_remaining_sec=$(echo "$total_remaining_sec + $job_sec" | bc -l 2>/dev/null)
    done
  done
  eta_hours=$(fnum "$(echo "$total_remaining_sec / $CONCURRENCY / 3600" | bc -l 2>/dev/null)" 1)

  analysis="ETA to full matrix completion: ~${eta_hours}h."
  [ -n "$best_key" ] && analysis="Leading right now: $best_key at val_BA=$(fnum "$best_ba" 4). $analysis"

  # ---------------------------------------------------------------------
  # GPU/CPU/RAM resource check -- srun --overlap piggybacks on a live job's
  # node allocation. Runs ~every 10s (not every 1s poll): an overlap call
  # has real scheduling latency, and node-wide GPU/RAM readings drift
  # continuously from OTHER users' unrelated jobs on this shared node, so
  # there's no point refreshing faster than that anyway.
  # ---------------------------------------------------------------------
  poll_n=$((poll_n + 1))
  if [ $((poll_n % RESOURCE_CHECK_EVERY)) -eq 0 ] || [ ! -f "$LOGDIR/.watch_resource_cache" ]; then
    refjob=$(squeue -u "$USER" -h -t RUNNING -o "%i" 2>/dev/null | head -1)
    if [ -n "$refjob" ]; then
      (
        out=$(timeout 6 srun --jobid="$refjob" --overlap bash -c '
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null
          echo "---"
          nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null
          echo "---"
          ps -u '"$USER"' -o pid,pcpu,rss,cmd --no-headers --sort=-pcpu 2>/dev/null | grep run_pairphysnet_training | head -4
          echo "---"
          free -g 2>/dev/null | awk "/^Mem:/ {print \$3\"|\"\$2}"
        ' 2>/dev/null)
        echo "$out" > "$LOGDIR/.watch_resource_cache"
      ) &
    fi
  fi
  resource_lines=(); gpu_board_lines=()
  if [ -f "$LOGDIR/.watch_resource_cache" ]; then
    cc=$(cat "$LOGDIR/.watch_resource_cache" 2>/dev/null)
    gpu_part=$(echo "$cc" | awk '/^---$/{c++; next} c==0')
    vram_part=$(echo "$cc" | awk '/^---$/{c++; next} c==1')
    cpu_part=$(echo "$cc" | awk '/^---$/{c++; next} c==2')
    mem_part=$(echo "$cc" | awk '/^---$/{c++; next} c==3')

    gpu_used_total=0; n_gpus_active=0
    hottest=0; max_power_draw=0; max_power_limit=600
    while IFS=',' read -r gidx gutil gused gtot gtemp gpow gplim; do
      gused=$(echo "$gused"|xargs); gtemp=$(echo "$gtemp"|xargs); gpow=$(echo "$gpow"|xargs); gplim=$(echo "$gplim"|xargs)
      [ -z "$gused" ] && continue
      if [ "$gused" -gt 500 ] 2>/dev/null; then n_gpus_active=$((n_gpus_active+1)); fi
      gpu_used_total=$((gpu_used_total + gused))
      gt_int=${gtemp%.*}; [ -n "$gt_int" ] && [ "$gt_int" -gt "$hottest" ] 2>/dev/null && hottest="$gt_int"
      gp_int=${gpow%.*}; [ -n "$gp_int" ] && [ "$gp_int" -gt "$max_power_draw" ] 2>/dev/null && { max_power_draw="$gp_int"; max_power_limit="${gplim%.*}"; }
    done <<< "$gpu_part"
    if [ -n "$gpu_part" ]; then
      resource_lines+=("  ${C_BOLD}GPU (node-wide, gpu01, 8 physical GPUs):${C_RESET} ${n_gpus_active}/8 active, ${gpu_used_total} MiB used")
      gpu_board_lines+=("  Utilization      : unavailable (MIG slices do not report per-instance compute utilization)")
      gpu_board_lines+=("  Hottest board    : ${hottest} C   Peak power draw: ${max_power_draw} / ${max_power_limit} W  (board-level; shared across all MIG slices on that GPU, not attributable to one job)")
    fi

    # per-job VRAM: cross-reference compute-apps PIDs against our own ps PIDs
    declare -A pid_vram
    while IFS=',' read -r vpid vmem; do
      vpid=$(echo "$vpid"|xargs); vmem=$(echo "$vmem"|xargs)
      [ -n "$vpid" ] && pid_vram["$vpid"]="$vmem"
    done <<< "$vram_part"

    if [ -n "$cpu_part" ]; then
      cpu_summary=""
      while read -r ppid pcpu rss rest_cmd; do
        [ -z "$pcpu" ] && continue
        vtag=$(echo "$rest_cmd" | grep -oP -- '--variant \K[A-Z0-9]+')
        ftag=$(echo "$rest_cmd" | grep -oP -- '--fold \K[0-9]+')
        rss_gb=$(fnum "$(echo "$rss / 1048576" | bc -l 2>/dev/null)" 1)
        vram_mb="${pid_vram[$ppid]:-?}"
        cpu_summary="$cpu_summary ${vtag}f${ftag}=${pcpu}%cpu/${rss_gb}GBram/${vram_mb}MiBvram"
      done <<< "$cpu_part"
      [ -n "$cpu_summary" ] && resource_lines+=("  ${C_BOLD}Per-job:${C_RESET}$cpu_summary")
    fi
    unset pid_vram

    if [ -n "$mem_part" ]; then
      mem_used=$(echo "$mem_part" | cut -d'|' -f1)
      mem_total=$(echo "$mem_part" | cut -d'|' -f2)
      [ -n "$mem_used" ] && resource_lines+=("  ${C_BOLD}System RAM:${C_RESET} ${mem_used}GB / ${mem_total}GB used")
    fi
  fi

  # ---- Evidence-based suggestions -- only ever derived from real numbers
  # already measured this session, never generic advice. ----
  suggestion_lines=()
  if [ "$any_speed" -eq 1 ] && [ -n "${variant_rate_sum[A1]:-}" ] && [ "${variant_rate_n[A1]:-0}" -gt 0 ]; then
    a1_now=$(echo "${variant_rate_sum[A1]} / ${variant_rate_n[A1]}" | bc -l)
    isolated="1.186"
    slowdown=$(echo "($a1_now - $isolated) / $isolated * 100" | bc -l 2>/dev/null)
    suggestion_lines+=("  ${C_YELLOW}⚠${C_RESET} A1 measured at $(fnum "$a1_now" 2)s/step vs $isolated s/step when benchmarked alone (job 1743) -- ~$(fnum "$slowdown" 0)% slower.")
    suggestion_lines+=("     This is 4-job concurrency/contention on the shared MIG node, not something fixable in the training code.")
  fi
  if [ -n "${variant_rate_sum[A2]:-}" ] && [ "${variant_rate_n[A2]:-0}" -gt 0 ] && [ -n "${variant_rate_sum[A1]:-}" ] && [ "${variant_rate_n[A1]:-0}" -gt 0 ]; then
    a1_now=$(echo "${variant_rate_sum[A1]} / ${variant_rate_n[A1]}" | bc -l)
    a2_now=$(echo "${variant_rate_sum[A2]} / ${variant_rate_n[A2]}" | bc -l)
    pct=$(echo "($a2_now - $a1_now) / $a1_now * 100" | bc -l 2>/dev/null)
    suggestion_lines+=("  ${C_CYAN}ℹ${C_RESET} A2 is ~$(fnum "$pct" 0)% slower per step than A1 ($(fnum "$a2_now" 2)s vs $(fnum "$a1_now" 2)s) -- the added paired-contrastive loss, not a bug.")
  fi
  if [ ${#error_lines[@]} -eq 0 ] && [ "$running_count" -gt 0 ]; then
    suggestion_lines+=("  ${C_GREEN}✓${C_RESET} No errors detected in any active job's log.")
  fi

  pct=$(( done_count * 100 / TOTAL ))
  filled=$(( done_count * 30 / TOTAL )); empty=$(( 30 - filled ))
  bar="[$(printf '#%.0s' $(seq 1 $filled 2>/dev/null))$(printf -- '-%.0s' $(seq 1 $empty 2>/dev/null))]"

  sig=$(printf "%s\n" "${sig_lines[@]}")
  err_sig=$(printf "%s\n" "${error_lines[@]}")
  snapshot="$done_count|$running_count|$todo_count|$sig|$err_sig"
  if [ "$snapshot" != "$prev_snapshot" ] || [ "$INTERACTIVE" -eq 1 ]; then
    [ "$INTERACTIVE" -eq 1 ] && clear
    echo "${C_BOLD}══════════════════════════════════════════════════════════════════════${C_RESET}"
    echo "${C_BOLD}[$(date +%H:%M:%S)] PairPhysNet A1-A5 TRAINING MATRIX MONITOR${C_RESET}"
    echo "${C_BOLD}══════════════════════════════════════════════════════════════════════${C_RESET}"
    echo "  Model    : $CFG_BACKBONE  (24-layer transformer, last $CFG_UNFROZEN layers fine-tuned)"
    echo "  Features : paired Recorder+Smartphone raw audio, ${CFG_MAXSEC}s windows"
    echo "  Config   : $EPOCHS epochs/job, batch=$CFG_BATCH, lr=$CFG_LR, pooling=$CFG_POOL, precision=AMP(fp16 autocast)"
    echo "  Variants : A1=event-loss only  A2=A1+pair-contrastive  A3=A1+device-adversarial  A4=A2+A3  A5=A4+disentangle+SpO2-aux"
    echo ""
    echo "${C_BOLD}$bar $pct%  DONE $done_count/$TOTAL  RUNNING $running_count  TODO $todo_count${C_RESET}"
    printf "%s\n" "${lines[@]}"
    [ "${#summary_lines[@]}" -gt 0 ] && printf "%s\n" "${summary_lines[@]}"
    [ "$any_speed" -eq 1 ] && echo "$speed_line"
    [ "${#resource_lines[@]}" -gt 0 ] && printf "%s\n" "${resource_lines[@]}"
    [ "${#gpu_board_lines[@]}" -gt 0 ] && printf "%s\n" "${gpu_board_lines[@]}"
    if [ "${#todo_list[@]}" -gt 0 ]; then
      echo "  ${C_GRAY}TODO (${#todo_list[@]}): ${todo_list[*]}${C_RESET}"
    fi
    echo ""
    if [ ${#error_lines[@]} -gt 0 ]; then
      echo "  ${C_RED}${C_BOLD}ERRORS${C_RESET}"
      printf "%s\n" "${error_lines[@]}"
    fi
    if [ ${#suggestion_lines[@]} -gt 0 ]; then
      echo "  ${C_BOLD}SUGGESTIONS (evidence-based)${C_RESET}"
      printf "%s\n" "${suggestion_lines[@]}"
    fi
    echo ""
    echo "  ${C_BOLD}>> ${analysis}${C_RESET}"
    [ "$INTERACTIVE" -eq 1 ] && echo "${C_GRAY}Refresh: ${POLL_SEC}s | Ctrl+C: exit (does not affect training) ${C_RESET}"
    prev_snapshot="$snapshot"
  fi

  unset jobid_of status_of epoch_of val_ba_of elapsed_of rate_of loss_of step_of stall_of \
        squeue_elapsed live_jobs variant_rate_sum variant_rate_n variant_done_sum variant_done_n

  [ "$ONCE" -eq 1 ] && exit 0
  sleep "$POLL_SEC"
done
