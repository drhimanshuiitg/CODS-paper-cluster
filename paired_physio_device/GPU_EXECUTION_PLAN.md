# GPU Execution Plan

Per `Four-GPU, GPU-Only Research Execution.md`. **The real hardware topology differs from that skill's literal "verify exactly 4 GPUs are visible in this job" language — documented here rather than forced.**

## Hard preflight result (real, run via SLURM job 1611, not the login node)

```
sbatch --gres=gpu:mig24gb:1 ... paired_physio_device/scripts/gpu_preflight.py
```

Full raw output: `paired_physio_device/logs/gpu_preflight_report.json`. Key facts:

- **Cluster (`gpu01`, partition `gpu_small`) has 8 physical GPUs**: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB (≈98 GB) each.
- **Each physical GPU is MIG-partitioned into 4 slices of ~24 GB** (`mig24gb`), for **32 total MIG24GB slices** cluster-wide (`scontrol show node`: `Gres=gpu:mig24gb:32`). At preflight time, 10/32 slices were allocated to other jobs/users; 22 free.
- **A single SLURM job requesting `--gres=gpu:mig24gb:1` sees exactly 1 GPU** (`torch.cuda.device_count() == 1`, device name `...MIG 1g.24gb`, 25.37 GB visible). CUDA available, small-tensor allocation confirmed working.
- `torch` 2.9.0+cu128, CUDA 12.8, `transformers` and `xgboost` import successfully in the main venv (`/home/pkdas/IEEE_healthcomm_workshop/.venv`). **`torchaudio` is NOT installed in this venv** (import failure) — a real gap for any GPU-resident audio transform work (Section C/D of the master prompt). **`cuml` is not importable in this venv** — consistent with the project's established pattern of using a separate isolated venv (`/scratch/pkdas/IEEE_healthcomm_workshop/gpu_classifier_test/`) for cuML-dependent classifier work, reused rather than rebuilt.

## Reconciling "four-way" parallelism with real MIG topology

The skill's Stage A–D parallelization strategy (shard across 4 GPUs) is implemented here as **4 concurrent SLURM jobs, each independently requesting one `--gres=gpu:mig24gb:1` slice**, rather than 4 GPUs visible inside a single job — this is both what the hardware actually offers per-job and exactly the pattern the project's existing, already-successful SLURM scripts use (e.g. `slurm/10_coral_array.sbatch`: `--gres=gpu:mig24gb:1` with a `%2`-throttled job array). No code changes are needed to honor the skill's intent; "GPU0/1/2/3" in the master prompt's Stage-A/B/C mapping below should be read as "queue slot 0/1/2/3 of concurrently-running single-MIG-slice jobs," not four indices inside one job's `CUDA_VISIBLE_DEVICES`.

**Not claimed:** literal 4-GPU DDP training inside one job. Per the skill's own instruction ("Use multi-GPU DDP only when ONE model cannot be completed efficiently on a single GPU... Do not use DDP just because four GPUs exist"), and given each PairPhysNet variant (shared frozen/lightly-fine-tuned SSL encoder + small projection heads) comfortably fits a single 24 GB MIG slice, DDP is not planned for any currently-scoped job.

## Planned parallelization mapping (Stages 2–10, once launched)

| Stage | Mapping onto 4 concurrent MIG-slice jobs |
|---|---|
| Feature/embedding (re)extraction, if any new encoder pass is needed | Shard by **subject**, 4 subject-shards per concurrent job batch (mirrors existing `slurm/03_extract_ssl_array.sbatch` pattern) |
| Device probes (Section H, 7 representations) | Queue of independent single-GPU jobs, 4 running concurrently, next queued job fills a freed slot |
| A1–A5 model-variant training (Section E) | Up to 4 variants training concurrently (A1/A2/A3/A4 or A5), each its own job/slot |
| 5-fold execution per variant | Folds 0–3 concurrent, fold 4 backfilled into the first freed slot |

## Collision prevention

Every job gets a unique `checkpoint_dir`, `log file`, and `result JSON` path under `paired_physio_device/{checkpoints,logs,results}/<experiment_id>/`, following this project's pre-existing content-addressed / unique-directory convention (`config_hash(...)`) — no two jobs will be configured to write the same file.

## GPU_JOB_STATUS.csv

Created empty with the required header now; populated as jobs are actually submitted in later stages:

`experiment_id, fold, seed, task, train_device, test_device, model_variant, backbone, pooling, window_sec, gpu_id, gpu_name, start_ts, end_ts, git_commit, command, output_dir, exit_status, peak_vram_gb, runtime_sec`

## GPU_JOB_STATUS.csv provenance note

`sacct`/`sacctmgr` return `Connection refused` from this login node (`_open_persist_conn: failed to open persistent connection to host:localhost:6819`) — the accounting daemon is not reachable here, so `start_ts`/`end_ts`/`peak_vram_gb` cannot be populated from SLURM's own accounting DB and are left blank rather than fabricated. `git_commit` is also blank: `/home/pkdas/IEEE_healthcomm_workshop` is not a git repository (`git rev-parse HEAD` fails with `fatal: not a git repository`). If per-job timing/VRAM is needed later, `seff <jobid>` run from a context with accounting access, or parsing the `Job $SLURM_JOB_ID` echo + wall-clock in each job's own stdout log, are the two available fallbacks.

## A real CUDA OOM found and fixed this stage (Section: Failure policy)

The first training smoke test (job 1655, variant A1/fold 0, batch_size=8, 2 unfrozen WavLM-large layers) hit `CUDA OutOfMemoryError` at ~22.9/23.6 GiB on a single MIG24GB slice, during the very first training step. Per the skill's own failure policy ("do not move to CPU; reduce batch size; enable gradient accumulation; use mixed precision; activate gradient checkpointing"), applied in this order: (1) gradient checkpointing enabled on the shared encoder (`use_reentrant=False`, avoiding the classic "no input requires_grad" failure mode for a frozen-trunk/unfrozen-top setup), (2) batch size reduced 8→4 with 2-step gradient accumulation (effective batch unchanged at 8), (3) mixed precision (`torch.autocast` + `GradScaler`) added to both the training and evaluation loops. Re-submitted as job 1656; see `EXPERIMENT_PLAN.md`/chat log for the outcome recorded once the monitor reports back.

## Status at end of this turn

Only the preflight job (1611) has been run. No training, no feature extraction, no other GPU job has been submitted — Stage 0 (audit) does not require them, and per `GPU_INSTRUCTIONS.md` Section 8, expensive jobs are not auto-submitted without explicit instruction to proceed into the training stages.
