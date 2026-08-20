---
name: gpu-only-4way
description: Plan and execute this research pipeline across exactly four available GPUs with hard failure on CPU fallback for numerical/model computation. Use for feature extraction, model training, inference, GPU classifiers, folds, ablations, and result jobs.
disable-model-invocation: false
---

# Four-GPU, GPU-Only Research Execution

## Scope
Use all four GPUs efficiently while preserving reproducibility and avoiding VRAM collisions.

"GPU-only" means:
- all numerical/model computation that has a CUDA/GPU implementation must run on GPU;
- no sklearn CPU classifiers;
- no CPU model inference;
- no silent CPU tensor fallback;
- no CPU XGBoost;
- no CPU feature computation when a torch/CUDA, CuPy, cuML, RAPIDS, or equivalent GPU path is feasible.

Unavoidable OS/process orchestration, file enumeration, metadata parsing, decoding handoff, and disk I/O may use CPU because the operating system requires it. Do not count these as computational fallback.

## Hard preflight
Run and save:
`nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv`

Then verify in Python:
- CUDA is available;
- exactly 4 GPUs are visible for this pipeline;
- torch CUDA device count is 4;
- each selected job can allocate a small CUDA tensor;
- required GPU libraries import successfully.

If any numerical/model stage would fall back to CPU:
STOP and report the missing CUDA backend.

## Backend policy
Use:
- PyTorch CUDA for neural models/tensors;
- torchaudio GPU transforms where supported;
- CuPy/cuSignal for signal computations where practical;
- cuML for SVM/RF/PCA/standard GPU ML where compatible;
- XGBoost with CUDA/device=`cuda`;
- CUDA-aware mixed precision for neural training where stable.

Never silently substitute sklearn/scipy CPU implementations for a required compute stage.

## Resource discovery
Do not assume GPU model or VRAM.
Inspect each GPU and create:
`paired_physio_device/GPU_EXECUTION_PLAN.md`

Sort heavy jobs by VRAM need and place them on GPUs with sufficient free memory.

## Parallelization strategy

### Stage A — Feature extraction
Shard SUBJECTS across four GPUs, not random windows.
Example:
- GPU0: subject shard 0
- GPU1: subject shard 1
- GPU2: subject shard 2
- GPU3: subject shard 3

This avoids duplicate extraction and keeps paired R/S events together.

### Stage B — Independent experiment queue
Prefer one process per GPU.

At any moment:
- GPU0: experiment/fold job A
- GPU1: experiment/fold job B
- GPU2: experiment/fold job C
- GPU3: experiment/fold job D

When one finishes, assign the next queued job.

Do not launch all folds/variants simultaneously without a queue.

### Stage C — Proposed model variants
When feasible:
- GPU0: CE-only
- GPU1: CE + Pair
- GPU2: CE + DANN
- GPU3: Full PairPhysNet

After each completes, use the freed GPU for remaining fold/seed jobs.

### Stage D — 5-fold execution
For one model requiring independent single-GPU folds:
- run folds 0–3 concurrently;
- when first GPU finishes, run fold 4 there.

### DDP
Use multi-GPU DDP only when ONE model cannot be completed efficiently on a single GPU or the user explicitly requests multi-GPU training for that model.
Do not use DDP just because four GPUs exist; independent fold/ablation jobs usually provide better research throughput.

## Reproducibility
Every GPU job must log:
- experiment_id
- config path
- fold
- seed
- GPU id
- GPU name
- start/end timestamps
- git commit
- command
- output directory
- exit status
- peak VRAM
- runtime

Append/update:
`paired_physio_device/GPU_JOB_STATUS.csv`

## Collision prevention
Each job must have a unique:
- checkpoint directory
- log file
- result JSON
- temporary cache path if it writes cache

Never have two jobs write the same file.

## Failure policy
If CUDA OOM:
1. do not move to CPU;
2. reduce batch size;
3. enable gradient accumulation;
4. use mixed precision if numerically appropriate;
5. activate gradient checkpointing;
6. select a smaller validated backbone only if scientifically justified;
7. record the deviation.

If a GPU crashes:
- preserve completed checkpoints;
- mark job failed;
- requeue only that job.

## Monitoring
Provide a terminal-friendly status view showing:
GPU id | current experiment | fold | epoch | loss | validation metric | VRAM | elapsed | ETA

Use `nvidia-smi` plus job logs.

## Completion
Do not declare the experiment matrix complete until:
- all planned jobs have a completion record;
- failed jobs are explicitly documented;
- no result is silently missing;
- GPU_JOB_STATUS.csv matches MASTER_RESULTS.csv.
