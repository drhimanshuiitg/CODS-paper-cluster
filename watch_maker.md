Inspect my **current machine-learning project** and create or replace `watch.sh` with a professional **real-time ML training monitor**.

The purpose of this script is to let me run:

```bash
./watch.sh
```

and immediately understand:

- which model is currently training
- which Python/script/module is running
- current epoch
- total epochs
- current batch/step
- total batches/steps
- training loss
- validation loss, if available
- learning rate
- elapsed training time
- estimated time remaining
- estimated epoch completion time
- GPU utilization
- GPU memory utilization
- GPU temperature
- GPU power usage, if available
- CPU utilization
- RAM utilization
- system/process memory usage
- whether GPU/CPU/RAM is being used efficiently
- latest errors/warnings
- whether training appears stalled
- actionable suggestions for making training faster based on **real measured statistics**

## FIRST: INSPECT THE PROJECT

Before writing anything, inspect the repository.

Determine:

1. ML framework:
   - PyTorch
   - TensorFlow/Keras
   - JAX
   - Hugging Face Trainer
   - Lightning
   - Accelerate
   - DeepSpeed
   - custom training loop
   - or something else

2. Find the actual training entrypoint.

Examples:

```text
train.py
main.py
run_training.py
trainer.py
src/train.py
scripts/train.sh
```

3. Determine how training is launched.

For example:

```bash
python train.py
torchrun ...
accelerate launch ...
deepspeed ...
python -m ...
```

4. Identify:
   - model name/config
   - number of epochs
   - batch size
   - learning rate
   - optimizer
   - scheduler
   - gradient accumulation
   - precision: FP32 / FP16 / BF16
   - number of GPUs
   - dataloader workers
   - dataset size
   - checkpoint configuration

5. Search for existing:
   - `watch.sh`
   - monitoring scripts
   - log files
   - TensorBoard logs
   - WandB usage
   - training logs
   - JSON/JSONL metrics
   - checkpoints
   - stdout redirection
   - progress bars such as tqdm

## OLD watch.sh

If `watch.sh` already exists:

1. Read it completely.
2. Understand what useful information it provides.
3. Keep the useful project-specific logic.
4. Remove obsolete/noisy/broken behavior.
5. Replace it with the new implementation.

Do not leave:

```text
watch_old.sh
watch_backup.sh
watch2.sh
```

or similar junk unless there is a genuine reason.

There should ultimately be one intended:

```text
watch.sh
```

---

# LIVE DASHBOARD

Create a clean terminal dashboard that refreshes approximately every 1–2 seconds without continuously filling terminal scrollback.

A layout similar to this is desirable:

```text
══════════════════════════════════════════════════════════════════════
                     ML TRAINING LIVE MONITOR
══════════════════════════════════════════════════════════════════════

PROCESS
  Status          : RUNNING
  PID             : 48291
  Script          : train.py
  Model           : <actual model>
  Framework       : PyTorch
  Device          : cuda:0
  Precision       : BF16

TRAINING
  Epoch           : 7 / 40              17.5%
  Step            : 1,842 / 8,400       21.9%
  Train Loss      : 0.3842
  Validation Loss : 0.4171
  Learning Rate   : 2.35e-5
  Batch Size      : 32
  Grad Accum      : 4

TIME
  Elapsed         : 01:42:17
  Current Epoch   : 00:14:21
  Avg Step        : 183 ms
  ETA Epoch       : 00:51:12
  ETA Training    : 06:37:44
  Expected Finish : 22:49

GPU 0
  Utilization     : 94%
  VRAM            : 19.8 / 24.0 GB   82%
  Temperature     : 71 C
  Power           : 284 / 350 W
  GPU Process     : python (PID 48291)

CPU
  Total           : 63%
  Training Proc   : 412%
  Cores           : 16
  Load Average    : 7.2

MEMORY
  RAM             : 38.1 / 64 GB     60%
  Training Proc   : 21.4 GB
  Swap            : 0.0 / 8 GB

EFFICIENCY
  GPU Efficiency  : EXCELLENT
  CPU Pressure    : NORMAL
  RAM Pressure    : NORMAL
  Data Pipeline   : HEALTHY

LATEST EVENT
  Epoch 7 | Step 1842 | loss=0.3842 | lr=2.35e-5

PERFORMANCE SUGGESTIONS
  ✓ GPU utilization is consistently >90%.
  ✓ VRAM has ~4 GB available.
  → Consider increasing batch size slightly if convergence permits.
  → CPU does not currently appear to bottleneck GPU feeding.

ERRORS
  No recent errors detected.

══════════════════════════════════════════════════════════════════════
Refresh: 1s | Ctrl+C: Exit
══════════════════════════════════════════════════════════════════════
```

This is only an example.

Adapt the actual dashboard to the project and machine.

---

# MODEL / PROCESS DETECTION

Automatically determine which ML training process is running.

Do not simply display every Python process.

Identify likely training processes based on:

- repository directory
- command line
- training entrypoint
- PID
- GPU process information
- project files

Show the full useful command without making the dashboard unreadable.

Example:

```text
PID     : 38291
Command : python train.py --model llama --epochs 40
```

If distributed training is running using:

```text
torchrun
accelerate
deepspeed
```

recognize that correctly.

Show:

```text
Workers : 4
GPUs    : 4
Main PID: ...
```

when appropriate.

---

# MODEL INFORMATION

Determine the actual model being trained when possible.

Examples:

```text
Model: resnet50
Model: bert-base-uncased
Model: Llama-3...
Model: CustomTransformer
```

Use:

- CLI arguments
- configuration files
- training logs
- Python source
- Hugging Face config
- checkpoint metadata

Do not fabricate a model name when it cannot be determined.

Use:

```text
Model: unknown
```

instead.

---

# TRAINING METRICS

Display live values when available:

```text
Epoch
Total epochs

Step
Total steps

Training loss
Validation loss

Learning rate

Batch size
Gradient accumulation

Samples/sec
Batches/sec
Tokens/sec
```

Tokens/sec is particularly useful for transformer/LLM training.

If the project exposes additional important metrics such as:

```text
accuracy
F1
perplexity
gradient norm
reward
KL
eval loss
```

show the relevant ones.

Do not overload the dashboard with irrelevant metrics.

---

# IMPORTANT: DO NOT INVENT METRICS

`watch.sh` cannot magically read Python variables that only exist in process memory.

Determine where live training information actually exists.

Prefer, in order:

1. structured JSON/JSONL training metrics
2. existing training logs
3. TensorBoard event information when practical
4. WandB local files when practical
5. Hugging Face trainer state
6. stdout/stderr logs
7. checkpoint metadata
8. process command arguments

If epoch, learning rate, loss, or step are not currently externally observable, clearly identify that.

Do NOT fake values.

---

# MINIMAL TRAINING INSTRUMENTATION IF NECESSARY

If the project currently provides no reliable machine-readable training metrics, you may make a **small and safe modification** to the training code so it writes a lightweight runtime status file.

For example:

```text
.tmp/training_status.json
```

or another appropriate ignored/runtime directory.

Possible structure:

```json
{
  "timestamp": 1234567890,
  "epoch": 7,
  "epochs": 40,
  "step": 1842,
  "total_steps": 8400,
  "train_loss": 0.3842,
  "val_loss": 0.4171,
  "learning_rate": 0.0000235,
  "batch_size": 32,
  "samples_per_sec": 912.4
}
```

Prefer an existing logging mechanism if one already exists.

Do not heavily refactor the training code merely for monitoring.

Any instrumentation must have negligible performance overhead.

Use atomic writes if appropriate so `watch.sh` never reads a half-written file.

Ensure runtime monitoring files are ignored by Git.

---

# ETA

Calculate useful ETA values from **actual observed progress**.

Show:

```text
Elapsed
Average step time
Current epoch elapsed time
ETA current epoch
ETA whole training
Expected completion time
```

Do not calculate ETA from a single noisy step.

Use a moving average or recent step/epoch timing where possible.

If enough information is unavailable:

```text
ETA: calculating...
```

is better than an incorrect ETA.

Detect pauses/stalls so ETA does not become absurd.

---

# GPU MONITORING

For NVIDIA systems, use `nvidia-smi` where available.

Collect relevant values such as:

```text
utilization.gpu
utilization.memory
memory.used
memory.total
temperature.gpu
power.draw
power.limit
clocks
GPU process PID
```

Support multiple GPUs.

For distributed training show each GPU separately.

Example:

```text
GPU 0 | util 96% | VRAM 22.1/24 GB | 73C | 310W
GPU 1 | util 94% | VRAM 21.8/24 GB | 72C | 305W
GPU 2 | util 97% | VRAM 22.0/24 GB | 74C | 312W
GPU 3 | util 95% | VRAM 21.9/24 GB | 73C | 309W
```

If NVIDIA is unavailable, investigate whether the machine uses:

- AMD / ROCm
- Apple Silicon
- CPU-only training

and degrade gracefully.

---

# GPU EFFICIENCY

Do NOT judge efficiency based on one sample.

Maintain a short rolling history, for example the most recent 20–60 seconds.

Calculate:

```text
Current GPU utilization
Average GPU utilization
Peak GPU utilization
VRAM utilization
```

Give a simple classification such as:

```text
EXCELLENT
GOOD
MODERATE
LOW
```

based on actual rolling statistics.

For example, approximate interpretations may be:

```text
>90% average GPU compute utilization = excellent
75–90% = good
50–75% = moderate
<50% = investigate
```

But use judgment and account for cases such as evaluation/checkpointing.

---

# CPU MONITORING

Show:

```text
Total CPU utilization
Training-process CPU utilization
Core count
Load average
```

Remember that process CPU can exceed 100% on multicore Linux.

For example:

```text
Training CPU: 620%
```

can mean approximately 6.2 cores.

Do not incorrectly call it an error.

---

# RAM MONITORING

Show:

```text
Used RAM
Total RAM
RAM percentage
Training-process RSS
Swap usage
```

Example:

```text
RAM  : 41.2 / 64 GB (64%)
Proc : 24.8 GB
Swap : 0 / 8 GB
```

---

# OPTIONAL SYSTEM SIGNALS

If available cheaply, monitor useful additional signals:

```text
Disk usage
Disk read/write activity
GPU PCIe utilization
CPU load
I/O wait
```

Only include these when they actually help diagnose training performance.

Do not turn the dashboard into a generic `htop` clone.

---

# ERRORS

Show recent errors prominently.

Detect things such as:

```text
Traceback
RuntimeError
CUDA out of memory
CUDA error
NaN
Inf
Killed
OOM
DataLoader worker exited
NCCL error
Segmentation fault
checkpoint failure
```

Show the most relevant recent error lines.

Example:

```text
ERROR
CUDA out of memory.
Tried to allocate 2.00 GiB...
```

Do not dump hundreds of traceback lines into the main dashboard.

Provide a way to inspect the full log if necessary.

---

# STALL DETECTION

Detect when training appears stuck.

Examples:

```text
Last step updated 2 seconds ago   → healthy
Last step updated 45 seconds ago  → warning
Last step updated 5 minutes ago   → possibly stalled
```

But account for legitimate long:

```text
evaluation
checkpoint saving
data preprocessing
validation
```

when detectable.

---

# REAL PERFORMANCE SUGGESTIONS

This is especially important.

The bottom of the dashboard should provide **dynamic recommendations based on measured system statistics**.

Never print generic recommendations merely because they sound useful.

Examples:

### Case 1: GPU starving

If over a meaningful period:

```text
GPU utilization = 30–50%
CPU utilization = high
GPU VRAM = moderate
```

suggest something like:

```text
⚠ GPU appears underfed.

Possible data-loader bottleneck.

Try:
- increasing num_workers
- enabling persistent_workers
- enabling pin_memory for CUDA
- profiling data augmentation
```

Only make these suggestions if appropriate for the detected framework/project.

---

### Case 2: Low GPU + low CPU

If:

```text
GPU = 35%
CPU = 20%
```

investigate possibilities such as:

```text
small batch size
synchronization
excessive logging
frequent validation
disk/network input waits
Python-side bottleneck
```

Give the most plausible project-specific suggestion.

---

### Case 3: GPU memory available

If:

```text
GPU util = high
VRAM = only 40–60%
```

then potentially suggest:

```text
VRAM headroom available.

If convergence and model behavior permit, benchmark a larger batch size.
```

Do not state that a larger batch size is automatically better.

---

### Case 4: GPU OOM risk

If:

```text
VRAM > 95%
```

warn:

```text
⚠ VRAM pressure is very high.
Little safety margin remains for temporary allocations.
```

Possible project-appropriate options:

```text
gradient accumulation
smaller batch
activation checkpointing
mixed precision
```

---

### Case 5: CPU bottleneck

If CPU is saturated while GPU utilization regularly drops:

```text
Possible CPU/data-pipeline bottleneck.
```

Use actual measurements.

---

### Case 6: RAM pressure

If RAM exceeds approximately 90% or swap activity becomes significant:

```text
⚠ Host memory pressure detected.
```

Investigate dataloader worker count, dataset caching, preprocessing, etc.

---

### Case 7: mixed precision

If supported hardware is being used but project is clearly training FP32, mention:

```text
Mixed precision may improve throughput on this GPU.
Benchmark BF16/FP16 if numerically appropriate.
```

Do not automatically change precision.

---

### Case 8: DataLoader

For PyTorch, inspect:

```python
num_workers
pin_memory
persistent_workers
prefetch_factor
```

If observed metrics suggest the GPU is waiting on data, provide recommendations based on the actual configuration.

---

### Case 9: compilation

If appropriate for the detected framework/version/model, and the observed workload could benefit, mention options such as:

```python
torch.compile(...)
```

but do NOT modify training behavior automatically.

---

# PERFORMANCE ADVICE MUST BE EVIDENCE-BASED

Each recommendation should ideally mention the evidence.

Good:

```text
⚠ GPU feeding may be slow:
  avg GPU utilization: 54%
  avg CPU utilization: 91%
  num_workers: 2

Suggestion:
  Benchmark num_workers=4 or 8.
```

Bad:

```text
Try increasing num_workers.
```

with no supporting measurements.

Another good example:

```text
✓ Compute pipeline looks healthy:
  avg GPU utilization: 96%
  VRAM utilization: 83%
  CPU utilization: 58%

No obvious hardware bottleneck detected.
```

Sometimes **no recommendation** is the correct recommendation.

---

# DO NOT AUTO-OPTIMIZE MY TRAINING

The watcher should analyze and recommend.

It must NOT automatically change:

```text
batch size
learning rate
precision
workers
model
optimizer
training parameters
```

because these can affect training behavior.

Display recommendations only.

---

# TRAINING QUALITY WARNINGS

Where the available metrics make it reasonable, detect simple anomalies such as:

```text
loss became NaN
loss became Inf
loss suddenly exploded
learning rate unexpectedly zero
no improvement for a long period
```

Do not make strong claims about convergence from a few observations.

Example:

```text
⚠ Loss increased sharply:
  previous rolling avg: 0.42
  current: 1.83
```

---

# REFRESH BEHAVIOR

Refresh approximately once every:

```text
1 second
```

or use an interval that has negligible impact on training.

Monitoring must not materially reduce training performance.

Avoid expensive commands every second if a cheaper approach is available.

Some expensive metrics may be refreshed less frequently.

---

# TERMINAL UX

Use readable sections and colors when the terminal supports them:

```text
green  = healthy
yellow = warning
red    = error
cyan   = informational
```

But the script must remain understandable without color.

Handle narrow terminals gracefully where reasonable.

Avoid flickering as much as practical.

Do not endlessly append complete dashboards.

---

# PROCESS SAFETY

`watch.sh` is primarily a MONITOR.

It should attach to and observe an already-running training process.

It should NOT terminate my training process when I press:

```text
Ctrl+C
```

Ctrl+C should exit only the watcher unless the watcher itself explicitly launched a process.

Never use broad commands like:

```bash
pkill python
killall python
```

---

# OPTIONAL ARGUMENTS

If useful, support simple options such as:

```bash
./watch.sh
./watch.sh --interval 2
./watch.sh --pid 12345
./watch.sh --log path/to/train.log
./watch.sh --once
```

Do not over-engineer argument parsing.

`./watch.sh` with no arguments should be the normal path.

---

# DEPENDENCIES

Prefer tools already installed on a normal Linux ML machine:

```text
bash
ps
awk
sed
grep
tail
date
nvidia-smi
/proc
```

If an optional tool such as:

```text
jq
bc
```

improves the implementation, either:

- provide a fallback, or
- clearly mark it as optional.

Do not require a large monitoring package simply to implement `watch.sh`.

---

# PERFORMANCE HISTORY

Keep a lightweight in-memory or temporary rolling history so the watcher can compare:

```text
GPU utilization over time
CPU utilization over time
RAM utilization
step duration
samples/sec
tokens/sec
loss
```

This history should allow recommendations to be based on trends instead of one instant measurement.

Clean up watcher-owned temporary data on exit.

Do not delete project/training data.

---

# TRAINING SPEED

Where enough information is available, prominently report actual throughput.

For example:

```text
Step time     : 182 ms
Steps/sec     : 5.49
Samples/sec   : 703
Tokens/sec    : 28,410
```

For LLM training, prioritize:

```text
tokens/sec
tokens/sec/GPU
```

when those values can be reliably calculated.

For image models, samples/sec may be more meaningful.

Choose metrics appropriate to this project.

---

# COMPARATIVE SPEED

After enough samples have been gathered, calculate a rolling comparison.

Example:

```text
Current throughput : 714 samples/sec
5 min average      : 681 samples/sec
Change             : +4.8%
```

This helps identify slowdowns.

Do not claim a statistically meaningful improvement from tiny measurement windows.

---

# MULTI-GPU TRAINING

If multiple GPUs are involved, display:

```text
GPU utilization per GPU
VRAM per GPU
temperature per GPU
power per GPU
```

Also detect major imbalance.

Example:

```text
⚠ GPU imbalance detected

GPU 0 average: 97%
GPU 1 average: 96%
GPU 2 average: 43%
GPU 3 average: 95%
```

Mention that GPU 2 may be waiting, misconfigured, or receiving uneven work.

For DDP/NCCL workloads, provide only evidence-based suggestions.

---

# FINAL DASHBOARD PRIORITIES

The most important information should be visible without scrolling:

1. training RUNNING/STOPPED/ERROR status
2. model
3. script
4. epoch
5. step
6. loss
7. learning rate
8. elapsed time
9. ETA
10. training throughput
11. GPU utilization
12. VRAM usage
13. CPU utilization
14. RAM utilization
15. latest error
16. bottleneck assessment
17. performance suggestions

---

# VALIDATION

After implementing:

1. Run:

```bash
bash -n watch.sh
```

2. Make it executable:

```bash
chmod +x watch.sh
```

3. Test it against the currently running training process if one exists.

4. Verify that actual PID/model/script detection works.

5. Verify GPU information against `nvidia-smi`.

6. Verify CPU/RAM values against the OS.

7. Verify epoch/step/loss/LR values against the project's actual training logs.

8. Confirm ETA changes reasonably as training progresses.

9. Confirm pressing Ctrl+C exits the watcher without killing training.

10. Confirm the monitor itself has negligible resource usage.

11. Confirm errors are detected correctly.

12. Confirm recommendations change according to real measurements rather than being permanently hard-coded.

---

# IMPORTANT IMPLEMENTATION PRINCIPLE

Do not build a pretty dashboard full of fake information.

I prefer:

```text
Learning rate: unavailable
```

over an invented learning rate.

I prefer:

```text
ETA: calculating...
```

over a fake ETA.

I prefer:

```text
No obvious bottleneck detected.
```

over generic optimization advice.

The dashboard must represent the **actual current state of my training job**.

Inspect the project thoroughly, implement the best solution for its actual framework and logging architecture, reuse valuable pieces of the old `watch.sh`, remove obsolete watcher logic, test the final implementation, and then briefly explain what you changed.