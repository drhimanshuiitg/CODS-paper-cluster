# GPU_INSTRUCTIONS.md

# IIT Guwahati GPU Cluster Instructions

## Absolute Project Path Rule

For this project, always follow this separation:

```text
CODE + RESULTS:
  /userhome/phd/h.sharma/CODS-paper

DATASET:
  /userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data
```

When creating training scripts, configuration files, SLURM scripts, logs, checkpoints, plots, CSV files, JSON summaries, Markdown reports, or any other experiment outputs, save them under:

```bash
/userhome/phd/h.sharma/CODS-paper
```

When loading training/validation/test data, use the dataset from:

```bash
/userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data
```

Do not reverse these paths. The dataset path contains spaces, so always quote it
in shell commands and SLURM scripts.

---

## GPU-Only Compute Policy (No CPU Fallback, Ever)

**Every stage of this pipeline that does real ML computation must run on GPU, and must fail loudly rather than silently compute on CPU.** This is a hardcoded, project-wide rule (2026-08-19), not a per-script preference:

- **Feature extraction** (`features.py:extract_feature_cache`) — every encoder (HuBERT, WavLM, WavLM-large, Wav2Vec2, Data2Vec-audio, Data2Vec-spectrogram, HeAR) requires GPU and raises `RuntimeError` if none is visible. The only exemptions are `classical` (handcrafted DSP features) and `odi_hb` (a per-subject SpO2 lookup) — neither involves model inference, so there is nothing to accelerate.
- **Classifier fit/predict** (`evaluation.py:build_estimator`) — `random_forest`/`svm_rbf` (cuML, hard-fails natively with no GPU), `xgboost` (`device="cuda"`, guarded by `_require_gpu()` since xgboost's own no-GPU behavior is a silent CPU fallback), `mlp` (`TorchMLPClassifier`, explicit `torch.cuda.is_available()` check at fit and predict time).
- **HeAR** (`hear_extractor/hear_worker.py`) — explicit `tf.config.list_physical_devices("GPU")` check, raises immediately if empty.
- **Any new isolated-venv/subprocess-bridge script added to this project must include the same kind of explicit, hard GPU-presence check before doing any real computation** — do not rely on a framework's own default behavior (several frameworks used here, including xgboost and early TensorFlow testing, were confirmed to silently fall back to CPU with only a warning or no message at all).
- **When shelling out via `subprocess.run(...)`, never pass a replacement `env=` dict built from scratch** — it silently strips `CUDA_VISIBLE_DEVICES` and other SLURM-injected GPU-visibility variables, which caused exactly this failure mode once already (a job ran 5+ minutes on CPU with zero error before being caught by manual inspection). Always inherit the parent environment (`{**os.environ, ...}`) and only add/override specific keys on top.
- **Deliberately, explicitly exempt** (there is no meaningful GPU equivalent, and this is not a violation of the policy): manifest/CSV/JSON parsing, audio file I/O and resampling, orchestration/control-flow/result-writing code, and PCA/CORAL's linear algebra (`sklearn.decomposition.PCA`, `scipy.linalg.eigh`) — small, one-shot-per-fold operations on already-frozen embeddings, not the computational bottleneck. If profiling ever shows otherwise, cuML has a GPU PCA (`cuml.decomposition.PCA`) that can be substituted via the same subprocess-bridge pattern.

See `CODEBASE_ARCHITECTURE_AND_EXPERIMENT_LEARNING_GUIDE.md` §26 ("GPU Flow") for the full, current account of every compute stage and its GPU-requirement status.

---


This file defines the operating rules for Codex and any automated coding agent working on the IIT Guwahati GPU Cluster.

## 1. Cluster Environment

Cluster login:

```bash
ssh <username>@gpu.iitg.ac.in
```

Current account and scheduler resources verified on 2026-08-20:

```text
Account user:      h.sharma
Login host:        clusterlogin
H100 partition:    gpu-H100
H100 node:         gpu-H100-01
H100 allocation:   1 GPU
H100 QoS:          h100
Node resources:    48 CPUs, 257675 MB configured memory
Partition limit:   15 days
```

The scheduler advertises the H100 resource, but the precise GPU model, VRAM,
driver, and CUDA compatibility must be recorded from inside an allocated job.
`nvidia-smi` is not installed on the login node. Do not treat that as evidence
that the compute node lacks a GPU.

## Project Paths

### Code and Results Workspace

All source code, scripts, configurations, logs, checkpoints, figures, tables, and experiment results for this project must be stored under:

```bash
/userhome/phd/h.sharma/CODS-paper
```

This is the **main working directory** for the project.

Codex should open, edit, create, and save project files here.

### Dataset Source

The dataset must be read from:

```bash
/userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data
```

Treat this location as the **read-only dataset/data source** for the project.

Do not move or duplicate the complete dataset into `/userhome/phd/h.sharma/CODS-paper`.

Do not duplicate large dataset files into the home directory unless explicitly instructed.

---

## 2. Critical Rule: Never Train on the Login Node

The login node is only for lightweight tasks such as:

- Editing code
- Reading files
- Inspecting logs
- Preparing SLURM scripts
- Git operations
- Checking job status
- Small configuration changes
- Lightweight validation

Never directly execute a heavy training command on the login node.

Do **not** run commands such as:

```bash
python train.py
python main.py
python experiment.py
python finetune.py
```

when they launch model training, large-scale feature extraction, or GPU-heavy processing.

All GPU workloads must be submitted through SLURM.

---

## 3. SLURM Job Submission

GPU jobs must be launched using:

```bash
sbatch gpu_job.sh
```

A typical IIT Guwahati GPU job script is:

```bash
#!/bin/bash

#SBATCH --job-name=my_experiment
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Started: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "========================================"

nvidia-smi

PROJECT="/userhome/phd/h.sharma/CODS-paper"
DATASET_ROOT="/userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data"
cd "$PROJECT"

# Activate the project environment after it has been created and validated.
source "$PROJECT/.venv/bin/activate"

python3 - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("SLURM allocated a job but PyTorch cannot see a CUDA GPU")
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM bytes:", torch.cuda.get_device_properties(0).total_memory)
PY

# Run the experiment here.
# Example:
# python train.py --config configs/experiment.yaml

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
```

For this cluster, use `gpu-H100`, `h100`, and `gpu:1`. Do not copy the old
project's `gpu_small` or `gpu:mig24gb:1` requests; those belong to a different
cluster. Choose CPU, RAM, and walltime per workload without exceeding the
verified node and partition limits above.

Inspect existing `.sh`, `.slurm`, `.sbatch`, or config files first.

---

## 4. Before Submitting a Job

Before running `sbatch`, verify:

1. Correct project directory
2. Correct Python/Conda environment
3. Correct dataset path
4. Correct output path
5. Correct GPU request
6. Correct batch size
7. Correct number of epochs
8. Correct random seed
9. Correct train/validation/test split
10. Existing results will not be overwritten
11. Required output/log directories exist
12. The command runs from `/userhome/phd/h.sharma/CODS-paper`
13. No heavy command is being executed directly on the login node

If an experiment already exists, inspect its logs and results before launching another run.

---

## 5. Job Monitoring

Useful commands:

```bash
sinfo
```

Check all jobs:

```bash
squeue
```

Check only the current user's jobs:

```bash
squeue -u $USER
```

Detailed information about a job:

```bash
scontrol show job <JOB_ID>
```

Cancel a job:

```bash
scancel <JOB_ID>
```

Monitor output:

```bash
tail -f logs/<output_file>.out
```

Monitor errors:

```bash
tail -f logs/<error_file>.err
```

---

## 6. GPU Monitoring Inside an Allocated Job

Use:

```bash
nvidia-smi
```

Useful live monitoring:

```bash
watch -n 2 nvidia-smi
```

Do not run repeated monitoring commands at unnecessarily high frequency.

During model training, prefer logging:

- Epoch
- Step
- Training loss
- Validation loss
- Learning rate
- Accuracy
- Macro-F1
- ROC-AUC when applicable
- MCC when applicable
- GPU memory usage
- Epoch duration
- Best validation score
- Current checkpoint

---

## 7. Codex Instructions

Codex may inspect, edit, and prepare files in this repository.

Codex must follow these rules:

### Allowed on Login Node

Codex may:

- Read project files
- Edit code
- Create scripts
- Inspect logs
- Inspect configuration
- Run `git status`
- Run `git diff`
- Run lightweight syntax checks
- Prepare SLURM jobs
- Check `squeue`
- Check `sinfo`

### Not Allowed on Login Node

Codex must not directly start:

- GPU training
- Large inference jobs
- Large feature extraction
- Large preprocessing pipelines
- Full dataset conversion
- Multi-hour CPU jobs

Instead, Codex should create the appropriate SLURM job script.

---

## 8. Automatic Execution Rule

Codex may automatically perform lightweight file and shell operations.

For any command that could consume substantial CPU, RAM, GPU, disk, or network resources, Codex should inspect the command first.

For GPU workloads:

```text
Never execute the training command directly.
Prepare or update the SLURM job script first.
```

Codex should not automatically submit expensive jobs unless explicitly instructed.

If instructed to submit a job, Codex must first show or verify:

- Job script
- Training command
- Dataset path
- Output directory
- GPU request

---

## 9. Dataset Safety

Never:

- Delete raw datasets
- Modify original annotations
- Overwrite raw recordings
- Use recursive destructive commands on dataset directories
- Run `rm -rf` on dataset or results folders without explicit permission

Derived files should be stored separately.

Suggested structure:

```text
project/
├── configs/
├── scripts/
├── src/
├── logs/
├── checkpoints/
├── results/
├── processed/
└── slurm/
```

The project repository and results live under:

```bash
/userhome/phd/h.sharma/CODS-paper
```

The large dataset lives under:

```bash
/userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data
```

Keep these roles separate.

---

## 10. Results Safety

Never overwrite previous experiments.

Use unique experiment directories such as:

```text
results/<experiment_name>/
```

or:

```text
results/<model>_<dataset>_<seed>/
```

Check whether a directory exists before writing into it.

Preserve:

- Configuration
- Random seed
- Dataset split
- Model parameters
- Checkpoints
- Training logs
- Test metrics
- SLURM Job ID

---

## 11. Reproducibility

Every experiment should record:

- Model name
- Dataset
- Subject/patient IDs
- Train/validation/test split
- Random seed
- Sampling rate
- Window length
- Preprocessing
- Feature extraction
- Batch size
- Learning rate
- Optimizer
- Scheduler
- Loss function
- Epoch count
- Early stopping parameters
- Best checkpoint
- Evaluation metrics
- SLURM Job ID

Prefer configuration files instead of hard-coded experiment settings.

---

## 12. Biomedical/Audio Research Rules

Where multiple windows or samples belong to the same patient/speaker:

```text
Never split windows randomly across train and test if this causes the same subject to appear in both.
```

Use patient-level / subject-level / speaker-level splitting where appropriate.

Always check for data leakage.

For imbalanced classification problems, do not rely only on accuracy.

Prefer reporting:

```text
Accuracy
Macro-F1
Weighted-F1
Precision
Recall
Balanced Accuracy / UAR
ROC-AUC
MCC
Confusion Matrix
Per-class F1
```

where applicable.

---

## 13. Python Environment Rules

Before installing packages:

1. Inspect the existing environment.
2. Check `requirements.txt`.
3. Check `environment.yml`.
4. Check `pyproject.toml`.
5. Inspect existing SLURM scripts.
6. Check the installed PyTorch/CUDA combination.

Avoid:

```bash
sudo
```

Avoid global Python package installations.

Do not upgrade PyTorch, CUDA, NumPy, Transformers, or other major packages without checking compatibility.

---

## 14. CUDA Safety

Never assume the CUDA version.

Inspect:

```bash
nvidia-smi
```

and:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Run these checks inside a short SLURM allocation on this cluster. The login node
does not currently provide `nvidia-smi`, and a login-node CUDA probe cannot
validate the H100 compute environment.

---

## 15. Storage Rules

Do not unnecessarily duplicate large datasets.

Avoid storing high-sampling-rate audio as uncompressed float32 arrays unless scientifically necessary.

Avoid:

```text
48-kHz audio -> float32 .npy
```

when a compact lossless representation is sufficient.

Use space-efficient storage for large audio datasets.

Clean temporary files only when their purpose and safety are clear.

---

## 16. Git Safety

Before major changes:

```bash
git status
```

After changes:

```bash
git diff
```

Do not automatically:

- Force push
- Hard reset
- Delete branches
- Delete untracked research outputs
- Rewrite repository history

Never commit credentials, tokens, API keys, passwords, or private cluster information.

---

## 17. Recommended Codex Workflow

When given a new research task:

### Step 1 — Inspect

Inspect:

- Repository structure
- Existing scripts
- Configurations
- Dataset loaders
- Model implementation
- Existing results
- Existing SLURM jobs

### Step 2 — Explain

Report:

- What is currently implemented
- What is missing
- Potential bugs
- Methodological risks
- Data leakage risks

### Step 3 — Modify

Make the smallest necessary changes.

Avoid unrelated refactoring.

### Step 4 — Validate

Run lightweight checks such as:

```bash
python -m py_compile <file.py>
```

or other safe validation commands.

Do not start full training during validation.

### Step 5 — Prepare SLURM

Create or update the appropriate job script.

### Step 6 — Report

Provide:

- Files changed
- Exact experiment command
- Exact `sbatch` command
- Expected log path
- Expected result path

---

## 18. Default Decision Rule

If Codex is uncertain whether a command is computationally heavy:

```text
DO NOT RUN IT ON THE LOGIN NODE.
```

Prepare a SLURM job instead.

---

## 19. Example Final Instruction to Codex

Use this when asking Codex to implement an experiment:

```text
Inspect the repository and understand the existing pipeline first.

Implement the requested experiment while preserving all existing results.

Do not run heavy computation directly on the login node.

Create or update the required SLURM job script.

Before launching anything, verify the dataset path, environment, output directory,
GPU request, train/test split, and experiment configuration.

Do not overwrite existing results.

After implementation, report:
1. files changed,
2. experiment configuration,
3. exact sbatch command,
4. expected output directory,
5. expected log files.

Do not submit the GPU job unless I explicitly ask you to submit it.
```

---

# Core Rule

> **All heavy GPU experiments must run through SLURM on `gpu-H100` with QoS `h100` and `gpu:1`. Code and results belong in `/userhome/phd/h.sharma/CODS-paper`, while the read-only dataset is loaded from `/userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data`. Never run training directly on the login node.**
