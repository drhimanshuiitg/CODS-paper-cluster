# PairPhysNet — Fresh-Machine Setup Guide

This repo contains the full code, configs, and documentation for the PairPhysNet
paired Recorder/Smartphone sleep-apnea research pipeline (Stage 6: A1–A5
ablation matrix training). This guide gets a **new machine** (different GPU,
different cluster) running the exact same pipeline from scratch.

## 0. What is and isn't in this repo

**Included** (small, version-controlled):
- All code: `paired_physio_device/scripts/`, `paired_physio_device/models/`,
  `src/sleep_quadnet/`, `slurm/`, `scripts/`
- All configs: `configs/base.yaml`, `paired_physio_device/configs/`
- All documentation: `paired_physio_device/AUDIT_REPORT.md`,
  `DATA_INTEGRITY_REPORT.md`, `EXPERIMENT_PLAN.md`, `GPU_EXECUTION_PLAN.md`,
  `STAGE_LOG.md`, `README_SETUP.md` (this file), `GPU_INSTRUCTIONS.md`,
  `CODEBASE_ARCHITECTURE_AND_EXPERIMENT_LEARNING_GUIDE.md`
- `metadata/` — the exact manifest and fold-assignment CSVs used (so a new
  machine reproduces the *identical* subject-disjoint splits, not new ones)
- `ara/` — the reverse-engineered knowledge base (claims, evidence, findings)
- `Q1_Paper_Artifact/` — the prior-stage manuscript/figures/tables

**Excluded** (regenerate on the new machine — see below):
- The raw dataset (Tao et al. 2025, ~53GB audio + PSG annotations) — not
  redistributable from this repo; acquire separately (§2)
- `cached_features/`, `checkpoints/`, `paired_physio_device/checkpoints/`,
  `.venv/` — all large, all regenerable
- `results/*/runs/`, `logs/`, `paired_physio_device/logs/` — raw run
  artifacts, regenerable by re-running

## 1. THE ONE THING YOU MUST CHANGE FIRST

**Every script in this project uses hardcoded absolute paths**, by this
project's own established convention (see `GPU_INSTRUCTIONS.md`):

```
CODE + RESULTS:  /home/pkdas/IEEE_healthcomm_workshop
DATASET:         /scratch/pkdas/IEEE_healthcomm_workshop
```

On a new machine, either:
- **(a) Recommended, fastest)** Recreate the same absolute paths on the new
  machine (`/home/<you>/IEEE_healthcomm_workshop`,
  `/scratch/<you>/IEEE_healthcomm_workshop` — or literally reuse
  `/home/pkdas/...` if you have that username there too), OR
- **(b)** Find-and-replace every occurrence of `/home/pkdas/IEEE_healthcomm_workshop`
  and `/scratch/pkdas/IEEE_healthcomm_workshop` across the repo with your new
  machine's real paths:
  ```bash
  grep -rl "/home/pkdas/IEEE_healthcomm_workshop\|/scratch/pkdas/IEEE_healthcomm_workshop" \
    paired_physio_device/ scripts/ src/ slurm/ configs/ | \
    xargs sed -i \
      -e 's#/home/pkdas/IEEE_healthcomm_workshop#<YOUR_CODE_PATH>#g' \
      -e 's#/scratch/pkdas/IEEE_healthcomm_workshop#<YOUR_DATASET_PATH>#g'
  ```
  Do this **before** running anything. This was never abstracted into an env
  var because the whole pipeline was built for one specific cluster
  (IIT Guwahati, SLURM, MIG-partitioned GPUs) — it is not currently a
  general-purpose portable framework.

## 2. Get the dataset

The raw dataset is Tao et al. 2025's public dual-device (Recorder +
Smartphone) sleep-apnea dataset — not included in this repo (too large, and
not this project's to redistribute). Acquire it separately and place it at:
```
<YOUR_DATASET_PATH>/dataset/V5/Data/{01..50}/
```
matching the structure: `{sid}_annotation.json`, `{sid}_phone.wav`,
`{sid}_recorder_1.wav`, `{sid}_recorder_2.wav`, `{sid}_sleep_stage.csv`,
`{sid}_SpO2.csv`, `{sid}_HR.csv` per subject. See
`paired_physio_device/AUDIT_REPORT.md` §1–2 for the exact confirmed format
(8kHz mono 16-bit PCM WAV, etc.) if you need to verify your copy matches.

The `metadata/*.csv` files in this repo (manifest, fold assignments,
alignment models) were built from this exact dataset and should NOT need to
be regenerated if your copy is identical — they will not match if your
dataset differs.

## 3. Environment

```bash
python3 -m venv .venv          # or wherever <YOUR_CODE_PATH>/.venv should live
source .venv/bin/activate
pip install -r requirements.txt
```

Check `requirements.txt` and `paired_physio_device/scripts/*.py` imports for
the full dependency list (`torch`, `transformers`, `tqdm`, `scikit-learn`,
`numpy`, `scipy`, `librosa`, `pyyaml`). GPU-accelerated classifiers (cuML,
XGBoost-CUDA) use an **isolated venv** pattern — see
`GPU_INSTRUCTIONS.md` §13 and `CODEBASE_ARCHITECTURE_AND_EXPERIMENT_LEARNING_GUIDE.md`
§26 for why, and rebuild those isolated venvs separately if you need the A0
baseline / CORAL / PCA scripts (not required for the A1–A5 PairPhysNet
matrix itself, which only uses the main venv).

Pre-download the HuggingFace model this pipeline needs:
```bash
export HF_HOME=<YOUR_DATASET_PATH>/cache/huggingface
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/wavlm-large')"
```

## 4. GPU policy — read before running anything

This project has a **hard, project-wide GPU-only compute policy**: every
stage that does real ML computation must run on GPU and must hard-fail
(never silently fall back to CPU). Full policy: `GPU_INSTRUCTIONS.md`.
On a non-SLURM machine (a single local GPU box, say), you will need to
adapt the `.sbatch` scripts in `paired_physio_device/scripts/*.sbatch` —
they assume SLURM (`sbatch`, `--gres=gpu:mig24gb:1`, `srun --overlap`). The
underlying Python scripts (`run_pairphysnet_training.py` etc.) do **not**
require SLURM themselves — they just need a visible CUDA GPU — so on a
single-GPU machine you can likely run them directly:
```bash
python3 paired_physio_device/scripts/run_pairphysnet_training.py \
  --variant A1 --fold 0 --epochs 15 --batch_size 8 --n_unfrozen_layers 2
```
(adjust `--batch_size`/gradient-checkpointing per your own GPU's VRAM — see
`paired_physio_device/GPU_EXECUTION_PLAN.md` for what was tuned on a 24GB
MIG slice and why; a machine with more VRAM per GPU may not need gradient
checkpointing or the reduced batch size at all — see the "Evidence-based
suggestions" logic in `watch.sh` for the isolated-vs-contended throughput
numbers this project already measured, useful context before re-tuning).

## 5. Run the exact same pipeline

```bash
# 1. Verify pairing/fold integrity against your copy of the dataset
python3 paired_physio_device/scripts/validate_pairing_and_folds.py

# 2. Smoke-test the model builds and trains one step on real data
#    (needs a GPU -- run via your cluster's job submission, or directly if local)
python3 paired_physio_device/scripts/test_pairphysnet_forward.py

# 3. Run the full A1-A5 x 5-fold matrix (25 jobs)
#    -- on SLURM: paired_physio_device/scripts/submit_pairphysnet_matrix_throttled.sh
#       (adjust MAX_CONCURRENT and the QOS assumptions for your cluster's limits --
#        see paired_physio_device/GPU_EXECUTION_PLAN.md for how this project's
#        specific QOS ceilings were discovered empirically; yours will differ)
#    -- standalone: loop scripts/run_pairphysnet_training.py over the 25
#       (variant, fold) combinations yourself

# 4. Watch it live
bash paired_physio_device/scripts/watch.sh
```

## 6. What "from scratch" actually means here

If you mean *exactly reproduce this project's splits/results*: keep
`metadata/*.csv` as-is (don't regenerate), just point at your copy of the
same dataset, fix paths (§1), and run.

If you mean *rebuild the manifest/folds/alignment from raw audio yourself*
(e.g. a genuinely different dataset): the manifest-building scripts
(`scripts/build_aligned_manifest.py`, `scripts/estimate_device_alignment_dense.py`)
are in this repo too, but were built and tuned against this exact dataset's
quirks (the two-part Recorder file split, the 8kHz native rate, the specific
clock-drift pattern) — expect to need to adapt them for a different dataset's
own conventions. See `paired_physio_device/AUDIT_REPORT.md` for what these
scripts actually assume.

## 7. Full context

- `paired_physio_device/EXPERIMENT_PLAN.md` — what this redesign is testing and why
- `paired_physio_device/STAGE_LOG.md` — exact provenance trail of everything run so far
- `paired_physio_device/AUDIT_REPORT.md` / `DATA_INTEGRITY_REPORT.md` — dataset ground truth and corrections
- `ara/` — the broader project's reverse-engineered claims/evidence base (predates this redesign)
