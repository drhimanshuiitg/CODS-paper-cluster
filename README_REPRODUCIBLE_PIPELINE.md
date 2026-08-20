# Reproducible Sleep-QuadNet revision pipeline

The authoritative scientific scope is `Sleep_QuadNet_Workshop_Experiment_Plan.md`; cluster execution rules are in `GPU_INSTRUCTIONS.md`.

## Final data contract

Raw audio and annotations remain read-only beneath `/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/Data`. Experiments consume `metadata/dataset_manifest_aligned.csv` and `metadata/subject_folds_5cv_aligned.csv`.

The available corpus contains 50 annotated subjects, 43 subjects with both device files, and 41 subjects with reliable synchronized regions. Recorder timestamps are mapped from the Smartphone/annotation reference clock with label-free RMS-envelope cross-correlation anchors. Subjects 34 and 35 and uncertain intervals between recording discontinuities are excluded. Full details and the original unaligned materialization are preserved in `results/audit/` and `metadata/quarantine_linear_alignment_v1/`.

## Leakage controls

- One fixed five-fold subject assignment is shared by every device and model.
- Fold `k` is test; fold `(k+1) mod 5` is validation; the other folds train.
- Cross-device validation uses only the source device for model selection.
- Scalers, classifiers, PCA, and CORAL source transforms are fit without test subjects.
- CORAL uses unlabeled target-device validation subjects only and never target test features.
- Every test-window prediction retains its subject and logical-window identity for subject-level bootstrap analysis.

## Resumability and preservation

Feature caches have independent completion bitmaps. A failed extraction restarts at the first incomplete row, and an extraction error is recorded and raised rather than replaced with a zero vector. Completed fold runs are found by a deterministic experiment key and skipped. Failed retries use new directories. The master log is append-only and file-locked. Final aggregators refuse to overwrite existing different outputs.

## Main entry points

```bash
.venv/bin/python scripts/show_progress.py
.venv/bin/python scripts/validate_splits.py
.venv/bin/python scripts/validate_windows.py
.venv/bin/python scripts/extract_features.py --feature classical
.venv/bin/python scripts/run_main_benchmark.py --representations classical --classifiers svm_rbf --protocols R_R,S_S,R_S,S_R
```

All large work is submitted through the scripts in `slurm/`. Because this cluster limits each user to four submitted jobs including interactive allocations, the `*s_*.sbatch` files execute resumable stages sequentially and the main benchmark is divided into three concurrent waves.

## Output map

- `cached_features/`: aligned feature matrices and completion maps
- `checkpoints/huggingface/`: downloaded encoder checkpoints
- `results/P0_*`, `results/P1_*`: structured experiment outputs
- `results/master_experiment_log.csv`: append-only run registry
- `results/tables/`: paper tables
- `figures/`: vector PDF figures
- `logs/`: SLURM stdout/stderr with progress bars
- `results/experiment_summary.md`: final factual project summary
