# Concepts

## Cross-device protocol
- **Notation**: `X_Y` where `X` = training device, `Y` = test device, `X, Y in {R (bedside recorder), S (smartphone)}`
- **Definition**: A training/evaluation configuration in which the classifier is fit on windows from device `X` only and evaluated on windows from device `Y` only, both restricted to their respective fold's subject sets. `R_R` and `S_S` are "matched-device" (same device train/test); `R_S` and `S_R` are "cross-device" (different device train/test). Implemented in `src/sleep_quadnet/evaluation.py:protocol_devices()`.
- **Boundary conditions**: A 5th protocol, `RS_RS` (train and test on both devices pooled), is defined in `configs/base.yaml` but has never been run (see problem.md G5) — it answers a different question (does pooling induce device-robustness) than the 4 protocols actually benchmarked (does a device-specific model transfer).
- **Related concepts**: Device gap, Subject-disjoint fold

## Subject-disjoint fold
- **Notation**: 5-fold assignment `folds: subject_id -> {0,1,2,3,4}`, fold `f`'s test set = `{s : folds[s] = f}`, val set = `{s : folds[s] = (f+1) mod 5}`, train set = remaining subjects.
- **Definition**: A cross-validation scheme in which no subject's windows (from either device) appear in more than one of train/validation/test within a given fold, enforced by a runtime `AssertionError` on every fold-run (`evaluation.py:95-96`), not merely by construction of the fold-assignment file.
- **Boundary conditions**: This prevents subject-level leakage but does not by itself guarantee window-level independence within a subject (e.g. temporally adjacent windows from the same recording night could still share short-timescale acoustic characteristics); no experiment in this ARA specifically tests for that finer-grained leakage mode.
- **Related concepts**: Cross-device protocol, Device gap

## Device gap
- **Notation**: `gap(rep) = mean_BA(rep, matched) - mean_BA(rep, cross)`, where `BA` = balanced accuracy.
- **Definition**: The drop in classifier performance when train and test devices differ, relative to when they match, for a fixed representation. The central quantity this project's main benchmark characterizes.
- **Boundary conditions**: Defined only for representations with both matched- and cross-device results computed with the same classifier-selection methodology; `odi_hb` has gap = 0 by construction (the feature does not vary by device), which is a degenerate rather than a genuinely device-robust case.
- **Related concepts**: Cross-device protocol, Domain adaptation

## Domain adaptation (PCA / CORAL, this project's usage)
- **Notation**: PCA: `z = W^T(x - mu)`, `W` fit on some subset of available feature vectors `x`. CORAL: `x' = (x - mu_s) Sigma_s^{-1/2} Sigma_t^{1/2} + mu_t`, source mean/cov `mu_s, Sigma_s`, target mean/cov `mu_t, Sigma_t`.
- **Definition**: Two post-hoc statistical corrections applied to frozen encoder embeddings before classifier fitting, intended to reduce the device gap without retraining the encoder. PCA reduces dimensionality (and, if fit including target-device data, implicitly re-centers/re-scales toward a shared subspace); CORAL explicitly whitens-and-recolors source features toward the target's covariance structure.
- **Boundary conditions**: Both are fit using only unlabeled data (PCA post-fix, CORAL always) from target-device validation subjects — never test-set data, and never target-device labels. This is a deliberate boundary enforced in code (`run_coral_fold`'s docstring: "target covariance uses unlabeled target-device validation subjects only; test features never fit the alignment").
- **Related concepts**: Device gap, Target-aware refit

## Target-aware refit
- **Notation**: `fit_data = X_train_source union X_val_source union X_val_target(unlabeled)`
- **Definition**: The specific fix applied to PCA in this project (`src/sleep_quadnet/advanced.py::run_pca_fold`, 2026-08-19): extending the final ("refit") PCA fit's input data to include unlabeled target-device validation features, in addition to the source-device train+val features it already used. Named to distinguish it from PCA's separate "tuning" stage, which remains source-only (no leakage risk there since tuning only selects hyperparameters, not the final transform).
- **Boundary conditions**: Only applied for cross-device protocols (`R_S`, `S_R`); for matched-device protocols (`R_R`, `S_S`) `target_val_idx` is empty by construction (`target_devices = test_devices - train_devices` is empty when train and test devices are the same), so the fix is a no-op there and matched-device PCA behavior is unchanged.
- **Related concepts**: Domain adaptation, CORAL

## Corroboration-filtered training
- **Notation**: `train' = train \ {w in train : label(w)=1 and not spo2_corroborated(w)}`, negatives always kept.
- **Definition**: A training-data filtering ablation that drops annotated-positive training windows whose event lacks a matching SpO2 desaturation within a 45-second lag tolerance (`scripts/audit_spo2_corroboration.py`), testing whether such windows are annotation noise. Test set is always the full, unfiltered set in both arms of any comparison.
- **Boundary conditions**: Corroboration is computed independently per event via `detect_desat_windows()`; a positive window's corroboration status is looked up via its `logical_window_id` shared across R/S device rows of the same underlying event (`scripts/build_window_corroboration.py`), so both devices' versions of the same event get the same filter decision.
- **Related concepts**: SpO2 desaturation, Label-quality ablation

## Fusion representation
- **Notation**: `full_fusion = concat(hubert, wavlm, wav2vec2, data2vec_audio, data2vec_spectrogram) in R^3840`; other fusions concatenate different subsets, each dimension fixed per `src/sleep_quadnet/features.py:FEATURE_DIMENSIONS`.
- **Definition**: A representation formed by concatenating multiple frozen encoders' per-window embedding vectors before classifier input, with no learned fusion weighting, attention, or gating — the simplest possible fusion strategy.
- **Boundary conditions**: Concatenation order and which encoders are included define a specific named representation in `configs/base.yaml:representations`; a "leave-one-out" ablation variant is a fusion representation with exactly one encoder's slot removed from `full_fusion`.
- **Related concepts**: Encoder redundancy, Main benchmark

## Paired subject-level bootstrap significance
- **Notation**: For two arms A, B with per-subject metric scores `s_A(subj), s_B(subj)` on the same aligned test set: `diff = s_A - s_B` per subject, resampled with replacement `iterations` times (2,000 throughout this project) over the subject index, 95% CI from the 2.5/97.5 percentiles of the resampled mean difference; two-sided p-value = `2 * min(P(diff<=0), P(diff>=0))` under the bootstrap distribution.
- **Definition**: The statistical test used for every significance claim in this project (`scripts/run_statistics.py`, `run_pca_fix_significance.py`, `run_corroboration_significance.py`, `run_ablation_significance.py` — 4 near-identical implementations of the same method for different comparison pairs). Requires the two arms' test sets to align exactly on `(subject_id, logical_window_id, label)`, enforced by an explicit `ValueError` if row counts diverge after the join.
- **Boundary conditions**: Because resampling is over subjects (not windows), it correctly accounts for within-subject correlation of window-level errors, at the cost of requiring enough test subjects per fold for the bootstrap to be meaningful (5-fold CV here means each test fold has a fairly small subject count).
- **Related concepts**: Device gap, all C01-C10 claims
