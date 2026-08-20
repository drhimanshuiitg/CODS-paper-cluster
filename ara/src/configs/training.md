# Training Configuration

## Random seed
- **Value**: 42 (base), fold-perturbed downstream: `seed + fold` (PCA random_state), `seed + fold*100` (hyperparameter selection), `seed + fold*1000` (final classifier fit)
- **Rationale**: single fixed base seed for reproducibility; per-fold perturbation avoids all 5 folds sharing identical random initialization while remaining deterministic given the base seed.
- **Search range**: not searched — fixed project-wide constant.
- **Sensitivity**: low for the base value itself (not tuned); the *pattern* of fold-dependent perturbation is load-bearing for avoiding identical folds, so changing the perturbation formula (not just the seed value) would be medium-sensitivity.
- **Source**: `configs/base.yaml:3` (`seed: 42`)

## Cross-validation folds
- **Value**: 5 (fixed), subject-disjoint (see concepts.md)
- **Rationale**: standard k-fold choice; 5 balances per-fold test-subject count against total compute budget given the ~41-50 subject cohort.
- **Search range**: not searched.
- **Sensitivity**: medium — fewer folds would give larger per-fold test sets (more stable per-fold metrics) but fewer independent train/test configurations; not empirically compared in this project.
- **Source**: `metadata/subject_folds_5cv_aligned.csv`, `src/sleep_quadnet/evaluation.py::split_indices`

## random_forest hyperparameter candidates
- **Value**: `{n_estimators: 300, max_depth: null|20, class_weight: balanced, n_jobs: 4}` (2 candidates, differing only in max_depth)
- **Rationale**: `class_weight: balanced` addresses the class-imbalanced binary task; `n_jobs`/`max_depth` candidates are selected per-fold via validation performance (`select_estimator`), not fixed a priori. `n_jobs` is dropped when routed through the GPU (cuML) path since it has no GPU analogue.
- **Search range**: max_depth in {unlimited, 20}
- **Sensitivity**: not independently ablated in this project — selection is per-fold via validation score, so sensitivity is implicitly handled by the pipeline rather than separately characterized.
- **Source**: `configs/base.yaml:54-56`

## xgboost hyperparameter candidates
- **Value**: `{n_estimators: 300, max_depth: 4|8, learning_rate: 0.05, subsample: 0.9, colsample_bytree: 0.9, tree_method: hist, device: cuda}` (2 candidates, differing only in max_depth; `device: cuda` set at construction time, not in this file, per `build_estimator`)
- **Rationale**: `tree_method: hist` is the modern GPU-compatible histogram method (not the deprecated `gpu_hist`); `subsample`/`colsample_bytree: 0.9` provide mild regularization against overfitting a high-dimensional (up to 3840-D fused) input.
- **Search range**: max_depth in {4, 8}
- **Sensitivity**: not independently ablated.
- **Source**: `configs/base.yaml:57-59`; `src/sleep_quadnet/evaluation.py::build_estimator`'s GPU-device-setting comment

## svm_rbf hyperparameter candidates
- **Value**: `{C: 1.0|10.0, gamma: scale, class_weight: balanced, cache_size: 4096}` (2 candidates, differing only in C); `probability: true` in the config but dropped when routed through cuML (no Platt-scaling equivalent) or through the PCA/CORAL paths (`_calibration_safe_candidates` strips it, using a manual-sigmoid decision-function fallback instead)
- **Rationale**: RBF kernel is a standard non-linear margin-based classifier appropriate for frozen high-dimensional embeddings; `class_weight: balanced` for the same imbalance reason as random_forest.
- **Search range**: C in {1.0, 10.0}
- **Sensitivity**: HIGH for the `probability=True`/Platt-scaling setting specifically — diagnosed this project (pre-session) as causing a probability-collapse failure mode on PCA/CORAL's reduced feature spaces (narrow probability band around 0.5 regardless of true separation); fixed by dropping Platt scaling for those two paths only, using `probability()`'s manual-sigmoid fallback on `decision_function()` instead. Not observed on the un-reduced main-benchmark/ablation representations.
- **Source**: `configs/base.yaml:60-62`; `src/sleep_quadnet/advanced.py::_calibration_safe_candidates` docstring

## mlp hyperparameter candidates
- **Value**: `{hidden_layer_sizes: [256, 128], max_iter: 400, alpha: 0.0001|0.001, early_stopping: true}` (2 candidates, differing only in alpha/L2 regularization strength)
- **Rationale**: a 2-hidden-layer MLP (256, 128 units) as the "small neural net" member of the classifier ensemble, deliberately not deep, so any observed effect is attributable to representation choice rather than extra classifier capacity (per `CODEBASE_ARCHITECTURE_AND_EXPERIMENT_LEARNING_GUIDE.md`'s stated design rationale).
- **Search range**: alpha in {0.0001, 0.001}
- **Sensitivity**: not independently ablated; implemented as a from-scratch PyTorch reimplementation (`TorchMLPClassifier`) rather than sklearn's `MLPClassifier` for GPU acceleration — explicitly documented as "a faithful approximation," not bit-identical to the sklearn original.
- **Source**: `configs/base.yaml:63-65`; `src/sleep_quadnet/evaluation.py::TorchMLPClassifier`

## Audio preprocessing
- **Value**: target sample rate 16,000 Hz (upsampled from the dataset's native 8,000 Hz on both devices); bandpass filter 20-4,000 Hz, order 4 (`filter`/`peak_filter` preprocessing variants only); peak-normalization epsilon 1e-8; SSL chunk length 20.0s (for encoders processing windows longer than one native forward pass)
- **Rationale**: 16kHz upsampling is required only for compatibility with SSL encoders pretrained at that rate — both devices' native 8kHz capture (Nyquist 4kHz) means no new spectral information is added by the upsampling itself (stated explicitly in `paper/conference_101719.tex`'s dataset section).
- **Search range**: `raw`/`peak`/`filter`/`peak_filter` preprocessing variants exist; `peak` is the default for all non-`classical` features, `peak_filter` for `classical`.
- **Sensitivity**: not independently ablated across variants in this ARA's covered experiments (all use each feature's documented default preprocessing).
- **Source**: `configs/base.yaml:21-32`

## Corroboration-filter lag tolerance
- **Value**: 45 seconds
- **Rationale**: the maximum plausible lag between an annotated apnea/hypopnea event's onset and its corresponding SpO2 desaturation (physiological response delay), used to decide whether a training-positive window has objective SpO2 corroboration.
- **Search range**: not searched — a single fixed domain-informed value.
- **Sensitivity**: not independently ablated; a materially smaller or larger tolerance would change which events count as "corroborated" and could shift C07's specific effect size, though the direction (filtering hurts) is unlikely to flip given the effect held across 32/32 combinations in the tested direction.
- **Source**: `scripts/audit_spo2_corroboration.py` (`LAG_TOLERANCE_SEC`)
