# Problem Specification

## Observations

### O1: The rejected submission's numbers came from a leaky, non-reproducible script
- **Statement**: The originally submitted paper's headline numbers (F1=94.7% matched, Combined F1=94.38%, S->R F1=90.9%, AUC=0.950) trace to `device_robust_sleep_apnea_experiments_v4.py`, a deprecated script whose config (`config_device_robust_v4.yaml`) sets `strict_patient_disjoint_cross_device: false`, so cross-device protocols train on every source-domain subject and test on every target-domain subject with paired subjects appearing on both sides via the opposite device.
- **Evidence**: `results/audit/current_pipeline_audit.md` (dated 2026-08-17, independent audit of the v4 script and its config).
- **Implication**: Those numbers are not trustworthy and are not reproducible against the real dataset (the v4 script's labels are filename-keyword-based and incompatible with this project's annotation-JSON-driven corpus). All numbers in this ARA come from the current, re-verified pipeline only.

### O2: The current pipeline enforces subject-disjoint splitting with a hard runtime assertion
- **Statement**: `split_indices()` computes `train_subjects & val_subjects`, `train_subjects & test_subjects`, `val_subjects & test_subjects` and raises `AssertionError("Subject leakage in split construction")` if any intersection is non-empty, on every fold-run, not just at setup time.
- **Evidence**: `src/sleep_quadnet/evaluation.py:71-100`.
- **Implication**: Subject-level leakage of the kind found in O1 is structurally prevented in the current pipeline, not just avoided by convention.

### O3: Cross-device performance is uniformly worse than matched-device performance across every representation tested
- **Statement**: Across all 14 representations in the main benchmark, matched-device balanced accuracy exceeds cross-device balanced accuracy by 0.0-7.5 points (mean gap ~4.7 points), with the smallest gap for `odi_hb` (0.0, itself a chance-level representation) and the largest for `hear` alone (7.5 points).
- **Evidence**: `results/P0_device_gap` (1,120 completed fold-runs), aggregated by representation x regime.
- **Implication**: The device gap is not an artifact of one representation or classifier; it is a property of the task.

### O4: HuBERT alone beats every tested fusion combination cross-device
- **Statement**: Mean cross-device balanced accuracy: HuBERT 0.5466, HuBERT+ODI/HB 0.5473 (statistically indistinguishable from HuBERT alone since ODI/HB is a per-subject constant), full_fusion (5-encoder concat) 0.5311, full_fusion_v2 (4-encoder) 0.5357, full_fusion_plus_hear (6-encoder incl. HeAR) 0.5288.
- **Evidence**: `results/P0_device_gap`, per-representation mean cross-device balanced accuracy.
- **Implication**: Concatenating more frozen SSL encoders does not help, and mildly hurts, cross-device generalization on this task.

### O5: PCA dimensionality reduction, as originally implemented, collapsed to a degenerate single-class predictor
- **Statement**: In `results/P1_dimension_control` (pre-fix), 43 of 60 audited representation/classifier/fold/dimension combinations produced test-set balanced accuracy of exactly 0.500, with maximum predicted test-window probability never exceeding 0.50 in the R->S direction (mirror-image all-positive collapse in S->R).
- **Evidence**: `results/P1_statistics_pca_fix/pca_fix_vs_baseline.csv` (`collapsed_baseline` column: 43/60), root-caused in `src/sleep_quadnet/advanced.py` prior to the 2026-08-19 fix.
- **Implication**: A domain-adaptation technique can appear to "fail the task" when it is actually failing due to an implementation-scope bug, not a fundamental limitation — this must be checked before concluding a technique doesn't work.

### O6: The PCA collapse was root-caused to a source-device-only refit and fixed by scoping it like CORAL
- **Statement**: The original `run_pca_fold()` fit its final ("refit") PCA on `concatenate([x_train, x_val])`, both restricted to source-device subjects by `split_indices()`. CORAL's own alignment, in the same codebase, already used unlabeled target-device validation data (`target_val_idx`) in its covariance fit. Extending PCA's refit to include the same unlabeled target-device validation data resolves the collapse.
- **Evidence**: `src/sleep_quadnet/advanced.py` (current `run_pca_fold()`, `target_val_idx` construction mirrors `run_coral_fold()`'s pre-existing pattern); `scripts/_pca_fix_smoketest.py` (standalone before/after comparison, +3.37pt BA on one combo, prior to the full validation run).
- **Implication**: The fix is not a new technique — it is applying CORAL's own already-correct scope to PCA, which had never had it.

### O7: The PCA fix produces a statistically significant, exclusively positive improvement at scale
- **Statement**: Paired subject-level bootstrap (2,000 iterations) comparing post-fix (`P1_dimension_control_v3`) vs. pre-fix (`P1_dimension_control`) on the 60 overlapping representation/classifier/fold/dimension combinations: mean balanced-accuracy difference +0.0299 (30/60 individually significant positive, 0/60 significant negative).
- **Evidence**: `results/P1_statistics_pca_fix/pca_fix_vs_baseline.csv`.
- **Implication**: This is not a marginal or mixed result — every significant effect found points the same direction.

### O8: Two independently-tested candidate improvements (SpO2-corroboration filtering, HeAR) both failed
- **Statement**: (a) Filtering training-positive windows lacking a corroborating SpO2 desaturation: paired bootstrap over 32 combinations shows balanced-accuracy point difference negative in 30/32 (2/32 marginally positive), significant negative in 12/32, zero significant positive (`results/P2_statistics/corroboration_filter_vs_baseline.csv`). (b) HeAR alone: cross-device balanced accuracy 0.5040 (weakest of all 14 tested representations, barely above the 0.50 chance floor); fused into full_fusion: 0.5288 vs. full_fusion alone 0.5311 (`results/P0_device_gap`).
- **Evidence**: as cited inline.
- **Implication**: Both were reasonable, motivated hypotheses (label-quality noise reduction; a health-domain-pretrained foundation model) that the data does not support on this task — reported as negative results per this project's stated verification discipline, not discarded.

### O9: A leave-one-encoder-out ablation shows the fusion's encoders are largely redundant, not complementary
- **Statement**: Paired bootstrap of each of 6 single-encoder-removed fusion variants vs. full full_fusion, over 89 comparable representation/classifier/protocol combinations (7 of 96 skipped for incomplete fold coverage at time of test): mean balanced-accuracy difference -0.0003, only 7/89 individually significant (4 positive, 3 negative).
- **Evidence**: `results/P0_ablation_statistics/ablation_vs_full_fusion.csv`.
- **Implication**: Removing any single encoder is statistically indistinguishable from keeping it, in the large majority of configurations — consistent with O4 (the full fusion underperforming a single good encoder) rather than contradicting it.

## Gaps

### G1: CLOSED (2026-08-19) — significance testing has been refreshed with the full 14-representation candidate set
- **Statement**: `results/P0_statistics/selection.json` recorded `best_single_encoder: "hubert"`, selected from a fixed candidate list of 5 original encoders that did not include `wavlm_large`, `hear`, or `full_fusion_plus_hear`. Extending `scripts/run_statistics.py`'s `candidate_singles` list and re-running (`results/P0_statistics_v2`) changed the answer: `wavlm_large` is now the validation-selected best single encoder (validation BA 0.6252 vs. hubert's 0.5940 under the primary classifier, svm_rbf), and full_fusion is not significantly different from it in either cross-device direction. This materially revised C01/C02/C08 (see those claims' Interpretation fields) rather than merely adding a missing significance annotation to an unchanged ranking.
- **Caused by**: O4's newer representations (wavlm_large, hear, full_fusion_plus_hear) were added to `configs/base.yaml` and benchmarked after `P0_statistics` was last computed.
- **Existing attempts**: closed via a Level 2 review finding (F05) and the resulting re-run, `results/P0_statistics_v2` / `evidence/tables/table11_refreshed_significance_full_candidate_set.md` and `table12_refreshed_confidence_intervals.md`.
- **Why they fail**: N/A — resolved, not a persisting failure mode. `run_statistics.py`'s `candidate_singles`/`principal` lists are still hardcoded (a future new representation would reopen a version of this same gap) -- worth a durable fix (e.g. deriving candidates from `configs/base.yaml` directly) rather than relying on manual re-runs each time.

### G2: PARTIALLY CLOSED (2026-08-19) — wavlm_large and hear now benchmarked; the 4 fusion representations remain deliberately un-benchmarked (derivable, not missing)
- **Statement**: `results/P0_efficiency/component_runs/` now has 8 JSON files (added wavlm_large: 26.0ms/clip; hear: 22.3ms/clip, 512-D). Extending coverage to `hear` caught and fixed a real bug: `benchmark_efficiency.py` had never been taught HeAR's isolated-venv subprocess-bridge extraction path (it crashed with `KeyError: 'hear'` trying to load it through the standard transformers path) -- fixed by adding a dedicated `benchmark_hear()` path and a matching `bench` mode in `hear_worker.py` that times single-clip (batch_size=1) latency after loading the model once, consistent with every other feature's measurement convention. `full_fusion`, `full_fusion_v2`, `full_fusion_plus_hear`, `data2vec_fusion` remain un-benchmarked, but deliberately: each is a concatenation of already-individually-benchmarked components with no separate model of its own to time -- their latency is the sum of their components' latencies, not a new measurement.
- **Caused by**: `wavlm_large` and `hear` were added after the efficiency sweep was last run; `hear`'s crash was caused by `benchmark_efficiency.py` predating HeAR's isolated-venv integration entirely.
- **Existing attempts**: `scripts/benchmark_efficiency.py --feature wavlm_large` and `--feature hear` (job 1603 for wavlm_large, succeeded; job 1606 for hear, after the fix, succeeded).
- **Why they fail**: N/A for wavlm_large/hear — resolved. The 4 fusion representations are not attempted because doing so would not be a new measurement (see Statement) — closing this properly would mean adding a small derived-sum row to Table 7, not running a new benchmark.

### G3: CORAL's scope is narrower than the main benchmark and was never extended after being fixed
- **Statement**: `results/P1_domain_adaptation` has exactly 30 completed combinations, spanning only `{data2vec_fusion, full_fusion, hubert}` x `{R_S, S_R}` x 5 folds x svm_rbf. No mlp/random_forest/xgboost CORAL runs exist, unlike the PCA-fix validation which was extended to all 4 classifiers.
- **Caused by**: CORAL was implemented and benchmarked before the project's GPU-classifier expansion to 4 classifiers became the default expectation.
- **Existing attempts**: none yet this session to extend it.
- **Why they fail**: no job has been submitted for the missing classifier coverage.

### G4: CLOSED (2026-08-19) — the sliding-window whole-night severity classifier's binary-first-cut run completed; 4-class/regression framing and device-invariance handling remain open
- **Statement**: `results/P3_sliding_window_severity` (80/80 combinations complete) shows cross-device balanced accuracy 0.4859 (below chance) and matched-device 0.5307, using single-encoder HuBERT features and a binarized severe-vs-not target, all 4 protocols. This resolves the "is it in progress" gap; it does not resolve the underlying limitations (binary-only framing, no device-invariance mechanism, no significance test yet for this specific result — see C11).
- **Caused by**: the target (`severity_bin` from `metadata/sliding_window_ahi_targets.csv`) was only just built this session and reduced to binary for a fast first pass reusing existing binary-classification infrastructure.
- **Existing attempts**: the binary-first-cut run is complete (C11); a 4-class severity or continuous AHI-proxy regression framing has not been attempted.
- **Why they fail**: the completed result is consistent with C01-C03's device-gap pattern — cross-device transfer fails (below-chance mean BA) with a single encoder and no explicit device-invariance mechanism, while the matched-device signal (weak but real) suggests the underlying target is learnable in principle. Whether a 4-class/regression framing or a device-invariance-aware model would change the cross-device picture is untested.

### G5: The `RS_RS` pooled-device protocol has never been run
- **Statement**: `configs/base.yaml`'s `protocols:` list includes `RS_RS` (train and test on both devices pooled) alongside `R_R, S_S, R_S, S_R`; only the latter 4 appear in any completed results directory.
- **Caused by**: not prioritized; the project's device-gap framing centers on directional transfer (R->S, S->R), which `RS_RS` does not directly measure.
- **Existing attempts**: none.
- **Why they fail**: not attempted, not a failure — a genuine open question ("does pooling both devices at train time make the model naturally device-robust?") distinct from the transfer question this benchmark answers.

## Key Insight
- **Insight**: A domain-adaptation technique's apparent failure on a task can itself be an artifact of how narrowly the technique was scoped (source-device-only data), not evidence the technique is unsuited to the task — the correct comparison is not "does technique X work" in the abstract, but "does technique X work when scoped consistently with how a working analogous technique (CORAL) in the same codebase was already scoped."
- **Derived from**: O5, O6, O7 (the PCA collapse, its root cause via comparison to CORAL's scope, and the fix's statistically significant, unidirectional effect).
- **Enables**: A more nuanced practical recommendation than a blanket "post-hoc alignment doesn't work for this problem" — some strategies are correctable, some (CORAL, per O8) are not, and this distinction should shape which representation-level interventions (contrastive training, domain-adaptive pretraining) are prioritized next.

## Assumptions
- A1: Balanced accuracy and F1 on held-out subject-disjoint folds are the correct primary metrics for this class-imbalanced binary (apnea-event-window vs. negative) task; this is a project convention (`configs/base.yaml`, `evaluation.py:metrics()`), not independently re-derived in this ARA.
- A2: The 41-50 subject cohort (device-count-dependent) is assumed representative enough for subject-disjoint 5-fold CV to produce meaningful, non-degenerate folds; no external dataset validation exists yet (see G-adjacent open item in constraints.md).
- A3: Frozen (not fine-tuned) SSL encoder embeddings are the intended comparison basis throughout — no experiment in this ARA fine-tunes an encoder.
- A4: All GPU-accelerated classifier paths (cuML for random_forest/svm_rbf, xgboost `device="cuda"`, a from-scratch PyTorch MLP) are assumed numerically valid substitutes for their CPU counterparts; this project enforces hard GPU-presence checks (no silent CPU fallback) but does not claim bit-identical output to sklearn's CPU implementations for classifiers ported from sklearn originals.
