# Experiments

## E01: Main cross-device benchmark
- **Verifies**: C01, C02, C03
- **Setup**:
  - Model: 4 classifiers (RBF-SVM via cuML, MLP via a from-scratch PyTorch reimplementation, Random Forest via cuML, XGBoost with `device="cuda"`), each fit on frozen encoder embeddings
  - Hardware: single MIG 24GB GPU slice per job, SLURM-scheduled
  - Dataset: dual-device (bedside recorder R, smartphone S) overnight audio + PSG-derived apnea/hypopnea event annotations, subject-disjoint 5-fold CV
  - System: `scripts/run_main_benchmark.py` driving `src/sleep_quadnet/evaluation.py::run_fold`, results written to `results/P0_device_gap`
- **Procedure**:
  1. For each of 14 representations, load cached frozen-encoder features (extracted once via `scripts/extract_features.py`, resumable content-addressed cache)
  2. For each of 4 classifiers x 4 protocols x 5 folds, split subjects per protocol/fold, tune classifier hyperparameters on validation, refit on train+val, score on held-out test
  3. Aggregate per-representation, per-regime (matched vs. cross) mean balanced accuracy / F1 / ROC-AUC
- **Metrics**: balanced accuracy, F1, ROC-AUC, MCC, sensitivity, specificity (all computed by `evaluation.py::metrics()`)
- **Expected outcome**: a single strong general-purpose encoder should be competitive with or exceed naive multi-encoder concatenation on cross-device transfer, if the encoders' pretraining signal is largely redundant rather than complementary for this task
- **Baselines**: `classical` (52-D handcrafted DSP features) as the non-SSL floor; `odi_hb` as a chance-level sanity-check floor
- **Dependencies**: none

## E02: Leave-one-encoder-out ablation
- **Verifies**: C02, C10
- **Setup**:
  - Model: same 4 classifiers as E01
  - Hardware: same as E01
  - Dataset: same as E01
  - System: `scripts/run_main_benchmark.py --representations full_minus_{X}` for each of 6 encoders X, results in `results/P0_ablation`; significance via `scripts/run_ablation_significance.py`, results in `results/P0_ablation_statistics`
- **Procedure**:
  1. For each of 6 "full_fusion minus one encoder" representations, run the same 4x4x5 grid as E01
  2. Paired subject-level bootstrap each ablated variant against full_fusion on the same test subjects, same classifier, same protocol
- **Metrics**: balanced accuracy (primary), paired bootstrap mean difference, 95% CI, two-sided p-value
- **Expected outcome**: if the fusion's encoders are redundant (per E01's finding that fusion underperforms a single encoder), removing any one encoder should show no consistent, significant directional effect relative to the full fusion
- **Baselines**: full_fusion (all 5 encoders)
- **Dependencies**: E01

## E03: PCA dimensionality reduction, pre-fix
- **Verifies**: C04
- **Setup**:
  - Model: primarily svm_rbf (calibration-safe candidate variant, `probability=False`, per `advanced.py::_calibration_safe_candidates`); later extended to all 4 classifiers post-fix (E04)
  - Hardware: same as E01
  - Dataset: same as E01, restricted to `full_fusion`/`full_fusion_v2` representations, cross-device protocols only (R_S, S_R — CORAL/PCA are not meaningful for matched-device)
  - System: `scripts/run_dimension_control.py` driving `src/sleep_quadnet/advanced.py::run_pca_fold`, results in `results/P1_dimension_control`
- **Procedure**:
  1. Fit a fold-local PCA (tuning stage: source-device train only; refit stage: source-device train+val only, pre-fix)
  2. Reduce to 384/768/1536-D targets, fit classifier on reduced train+val, score on reduced test
- **Metrics**: balanced accuracy, ROC-AUC, predicted-probability range (used diagnostically to detect the collapse)
- **Expected outcome** (as originally run, before the collapse was understood): PCA reduction should retain most of the uncorrected representation's cross-device signal at lower dimensionality
- **Baselines**: uncorrected full_fusion/full_fusion_v2 (from E01)
- **Dependencies**: E01

## E04: PCA dimensionality reduction, target-aware-refit fix
- **Verifies**: C05
- **Setup**: same as E03, plus `wavlm_large` and `data2vec_fusion` representations, and all 4 classifiers (extended scope vs. E03)
- **Procedure**:
  1. Same as E03, but the refit stage's fit data now also includes unlabeled target-device validation features (`target_val_idx`, constructed identically to CORAL's own target-validation construction)
  2. Paired subject-level bootstrap each post-fix combination against its pre-fix (E03) counterpart on the same test subjects
- **Metrics**: balanced accuracy (primary), paired bootstrap mean difference, 95% CI, two-sided p-value, count of pre-fix combinations with balanced accuracy exactly 0.500 (collapse indicator)
- **Expected outcome**: if the collapse was caused by the refit's data scope (not a fundamental PCA limitation on this task), extending that scope to include unlabeled target-device data should resolve the collapse and produce a positive, significant effect
- **Baselines**: E03 (pre-fix PCA)
- **Dependencies**: E03

## E05: CORAL feature-space alignment
- **Verifies**: C06
- **Setup**:
  - Model: svm_rbf only (narrower scope than E01/E04 — see problem.md G3)
  - Hardware: same as E01
  - Dataset: `{data2vec_fusion, full_fusion, hubert}` representations, cross-device protocols only
  - System: `scripts/run_coral.py` driving `src/sleep_quadnet/advanced.py::run_coral_fold`, results in `results/P1_domain_adaptation`
- **Procedure**:
  1. Compute CORAL whitening-recoloring transform from source train+val features and unlabeled target-device validation features
  2. Apply the transform to source train+val (for fitting) and to test features (via the same fitted transform, never re-fit on test)
  3. Fit classifier on aligned train+val, score on aligned test
- **Metrics**: balanced accuracy, sensitivity, specificity (used diagnostically to show the majority-class-collapse pattern)
- **Expected outcome**: covariance alignment toward the target device's feature distribution should reduce the device gap relative to uncorrected features, if the gap is primarily a second-order (covariance-shape) distributional shift rather than a signal-content difference
- **Baselines**: uncorrected features (from E01), same representations/protocols
- **Dependencies**: E01

## E06: SpO2-corroboration training-label-quality ablation
- **Verifies**: C07
- **Setup**:
  - Model: all 4 classifiers
  - Hardware: same as E01
  - Dataset: `{full_fusion, hubert}` representations (primary test), `{wavlm, data2vec_fusion}` (extended coverage), all 4 protocols
  - System: `scripts/build_window_corroboration.py` (label prep) + `scripts/run_main_benchmark.py --filter-uncorroborated-training`, results in `results/P2_label_quality_ablation`; significance via `scripts/run_corroboration_significance.py`, results in `results/P2_statistics`
- **Procedure**:
  1. Cross-reference every annotated positive event against objective SpO2 desaturation timing (45s lag tolerance) to produce a per-window corroboration flag
  2. Retrain with corroboration-filtered positive training windows (test set always unfiltered)
  3. Paired subject-level bootstrap filtered vs. unfiltered baseline on the same test subjects
- **Metrics**: balanced accuracy (primary), F1, paired bootstrap mean difference, 95% CI, two-sided p-value
- **Expected outcome**: if uncorroborated positive events are annotation noise, filtering them out of training should improve or leave unchanged cross-device sensitivity; if they are real (differently-scored) positives, filtering should reduce available signal and hurt performance
- **Baselines**: unfiltered training (from E01)
- **Dependencies**: E01

## E07: HeAR (health-acoustic foundation model) integration and benchmark
- **Verifies**: C08
- **Setup**:
  - Model: all 4 classifiers, HeAR itself frozen (ViT-L MAE, gated `google/hear` checkpoint) accessed via an isolated TensorFlow/Keras virtual environment reached by subprocess bridge
  - Hardware: same as E01, plus a one-time isolated-venv GPU verification step (`tf.config.list_physical_devices("GPU")` hard-checked, no silent CPU fallback)
  - Dataset: same manifest as E01; each window reduced to a single fixed 2.0s/16kHz/32,000-sample clip (center-crop if longer, zero-pad if shorter) since HeAR has no variable-length input mode
  - System: `scripts/extract_hear_features.py` (extraction) + `scripts/run_main_benchmark.py --representations hear,full_fusion_plus_hear`, results in `results/P0_device_gap`
- **Procedure**:
  1. Extract HeAR embeddings (512-D) for all manifest windows via the isolated venv, batched to amortize model-load cost
  2. Benchmark `hear` alone and `full_fusion_plus_hear` (full_fusion + HeAR) through the same E01 grid
- **Metrics**: balanced accuracy, F1, ROC-AUC (same as E01)
- **Expected outcome**: a foundation model pretrained specifically on health acoustics (coughs, breathing, ~174k hours) should provide complementary or superior signal to general-purpose speech SSL encoders for this respiratory-sound task
- **Baselines**: full_fusion (no HeAR), each general-purpose encoder alone (from E01)
- **Dependencies**: E01

## E08: ODI/Hypoxic-Burden as a per-window classifier feature
- **Verifies**: C09
- **Setup**:
  - Model: all 4 classifiers
  - Hardware: same as E01 for the classifier stage; ODI/HB computation itself is CPU-only (no model inference involved)
  - Dataset: same manifest as E01, `odi_hb` (2-D, ODI + hypoxic burden) and `hubert_odi_hb` (HuBERT 768-D concatenated with the same 2-D feature) representations
  - System: `scripts/compute_odi_hypoxic_burden.py` (per-subject computation from raw SpO2, sleep-time-corrected via `awake_intervals`) + `scripts/run_main_benchmark.py --representations odi_hb,hubert_odi_hb`, results in `results/P0_device_gap`
- **Procedure**:
  1. Compute per-subject ODI (desaturation-event count / sleep hour) and hypoxic burden (time-integrated desaturation area / sleep hour) from raw SpO2, excluding awake intervals
  2. Validate against PSG-annotated OSA+hypopnea event counts (Pearson correlation)
  3. Benchmark the per-subject-constant feature alone and concatenated onto HuBERT, through the same E01 grid
- **Metrics**: balanced accuracy, F1 (classifier stage); Pearson r (validation stage, against annotated event counts, not part of the classifier grid)
- **Expected outcome**: a value constant within a subject across all of that subject's windows cannot add per-window discriminative signal to a per-window classification task, regardless of the value's subject-level clinical validity
- **Baselines**: HuBERT alone (from E01), for the fused variant; chance level, for the alone variant
- **Dependencies**: E01

## E09: Sliding-window whole-night severity classification (in progress)
- **Verifies**: C11
- **Setup**:
  - Model: all 4 classifiers
  - Hardware: same as E01
  - Dataset: 6,013 five-minute whole-night epochs (50 subjects, from raw SpO2, independent of PSG event annotations) reduced to 9,509 usable (subject, device, epoch) audio rows (441 dropped for running past that device's actual audio duration; 9 subjects skipped for lacking a usable audio-file template), HuBERT features only so far, binarized severity target (severe vs. not, from a 4-class `severity_bin`)
  - System: `scripts/build_sliding_window_ahi_targets.py` (ground truth) + `scripts/build_sliding_window_manifest.py` (audio-manifest bridging) + `scripts/extract_features.py` (feature extraction, separate cache root) + `scripts/run_sliding_window_severity.py` (training/eval, self-contained subject-disjoint splitting reusing the main pipeline's fold assignments), results in `results/P3_sliding_window_severity`
- **Procedure**:
  1. Bin each subject's full night into fixed, non-overlapping 5-minute clock epochs (not centered on annotated events), computing per-epoch desaturation count and hypoxic-burden area, scaled to an hourly-rate proxy
  2. Bridge to a `load_manifest_window()`-compatible audio manifest by borrowing audio-file templates from the main manifest per (subject, device)
  3. Extract HuBERT features per epoch (reusing `extract_feature_cache()` unchanged)
  4. Train/evaluate binary severe-vs-not classifiers with subject-disjoint splitting (same fold file as E01) across all 4 protocols
- **Metrics**: balanced accuracy, F1, ROC-AUC (same computation as E01)
- **Expected outcome**: unknown / genuinely open — this experiment exists to test feasibility of a non-annotation-privileged severity target, not to confirm a specific predicted direction
- **Baselines**: none yet defined within this experiment; chance level (0.5) is the only implicit comparator so far
- **Dependencies**: E08 (shares the same underlying ODI/HB desaturation-detection logic, applied per-epoch instead of per-subject)

## E10: Deployment/efficiency latency-memory benchmark (partial)
- **Verifies**: none (this experiment measures deployment cost, not any accuracy-bearing claim; it supplies context for the practical-deployment framing of C01 and C02 without itself being their proof)
- **Setup**:
  - Model: `_load_model`-loaded frozen encoders, single-clip (batch size 1) inference
  - Hardware: single MIG 24GB GPU slice
  - Dataset: a representative sample of manifest windows per feature (`--clips 20` default), warm-GPU timing (3 warmup clips excluded)
  - System: `scripts/benchmark_efficiency.py`, results in `results/P0_efficiency/component_runs/*.json`
- **Procedure**:
  1. For each feature, load the model once, run warmup clips, then time `--clips` further clips with `torch.cuda.synchronize()` bracketing
  2. Record latency, peak GPU memory, real-time factor, CPU RSS
- **Metrics**: latency (s/clip), peak GPU memory (bytes), real-time factor (processing time / audio duration)
- **Expected outcome**: no specific directional hypothesis — this experiment characterizes cost, not accuracy
- **Baselines**: none (absolute measurement)
- **Dependencies**: none (independent of E01's classifier-stage results; measures only the encoder-extraction stage)
