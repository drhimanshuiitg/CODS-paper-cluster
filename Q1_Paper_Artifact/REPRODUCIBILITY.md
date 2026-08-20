# Reproducibility

Distinguishes facts independently verifiable from the repository/results outputs ("reproducible") from facts documented only in narrative form and not independently re-verified in this pass ("documented, not independently re-verified").

## Environment — documented, not independently re-verified this pass

- Python environment for main pipeline: repository-level virtualenv (path referenced in project documentation as the primary env; exact `pip freeze` snapshot at time of the cited runs was not re-captured for this artifact — see `MISSING_EXPERIMENTS.md` T2.3).
- Two additional **isolated** virtualenvs exist for conflicting dependency stacks, confirmed present on disk:
  - `/scratch/pkdas/IEEE_healthcomm_workshop/gpu_classifier_test/` — cuML GPU-classifier environment, bridged via subprocess.
  - `/scratch/pkdas/IEEE_healthcomm_workshop/hear_extractor/` — isolated TF/Keras environment for the HeAR foundation model, bridged via subprocess.
- Subprocess bridges confirmed (by code inspection earlier this session) to inherit `os.environ` rather than replace it — a real bug of the opposite kind (env replacement silently stripping `CUDA_VISIBLE_DEVICES`) was found and fixed in the HeAR bridge earlier in this project's history.

## Package versions — NOT independently re-verified

Encoder checkpoints used (from HuggingFace, as referenced in `src/sleep_quadnet/features.py`): `facebook/hubert-base-ls960`, `microsoft/wavlm-base`, `microsoft/wavlm-large`, `facebook/wav2vec2-base`, `facebook/data2vec-audio-base-960h`, plus a Data2Vec vision variant applied to rendered spectrograms, and `google/hear`. Exact pinned version numbers of `transformers`/`torch`/`librosa`/`scikit-learn`/`cuml`/`xgboost` were not re-captured in this pass. **This is a genuine reproducibility gap**, flagged in `REVIEWER_AUDIT.md` and `MISSING_EXPERIMENTS.md` (T2.3).

## Seeds — reproducible

Fold assignment and classifier random seeds are set per the project's established convention (validation-tuned hyperparameter selection, deterministic fold assignment by subject ID). Exact seed values are stored per-run in each `completion.json` alongside its metrics (visible in the `MASTER_RESULTS.csv` `run_id`/`experiment_key` fields, which are content-addressed hashes of the full run configuration including seed) — **reproducible by inspecting any individual `completion.json`**, not independently re-listed in aggregate here.

## Split generation — reproducible

5 subject-disjoint folds; for fold *f*, test = fold *f*, validation = fold (*f*+1 mod 5), train = remainder. Verified (this session, multiple points) via a **runtime assertion on every fold-run** checking pairwise disjointness of train/validation/test subject ID sets — not merely at fold-file construction time. This is the direct fix adopted after an earlier, deprecated benchmark script was found to have subject-level cross-device leakage (see Limitations, Section 8 of `manuscript.md`).

## Preprocessing — reproducible

Both devices' native 8,000 Hz audio upsampled to 16,000 Hz (SSL-encoder compatibility only, no new spectral content added); peak normalization; a subset of experiments apply a 20–4,000 Hz order-4 Butterworth bandpass. Windows constructed around PSG-annotated apnea/hypopnea events plus duration-matched negatives, aligned across devices via a fitted device-clock drift correction (smartphone as reference, recorder corrected with a piecewise-linear model). All steps exist as concrete, inspectable code in `src/sleep_quadnet/` and `scripts/`.

## Training configuration — reproducible

- 4 classifiers: RBF-SVM, Random Forest, XGBoost, shallow MLP (2 hidden layers, 256/128 units), each with 2 validation-selected hyperparameter candidates.
- Domain-adaptation methods: PCA (384/768/1536-D targets, two-stage fit — tuning-stage source-only, refit-stage including unlabeled target-device validation data post-fix), CORAL (covariance whitening/recoloring using unlabeled target-device validation data only).
- All fusions are simple concatenation, no learned fusion weighting.

## Model-selection rule — reproducible

Primary classifier selected by validation performance (RBF-SVM, per the F05 review-finding refresh); best single encoder selected by validation performance under that classifier (WavLM-large). This two-stage, validation-only selection rule is the one used for the Section 6.2 "best representation" claim and is explicitly named as a specific, reproducible protocol rather than an unstated convention — precisely because Section 6.2 shows the answer changes under the alternative (raw point-estimate) convention.

## Hardware — documented, not independently re-verified this pass

GPU-only compute policy is hardcoded project-wide: every script performing model inference or classifier fit/predict includes an explicit hard GPU-presence check that raises immediately if no GPU is visible (no silent CPU fallback). Exact GPU model/count for the specific historical runs cited was not re-captured in this pass (`MISSING_EXPERIMENTS.md`, T2.3). Deliberately CPU-only-by-design and documented as such (not a policy gap): manifest/CSV/JSON parsing, audio I/O, orchestration code, the `classical`/`odi_hb` handcrafted features, and PCA/CORAL's linear algebra (small, one-shot-per-fold).

## Evaluation code — reproducible

Metrics computed via `src/sleep_quadnet/evaluation.py`; every completed fold-run's exact metric values are content-addressed and stored in that run's `completion.json`, aggregated into this artifact's `MASTER_RESULTS.csv` via `scripts/build_master_results.py` (2,720 rows across 7 experiment families). Significance testing via paired subject-level bootstrap (2,000 resamples), implemented in `ara/src/execution/paired_bootstrap_significance.py` and `scripts/run_statistics.py`.

## Checkpoint handling — reproducible

Content-addressed result caching (`config_hash(...)` over full run configuration, with version-salt bumps whenever underlying computation behavior changes) prevents silent reuse of stale results after a code change — confirmed as the mechanism that caught and forced re-runs after the F05 review-finding fix (candidate encoder/principal-representation lists were refreshed and results/P0_statistics_v2 was produced fresh rather than reusing P0_statistics).

## Known historical incident, disclosed

An earlier, now-deprecated script (`device_robust_sleep_apnea_experiments_v4.py`) produced this project's original submitted headline numbers (F1=94.7%, cross-device F1=90.9%, AUC=0.950) against a since-vanished pre-extracted dataset, and was later found to have a confirmed subject-level cross-device leakage bug. None of the numbers in this artifact derive from that script or that dataset; every number here traces to the current, leakage-asserted pipeline. Disclosed explicitly in Limitations (Section 8 of `manuscript.md`) rather than omitted.
