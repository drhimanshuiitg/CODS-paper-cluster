# Heuristics

## H01: Include unlabeled target-device validation data in PCA's refit, not its tuning stage
- **Rationale**: The tuning stage only selects hyperparameters (which dimension/classifier-setting performs best on validation) — including target data there would risk letting hyperparameter choice implicitly peek at target-domain structure across many candidate evaluations. The refit stage produces the actual transform used at test time and has no such repeated-comparison risk, so it can safely absorb the target-aware fix.
- **Sensitivity**: high — this exact split (tuning source-only, refit target-aware) is what makes the fix leakage-safe; applying it to tuning instead would need separate justification and re-validation.
- **Bounds**: applies only to cross-device protocols (`target_devices` non-empty); silently a no-op for matched-device protocols.
- **Code ref**: [../../../src/sleep_quadnet/advanced.py](../../../src/sleep_quadnet/advanced.py) (`run_pca_fold`)
- **Source**: `results/P1_statistics_pca_fix/pca_fix_vs_baseline.csv` (validates the heuristic empirically, 30/60 significant positive, 0/60 significant negative)

## H02: Salt cache keys with a version tag whenever a fix changes what a "complete" run means
- **Rationale**: This project's result caching is content-addressed by `config_hash(...)`; a completion.json produced under old, buggy logic (e.g. pre-fix PCA collapse) would otherwise be indistinguishable from a correct one under naive resumability, silently preventing the fix from ever re-running.
- **Sensitivity**: high — omitting the version bump after the PCA fix would have made the fix invisible to the pipeline's own resumability logic.
- **Bounds**: applied at `dimension_control_v2 -> v3` for this fix; the same pattern (`gpu_v1` tag, `corrob_filtered_v1` tag) recurs at every GPU-migration and ablation-scoping point in this codebase.
- **Code ref**: [../../../src/sleep_quadnet/advanced.py](../../../src/sleep_quadnet/advanced.py) (`key = config_hash("dimension_control_v3", ...)`)
- **Source**: inline code comment at the `dimension_control_v3` key construction

## H03: Never let a GPU-accelerated code path silently fall back to CPU
- **Rationale**: Confirmed empirically twice this session — xgboost's `device="cuda"` with no GPU present only warns and silently computes on CPU; a subprocess `env=` dict built from scratch (rather than inheriting the parent environment) silently strips `CUDA_VISIBLE_DEVICES`, making a correctly-GPU-allocated job compute on CPU with zero error for 5+ minutes before being caught by manual inspection.
- **Sensitivity**: high — a silent fallback changes the actual computation being benchmarked (CPU vs. GPU numerics/scheduling) without changing any recorded metadata field that would flag it.
- **Bounds**: applies to every classifier (`_require_gpu`, inline `torch.cuda.is_available()` checks) and every isolated-venv subprocess bridge (HeAR's `hear_worker.py` checks `tf.config.list_physical_devices("GPU")`).
- **Code ref**: [../../../src/sleep_quadnet/evaluation.py](../../../src/sleep_quadnet/evaluation.py) (`_require_gpu`, `TorchMLPClassifier`)
- **Source**: `GPU_INSTRUCTIONS.md` ("GPU-Only Compute Policy" section, added this session after the incident)

## H04: Match a new representation's PCA target dimensions to its native dimensionality
- **Rationale**: `run_pca_fold` requires each target dimension to be strictly less than `min(n_samples, n_features)`; requesting a dimension equal to a representation's own native dimension (e.g. 1536 for the exactly-1536-D `data2vec_fusion`) is not a real reduction and is correctly rejected — but only checked at run time, not at submission time.
- **Sensitivity**: medium — a mismatch crashes the very first combo of a job (fast, cheap failure) rather than corrupting results, but still wastes a GPU-job submission if not caught.
- **Bounds**: `wavlm_large` (1024-D) and `data2vec_fusion` (1536-D) both needed a reduced `--dimensions 768,384` (dropping their would-be-native or larger target) rather than the default `1536,768,384`.
- **Code ref**: [../../../scripts/run_dimension_control.py](../../../scripts/run_dimension_control.py) (`--dimensions` CLI argument, added this session)
- **Source**: job 1542's crash log (`ValueError: PCA dimension 1536 invalid for train shape (11805, 1536)`)

## H05: Center-crop-or-zero-pad variable-length windows to a foundation model's fixed native input size
- **Rationale**: HeAR has no variable-length or streaming input mode (unlike the transformers-based encoders, which chunk-and-pool internally via `_audio_ssl_vector`); some fixed-length reduction is unavoidable, and center-crop-or-pad is the simplest well-defined choice that preserves the temporal center of each window.
- **Sensitivity**: medium — this is a real, documented information-loss step (not a hidden approximation), flagged in the feature cache's own metadata (`clip_policy: "center_crop_or_pad"`) so downstream consumers can see it was applied.
- **Bounds**: applies per-window in the main manifest; the sliding-window manifest's much-longer (300s) epochs were never run through HeAR at all in this session (only through HuBERT, which does support variable-length input via chunking).
- **Code ref**: [../../../scripts/extract_hear_features.py](../../../scripts/extract_hear_features.py) (`to_fixed_clip`)
- **Source**: `extract_hear_features.py` module docstring

## H06: Drop (never silently clip) sliding-window epochs that run past a device's actual available audio
- **Rationale**: Epochs were binned off the SpO2 channel's own timeline, which does not necessarily end at the same instant as a given device's audio recording; silently clipping such an epoch to whatever audio does exist would produce an inconsistently-shorter-than-labeled clip without any record of the discrepancy, while dropping it entirely keeps every remaining epoch's audio content matching its labeled duration exactly.
- **Sensitivity**: medium — affected 441/9,950 candidate rows (4.4%), concentrated near the end of each recording.
- **Bounds**: computed per (subject, device) pair independently, since R and S devices for the same subject can have different actual recording durations.
- **Code ref**: [../../../scripts/build_sliding_window_manifest.py](../../../scripts/build_sliding_window_manifest.py) (`available_duration_sec`, the `skipped_past_audio_end` branch)
- **Source**: job 1548's crash log (`ValueError: Window length mismatch: got 463360, expected 2400000`)

## H07: Match a device-alignment-sensitive join on the smartphone's reference-clock rows only, then propagate via a shared logical-window id
- **Rationale**: The Recorder device's timestamps carry an additional piecewise-linear clock-drift correction not present in the Smartphone's (reference-clock) timestamps, so a direct float-equality (or even small-tolerance) time-based join between the SpO2-corroboration audit's event timings and the main manifest matched only ~50% of rows when attempted against both devices; matching only S-device rows by time, then propagating the corroboration flag to both devices' rows via their shared `logical_window_id`, reaches 100% match rate.
- **Sensitivity**: high — this is a device-clock-alignment subtlety specific to this dataset's dual-device design, not a generic join-tolerance issue (the first fix attempt, widening the float tolerance to 0.01s, did not resolve it).
- **Bounds**: applies to any future join between SpO2-timeline-derived events and manifest windows; does not apply within a single device's own internal timing.
- **Code ref**: [../../../scripts/build_window_corroboration.py](../../../scripts/build_window_corroboration.py)
- **Source**: investigation notes from this session (float-precision hypothesis tested and rejected before the clock-alignment hypothesis was confirmed)

## H08: Hoist a shared exemption set to one module-level constant instead of letting call sites redeclare their own copy
- **Rationale**: `benchmark_efficiency.py` had its own local copy of "features that don't need a GPU" that only included `classical`, missing `odi_hb` (added to the canonical set in `features.py` later); the two copies silently drifted apart, which would have made the efficiency benchmark incorrectly *require* a GPU for a feature that doesn't need one, had it ever been invoked with `--feature odi_hb`.
- **Sensitivity**: low-to-medium — caught by direct code audit, not by a failing run (the drifted call site was never actually exercised with `odi_hb` this session), so the practical impact was latent rather than observed.
- **Bounds**: general pattern — any project-wide invariant (GPU exemption sets, classifier lists, protocol lists) duplicated across multiple files is a drift risk; this project now hoists this specific one to `features.NO_GPU_NEEDED`.
- **Code ref**: [../../../src/sleep_quadnet/features.py](../../../src/sleep_quadnet/features.py) (`NO_GPU_NEEDED`)
- **Source**: direct project-wide grep audit for GPU-fallback risk, conducted this session per an explicit "no CPU code anywhere" review request
