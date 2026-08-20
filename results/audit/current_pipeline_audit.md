# Current Sleep-QuadNet Pipeline Audit

Audit date: 2026-08-17 (Asia/Calcutta)  
Authoritative plan: `Sleep_QuadNet_Workshop_Experiment_Plan.md`  
Existing implementation audited: `device_robust_sleep_apnea_experiments_v4.py` and `config_device_robust_v4.yaml`

## Executive finding

The supplied v4 runner cannot be executed validly on the available dataset without additional data-manifest/windowing code. It expects pre-extracted labelled WAV clips whose names contain class keywords, while the available dataset contains continuous overnight recordings named only by subject and device. No prior extracted clips, feature caches, results, checkpoints, logs, SLURM jobs, or clip-generation code are present in the project or accessible home workspace.

The current v4 cross-device implementation is also not patient-disjoint under its supplied configuration: `strict_patient_disjoint_cross_device` is `false`, and each cross-device protocol trains on every subject in the source domain and tests on every subject in the target domain. Paired subjects therefore occur in both sets through opposite devices.

## 1. Dataset paths and subjects

- Dataset root specified by cluster policy: `/scratch/pkdas/IEEE_healthcomm_workshop`
- Raw corpus used for the audit: `/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/Data`
- Dataset utility supplied with the corpus: `/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/osa_data_eng.py`
- Subject IDs: `01` through `50`
- Unique annotation subjects: 50
- Annotation files: 50
- Phone WAVs: 48
- Recorder WAV segments: 98 across 45 subjects
- Subjects with both phone and recorder audio: 43
- Phone-only subjects: `12, 22, 23, 24, 25`
- Recorder-only subjects: `11, 17`
- Paired subjects suitable for controlled R/S comparisons: `01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 13, 14, 15, 16, 18, 19, 20, 21, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50`

This differs from the plan's nominal description of 50 synchronized dual-device subjects. The core device benchmark must use the 43 paired subjects unless missing device recordings are supplied later.

## 2. Devices and internal labels

- Smartphone domain, planned label `S`: files named `<subject>_phone.wav`
- Professional/clinical recorder domain, planned label `R`: files named `<subject>_recorder.wav` or `<subject>_recorder_<segment>.wav`
- Recorder files are split into sequential segments for most subjects and must be treated as a logical concatenation in numeric order.
- The current v4 config uses internal names `smartphone` and `recorder`; these map directly to `S` and `R`.

## 3. Audio sampling and coverage

- All 146 WAV headers are Microsoft PCM, 16-bit, mono, 8,000 Hz.
- Existing v4 preprocessing resamples to 16,000 Hz with `librosa.resample`.
- Existing code does not explicitly set the resampling algorithm, so anti-aliasing behavior depends on the installed librosa version/default resampler. This must be pinned and logged in the revised pipeline.
- Phone duration range: 21,299.160 to 62,744.160 seconds; median 37,945.428 seconds.
- Concatenated recorder duration range among available recorder subjects: 27,625.728 to 57,342.080 seconds; median among all 50 subjects is 38,300.128 seconds when missing recorder subjects are included as zero.
- Paired phone/recorder absolute duration differences range from 1.856 to 37,573.392 seconds (median 35.344 seconds). Several early subjects have materially truncated phone coverage.
- For paired modeling, windows must lie inside the common phone/recorder time coverage for the subject. Otherwise matched and cross-device protocols would not compare the same respiratory event.

## 4. Annotation schema and labels

- Each subject has `<subject>_annotation.json` with keys `record_start`, `awake_intervals`, and `events`.
- Each event contains `event_type`, misspelled source key `evnet_start`, `event_duration`, and `sleep_stage`.
- Event starts are absolute seconds-of-day; the relative audio start is `evnet_start - record_start`.
- Total events: 13,455
- `osa`: 4,539
- `hypo`: 8,916
- Event-duration range: 0.9 to 126.0 seconds; median 21.9 seconds; mean 24.59 seconds.
- Events marked sleep stage `W`: 531.
- No event has a negative relative start.

The supplied v4 runner does not read these annotations. Its positive rule is filename matching against `osa`, `csa`, `msa`, `mixed`, `hypo`, or `apnea`; its negative rule is filename matching against `normal`. Since the raw WAV names contain neither class, the current scanner finds zero labelled clips.

No existing code or manuscript was found that defines how normal windows were originally sampled or whether wake-stage respiratory events were excluded. These details are not recoverable from the current project and must be documented as an explicit revised-pipeline decision before interpreting results.

### Annotated respiratory events per subject

| Subject | Events | Subject | Events | Subject | Events | Subject | Events | Subject | Events |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 86 | 11 | 209 | 21 | 165 | 31 | 122 | 41 | 260 |
| 02 | 574 | 12 | 89 | 22 | 221 | 32 | 104 | 42 | 342 |
| 03 | 229 | 13 | 165 | 23 | 314 | 33 | 341 | 43 | 55 |
| 04 | 38 | 14 | 220 | 24 | 221 | 34 | 270 | 44 | 221 |
| 05 | 435 | 15 | 203 | 25 | 203 | 35 | 455 | 45 | 429 |
| 06 | 280 | 16 | 122 | 26 | 481 | 36 | 136 | 46 | 277 |
| 07 | 58 | 17 | 322 | 27 | 455 | 37 | 397 | 47 | 231 |
| 08 | 319 | 18 | 155 | 28 | 184 | 38 | 381 | 48 | 397 |
| 09 | 306 | 19 | 303 | 29 | 81 | 39 | 403 | 49 | 317 |
| 10 | 227 | 20 | 503 | 30 | 419 | 40 | 198 | 50 | 532 |

## 5. Existing clip/window extraction logic

- The v4 runner assumes each input WAV is already one classification clip.
- It loads the complete WAV and optionally truncates from the beginning to `audio.max_seconds`.
- Supplied config: `max_seconds: null`, so the complete supplied clip would be used.
- No window length, hop, overlap, event-centering, padding, or negative-window sampling exists in the code.
- No extracted clip dataset is present.
- Therefore the original fixed/variable clip duration and extraction overlap are unknown.

Current materialized window counts are consequently:

- Positive windows: 0
- Negative windows: 0
- Windows per subject: 0

The raw event table above is not yet a model-window table.

## 6. Existing split logic and leakage audit

### In-domain and combined protocols

- Uses one `GroupShuffleSplit` with `test_size=0.25`, seed 42, grouped by `patient_id`.
- No five-fold evaluation.
- No subject-disjoint validation set.
- No validation-only hyperparameter selection.

### Cross-device protocols

- Supplied config sets `strict_patient_disjoint_cross_device: false`.
- Source training uses every source-device clip; target testing uses every target-device clip.
- Because the devices are paired, the same subjects occur in source training and target testing: patient leakage.
- Setting the existing flag to `true` is not a valid repair: it removes all target rows for paired subjects after training on all source subjects and can leave no controlled paired test set.

### Required correction

One fixed five-fold assignment must be generated for the 43 paired subjects. For fold `k`, test subjects are fold `k`, validation subjects must be selected only from the remaining subjects, and training excludes both. The identical subject identities and time windows must be used for R and S. Scaling, PCA, tuning, and any learned alignment must be fitted on training subjects only.

## 7. Random seeds

- Supplied global seed: 42
- Python, NumPy, PyTorch, and CUDA seeds are set by the runner.
- The only current split seed is also 42.
- Classifier random state is 42 where supported.
- No deterministic-algorithm setting or environment capture exists.

## 8. Existing preprocessing

### Common audio loader

1. Read with SoundFile.
2. Average channels if multichannel (the available corpus is mono).
3. Cast to float32.
4. Resample to 16 kHz when needed.
5. Optionally truncate from the beginning; disabled in supplied config.
6. Peak-normalize by maximum absolute amplitude; enabled in supplied config.

### Handcrafted branch

- Fourth-order Butterworth band-pass: 20--4,000 Hz.
- Zero-phase `scipy.signal.filtfilt`, with unfiltered fallback on failure.
- Frame/FFT length: 400 samples (25 ms at 16 kHz).
- Hop: 160 samples (10 ms).
- Features: RMS, zero-crossing rate, spectral centroid, bandwidth, rolloff, flatness, and 20 MFCCs.
- Aggregation: mean and standard deviation for every descriptor.
- Dimension: 52.

### Spectrogram branch

- 128 Mel bands.
- FFT: 1,024 samples.
- Hop: 256 samples.
- `fmax`: 8,000 Hz.
- Power spectrogram converted to dB relative to its maximum, min-max normalized, colored with the `magma` map, and converted to an RGB image.

## 9. Existing representations and pooling

- Wav2Vec2: `facebook/wav2vec2-base`, temporal mean of `last_hidden_state`, expected 768-D.
- HuBERT: `facebook/hubert-base-ls960`, temporal mean, expected 768-D.
- WavLM: `microsoft/wavlm-base`, temporal mean, expected 768-D.
- Data2Vec Audio: `facebook/data2vec-audio-base-960h`, temporal mean, expected 768-D.
- Data2Vec Spectrogram: `facebook/data2vec-vision-base` on the rendered Mel image, patch-token mean including every returned token, expected 768-D.
- Existing named `data2vec`: audio and spectrogram concatenation, expected 1,536-D.
- Existing named `ssl_fusion`: Wav2Vec2 + HuBERT + WavLM + combined Data2Vec, expected 3,840-D.

The required separate Data2Vec-Audio and Data2Vec-Spectrogram experiment outputs and leave-one-branch-out fusions are not implemented.

## 10. Existing downstream classifiers

- Random Forest: 300 trees, balanced class weights, unrestricted depth, all CPU cores.
- XGBoost: 300 trees, depth 4, learning rate 0.05, subsample/column sample 0.9, source-training class ratio used as `scale_pos_weight`.
- RBF-SVM: C=1, gamma=`scale`, probability calibration enabled, balanced class weights, standardized features.
- MLP: hidden layers 256/128, alpha 0.0001, maximum 400 iterations, internal early stopping, standardized features.

The runner uses fixed hyperparameters and has no fold-specific validation/tuning policy.

## 11. Existing metrics and outputs

Implemented metrics:

- Accuracy
- Balanced accuracy
- Precision
- Sensitivity/recall
- Specificity
- Binary F1
- Macro-F1
- MCC
- Cohen's kappa
- ROC-AUC
- PR-AUC
- Confusion-matrix counts

Intended outputs include predictions, joblib classifiers, PNG ROC/PR/confusion/calibration plots, comparison CSVs, and LaTeX tables. No such outputs currently exist.

Missing plan-required outputs include fold metrics, fixed fold files, subject-level aggregated predictions, device-gap summaries, 95% confidence intervals, paired bootstrap comparisons, efficiency benchmarks, device-acoustics analyses, preprocessing ablation, dimension control, CORAL, the master experiment log, and publication-ready vector PDFs.

## 12. Feature caching and observed resource use

- The v4 runner implements per-file NumPy caches and whole-matrix NumPy caches beneath its output directory.
- Cache keys include file path, size, integer modification time, feature name, and a feature-config hash.
- Data2Vec is cached only as the already-concatenated 1,536-D vector, so its two branches cannot be ablated from existing caches.
- No feature cache currently exists.
- No previous runtime, CPU RAM, GPU memory, checkpoint, or feature-storage measurement exists; these values must not be guessed.
- The login-node system Python is 3.12.13 and lacks NumPy and the project dependencies. No Conda environment or loaded environment module was found in the initial audit.

## 13. Code/plan mismatches requiring correction

1. Kaggle input/output paths instead of IITG home/scratch paths.
2. Pre-extracted filename-labelled clips are assumed but absent.
3. No annotation-driven window manifest.
4. No fixed subject-disjoint five-fold evaluation.
5. Cross-device patient leakage in the supplied configuration.
6. No subject-disjoint validation set or validation-only classifier selection.
7. Only 43, not 50, subjects have both device recordings.
8. Some paired devices have different recording coverage; not every annotated event is available in both.
9. Data2Vec audio and spectrogram branches are inseparable in the current cache/representation interface.
10. No leave-one-encoder-out fusion set.
11. Failed feature extraction is silently converted to zero vectors, which could create invalid results instead of failing a run.
12. No fold-specific, subject-level predictions or statistical unit enforcement.
13. No efficiency, acoustic-shift, preprocessing, PCA, or CORAL implementation.
14. No master append-only experiment log.
15. No environment or dependency specification.
16. No SLURM job scripts.
17. Existing figures are raster PNG rather than plan-preferred vector PDF.
18. No Git repository/commit is available; runs must log `git_commit=unavailable` unless version control is initialized later.

## 14. Methodological facts that remain unavailable

The following original-pipeline details are not present in any accessible code, config, result, archive, or manuscript and must not be presented as recovered facts:

- Original event clip/window duration policy
- Original overlap/hop between classification clips
- Original normal-window sampling rule
- Whether wake-stage annotated respiratory events were retained
- Original denoising, if any
- Original pre-extracted class balance and per-subject window counts
- Prior GPU/CPU resource observations

The revised implementation must make any replacement choices explicit, deterministic, leakage-free, and auditable rather than attributing them to the unavailable original pipeline.
