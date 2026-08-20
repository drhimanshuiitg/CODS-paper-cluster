# Architecture

## Component graph

```
Raw dual-device audio (R: bedside recorder, S: smartphone) + PSG annotations
+ SpO2/HR/Flow_DR/sleep_stage channels, all record_start-relative-aligned
      |
      v
[Manifest construction]                    metadata/dataset_manifest_aligned.csv
  inputs: raw WAV files, annotation JSON, device-clock-alignment correction
  outputs: one row per (window, device) with audio_paths_json/durations, start/end_sec, label
      |
      +--> [Main-benchmark manifest] (event-centered positive/negative windows)
      |
      +--> [Sliding-window manifest] (build_sliding_window_manifest.py)
              non-annotation-privileged, fixed 5-min clock epochs
      |
      v
[Feature extraction]                       features.py::extract_feature_cache
  inputs: manifest row (audio path/segment), preprocessing spec
  outputs: resumable memory-mapped feature cache (cached_features/{feature}/{preprocessing}/)
  models: HuBERT, WavLM(-large), Wav2Vec2, Data2Vec-audio/spectrogram (transformers, GPU-required)
          HeAR (isolated TF-Keras venv, subprocess bridge, GPU-required, fixed 2s/16kHz clips)
          classical (handcrafted DSP, CPU, no model)
          odi_hb (per-subject CSV lookup, CPU, no model, no audio read at all)
      |
      v
[Representation composition]               configs/base.yaml: representations:
  single encoders (hubert, wavlm, ...) or concatenations (full_fusion, data2vec_fusion, ...)
      |
      v
[Fold splitting]                           evaluation.py::split_indices
  inputs: subject_folds_5cv_aligned.csv, protocol (device pairing)
  outputs: train/val/test row indices, subject-disjoint-asserted
      |
      +--> [PCA path]  advanced.py::run_pca_fold
      |      tuning-stage PCA: source-train-only fit
      |      refit-stage PCA: source-train+val [+ unlabeled target-val, post-fix] fit
      |
      +--> [CORAL path]  advanced.py::run_coral_fold
      |      covariance whitening-recoloring, source train+val + unlabeled target-val fit
      |
      +--> [Corroboration-filtered path]  evaluation.py::filter_uncorroborated
      |      drops uncorroborated positive train/val rows only, never test
      |
      v
[Classifier fit]                           evaluation.py::build_estimator / select_estimator
  random_forest, svm_rbf --> cuML via isolated-venv subprocess bridge (GPUSubprocessEstimator)
  xgboost --> device="cuda", hard GPU-presence guard (_require_gpu)
  mlp --> TorchMLPClassifier, from-scratch PyTorch, hard GPU-presence guard
      |
      v
[Scoring + result persistence]             evaluation.py::run_fold
  outputs: results/{root}/runs/{content_addressed_key}/completion.json + window_predictions.csv.gz
      |
      v
[Significance testing]                     4 near-identical paired-bootstrap scripts
  run_statistics.py (main benchmark), run_pca_fix_significance.py, run_corroboration_significance.py,
  run_ablation_significance.py
  outputs: results/{P0_statistics,P1_statistics_pca_fix,P2_statistics,P0_ablation_statistics}/*.csv
```

## Component notes

- **Manifest construction** is upstream of everything and not re-derived in this ARA; it encodes the device-clock-alignment correction (Smartphone as reference clock, Recorder's piecewise-linear drift correction) established before this session.
- **Feature extraction is the single shared cache layer** every downstream experiment (E01-E09) reads from — a representation is never re-extracted per experiment, only re-composed (concatenated) or re-consumed by a different classifier/split logic.
- **The PCA and CORAL paths are structurally parallel** (both: fit an unsupervised transform using unlabeled target-device validation data + source labeled data, apply to source+test, fit a classifier on the transformed source) — this structural parallel is exactly what made the PCA bug diagnosable (comparing its fit-data scope against CORAL's revealed the discrepancy) and is the architectural basis of claim C05.
- **The sliding-window path reuses the feature-extraction and classifier-fit components unchanged**, only replacing the manifest and adding a self-contained splitting function (`run_sliding_window_severity.py::split_indices`, deliberately separate from `evaluation.py::split_indices` since the sliding-window manifest lacks the `label` column and event-window semantics the main splitter assumes) — an explicit architectural choice to reuse infrastructure rather than fork it.
