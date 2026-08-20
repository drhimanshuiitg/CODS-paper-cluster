# Model / Representation Configuration

## Encoder checkpoints and native dimensions
- **Value**:
  - `classical`: handcrafted DSP, 52-D, no checkpoint (CPU-only)
  - `odi_hb`: per-subject CSV lookup, 2-D, no checkpoint, no audio read (CPU-only)
  - `hubert`: `facebook/hubert-base-ls960`, 768-D
  - `wavlm`: `microsoft/wavlm-base`, 768-D
  - `wavlm_large`: `microsoft/wavlm-large`, 1024-D
  - `wav2vec2`: `facebook/wav2vec2-base`, 768-D
  - `data2vec_audio`: `facebook/data2vec-audio-base-960h`, 768-D
  - `data2vec_spectrogram`: `facebook/data2vec-vision-base` (applied to rendered Mel-spectrogram images), 768-D
  - `hear`: `google/hear` (gated HF license), 512-D, isolated TF-Keras venv, fixed 2.0s/16kHz/32,000-sample input only
- **Rationale**: 5 general-purpose speech SSL encoders (chosen for strong SUPERB-benchmark performance per RW02-RW05) plus one health-domain-specific foundation model (HeAR, RW-adjacent, tested as a candidate improvement) plus two non-learned baselines (classical DSP floor, ODI/HB clinical-severity floor).
- **Search range**: not applicable — each is a fixed pretrained checkpoint, never fine-tuned.
- **Sensitivity**: not applicable (frozen).
- **Source**: `src/sleep_quadnet/features.py::MODEL_SPECS`, `FEATURE_DIMENSIONS`

## Fusion representation definitions
- **Value**:
  - `full_fusion` = hubert + wavlm + wav2vec2 + data2vec_audio + data2vec_spectrogram = 3,840-D
  - `full_fusion_v2` = hubert + wavlm + wav2vec2 + data2vec_audio = 3,072-D
  - `full_fusion_plus_hear` = full_fusion + hear = 4,352-D
  - `data2vec_fusion` = data2vec_audio + data2vec_spectrogram = 1,536-D
  - `hubert_odi_hb` = hubert + odi_hb = 770-D
  - `full_minus_{X}` (6 variants) = full_fusion with encoder X's slot removed
- **Rationale**: simple concatenation, no learned fusion weighting/attention/gating — deliberately the simplest possible fusion strategy, so any observed fusion-vs-single-encoder difference is attributable to the encoders' raw signal content rather than a fusion mechanism's own capacity.
- **Search range**: not applicable — each is a fixed named representation.
- **Sensitivity**: high, empirically — C02/C10 found that this simple concatenation strategy specifically does not help (and mildly hurts) cross-device generalization; whether a *learned* fusion mechanism would behave differently is untested (flagged as future work, the "paired-recording contrastive device-invariance head").
- **Source**: `configs/base.yaml:representations`

## HeAR fixed-input clip policy
- **Value**: 32,000 samples (2.0s at 16kHz), center-crop if the source window is longer, zero-pad (centered) if shorter
- **Rationale**: HeAR (a masked-autoencoder ViT-L) has no variable-length input mode; some fixed-length reduction of every manifest window is structurally required.
- **Search range**: not searched — a single fixed policy.
- **Sensitivity**: medium-to-high, plausible but untested — a different reduction strategy (e.g. multiple sub-clips averaged, rather than one center clip) could change HeAR's measured performance (C08); not ruled out as a contributing factor to HeAR's weak result, though the effect size (worst of 14 representations, not just mediocre) makes a clip-policy artifact a less likely sole explanation.
- **Source**: `scripts/extract_hear_features.py`

## Sliding-window epoch definition
- **Value**: 300 seconds (5 minutes), fixed non-overlapping clock grid per subject, independent of annotated-event locations
- **Rationale**: long enough for a stable local desaturation-rate estimate, short enough for genuine within-night localization; deliberately not centered on annotated events (unlike the main manifest's windows), to avoid the annotation-privileged-window problem this experiment (E09) was designed to escape.
- **Search range**: not searched — a single fixed epoch length.
- **Sensitivity**: not empirically ablated; a shorter epoch would increase label noise (fewer events per epoch to estimate a rate from), a longer epoch would reduce temporal localization — untested tradeoff.
- **Source**: `scripts/build_sliding_window_ahi_targets.py` (`EPOCH_SEC = 300.0`)

## Desaturation-event detection parameters (shared by ODI/Hypoxic-Burden and sliding-window ground truth)
- **Value**: drop threshold 3.0% below a 100-second trailing-max rolling baseline, minimum event duration 8 seconds
- **Rationale**: standard portable-oximetry convention for defining a qualifying desaturation event; explicitly documented as "NOT the exact algorithm any specific commercial oximeter uses" (module docstring) — internally consistent for cross-subject comparison within this dataset, not externally calibrated.
- **Search range**: not searched.
- **Sensitivity**: validated indirectly — the resulting per-subject ODI/hypoxic-burden correlate with PSG-annotated event counts at Pearson r=0.83 (ODI) and r=0.61 (hypoxic burden), suggesting the parameters are reasonable, though no sensitivity sweep over the 3%/100s/8s parameters themselves was performed.
- **Source**: `scripts/compute_odi_hypoxic_burden.py`
