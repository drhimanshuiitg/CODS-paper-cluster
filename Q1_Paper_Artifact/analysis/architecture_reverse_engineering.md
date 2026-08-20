# Architecture Reverse-Engineering

Reconstructed from `src/sleep_quadnet/`, `scripts/extract_features.py`, `scripts/train_classifier.py`, and the isolated-venv subprocess bridges (`gpu_classifier_test/`, `hear_extractor/`). This is a description of what the code actually does, not an idealized architecture diagram.

## Stage 1 — Audio preprocessing
Both devices' native 8,000 Hz audio upsampled to 16,000 Hz (SSL-encoder input-rate compatibility only — no new spectral content is added, Nyquist remains 4,000 Hz). Peak normalization applied. A bandpass-filtered variant (20–4,000 Hz, order-4 Butterworth) exists and is used in a subset of experiments. Windows are constructed centered on PSG-annotated events (positive) and duration-matched non-event segments (negative), device-clock-aligned via a fitted drift-correction model.

## Stage 2 — Representation extraction (14 branches)
Each representation is a **frozen, independently-computed feature extractor** — there is no shared trunk or joint training across representations:
- `classical` (52-D): handcrafted MFCC-family + spectral statistics, computed directly (no external model).
- 5 frozen general-purpose SSL speech encoders (HuBERT-base, WavLM-base, WavLM-large, Wav2Vec2-base, Data2Vec-audio-base) — each loaded from HuggingFace, mean-pooled over the window, no fine-tuning.
- 1 frozen SSL vision-encoder variant (Data2Vec applied to rendered Mel-spectrogram images) — a genuinely different input modality (image, not waveform) despite deriving from the same audio.
- `hear` (512-D): Google's health-acoustic foundation model, extracted via an isolated TF/Keras venv bridged through a subprocess that inherits (not replaces) the parent environment — this specific env-inheritance detail was the subject of a previously-fixed bug (`CUDA_VISIBLE_DEVICES` stripped by env replacement).
- `odi_hb` (2-D): a per-subject-constant clinical severity feature (Oxygen Desaturation Index, Hypoxic Burden), computed from raw SpO2, not from audio at all.
- 5 fusion variants: plain concatenation of subsets of the above (`data2vec_fusion`, `full_fusion`, `full_fusion_v2`, `full_fusion_plus_hear`, `hubert_odi_hb`). **No learned fusion weighting, gating, or attention mechanism exists anywhere in the codebase.** This is confirmed by direct code inspection, not inferred from naming — the fusion step is a NumPy/pandas concatenation.

## Stage 3 — Domain adaptation (optional, applied post-extraction)
- **PCA**: fold-local, two-stage fit (tuning-stage on source-device training data; refit-stage — post-fix — additionally including unlabeled target-device validation data, matching CORAL's scope). Target dimensionalities 384/768/1536.
- **CORAL**: covariance whitening-and-recoloring, fit using unlabeled target-device validation subjects only (never test data/labels).
Both operate on already-extracted frozen features; neither modifies the encoders themselves.

## Stage 4 — Classification (4 branches)
RBF-SVM, Random Forest, XGBoost, and a shallow 2-hidden-layer (256/128 units) MLP, each independently fit per fold/representation/protocol with 2 validation-selected hyperparameter candidates. cuML implementations are used for GPU acceleration where available, bridged from an isolated venv (`gpu_classifier_test/`) via subprocess, inheriting the parent environment.

## Stage 5 — Evaluation protocols (5 branches)
R_R, S_S (matched-device); R_S, S_R (bidirectional cross-device); RS_RS (pooled-device train+test). Implemented as distinct manifest-filtering + train/test-assignment logic in `src/sleep_quadnet/evaluation.py`, not distinct code paths — the same downstream classifier-fitting code runs for every protocol.

## What is explicitly NOT in the architecture

Per the master prompt's instruction to check whether claimed components have actual ablation evidence:
- **No learned fusion mechanism** — the ablation (Section 6.4) therefore cannot and does not claim to isolate a "fusion contribution"; it isolates individual-encoder contribution to a concatenation, which is a different and narrower question.
- **No attention mechanism, no cross-modal alignment layer, no domain-adversarial component, no contrastive device-invariance objective** — none of these exist in the evaluated system. Any manuscript language implying otherwise would be fabrication; `manuscript.md` Section 4 explicitly disclaims a "novel architecture" framing for this reason.
- **No end-to-end fine-tuning of any encoder** — all SSL/foundation-model representations are frozen; only the downstream classifier is trained.

## Correspondence to `ara/logic/solution/architecture.md`

This reconstruction is consistent with and cross-references the ARA's own `architecture.md`/`algorithm.md`/`constraints.md` documentation, produced independently during the earlier ARA-compilation phase of this project — no contradictions were found between the two reconstructions during this audit.
