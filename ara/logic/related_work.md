# Related Work

## RW01: Tao et al., 2025 (dataset)
- **DOI**: 10.1038/s41597-025-05583-8
- **Type**: imports
- **Delta**:
  - What changed: this project uses Tao et al.'s public multimodal sleep apnea dataset (N=50, synchronized PSG + bedside-recorder audio + smartphone audio) as its entire data source, unmodified except for the project's own device-clock-alignment correction and manifest construction.
  - Why: it is the only publicly available dataset in this space with concurrent, time-aligned dual-device audio, which is the structural precondition for every cross-device experiment in this ARA (E01-E09).
- **Claims affected**: C01-C11 (all — this is the dataset every claim is scoped to)
- **Adopted elements**: raw audio, PSG annotations, SpO2/HR/Flow_DR/sleep_stage channels, device identity metadata

## RW02: Hsu et al., 2021 (HuBERT)
- **DOI**: not specified in paper (IEEE/ACM TASLP)
- **Type**: baseline
- **Delta**:
  - What changed: HuBERT is used unmodified as a frozen feature extractor (`facebook/hubert-base-ls960`, 768-D), never fine-tuned.
  - Why: it is this project's single best-performing representation (C01) and the primary practical-recommendation baseline against which every fusion is compared.
- **Claims affected**: C01, C02, C03, C09, C10
- **Adopted elements**: pretrained checkpoint weights only; masked-hidden-unit-prediction pretraining objective not reproduced or modified

## RW03: Chen et al., 2022 (WavLM)
- **DOI**: not specified in paper (IEEE JSTSP)
- **Type**: baseline
- **Delta**:
  - What changed: both WavLM-base and WavLM-large used as frozen feature extractors; WavLM-large is the strongest matched-device representation found (0.6114 BA) but has the largest matched-to-cross-device gap (0.071) of any real (non-chance) representation.
  - Why: WavLM's masked-speech-denoising pretraining objective is explicitly designed for noise/channel robustness, motivating its inclusion as a plausible candidate for better device-invariance — a hypothesis this project's results (largest gap, not smallest) do not support.
- **Claims affected**: C01, C02, C03
- **Adopted elements**: pretrained checkpoint weights only

## RW04: Baevski et al., 2020 (Wav2Vec 2.0)
- **DOI**: not specified in paper (NeurIPS)
- **Type**: baseline
- **Delta**: used unmodified as a frozen feature extractor, one component of `full_fusion`.
- **Claims affected**: C02, C10
- **Adopted elements**: pretrained checkpoint weights only

## RW05: Baevski et al., 2022 (Data2Vec)
- **DOI**: not specified in paper
- **Type**: baseline
- **Delta**: both audio and vision (spectrogram-input) variants used, concatenated as `data2vec_fusion`; the vision variant is applied to rendered Mel-spectrogram images rather than raw audio, a repurposing beyond Data2Vec's original audio use case.
- **Claims affected**: C02, C10
- **Adopted elements**: pretrained checkpoint weights only (both audio and vision variants)

## RW06: Moummad et al., 2023 (domain-adversarial training for cross-device acoustic monitoring)
- **DOI**: not specified in paper (EUSIPCO)
- **Type**: bounds
- **Delta**:
  - What changed: this project does not implement domain-adversarial training at all (it is listed only as future work in `paper/conference_101719.tex`'s conclusion); Moummad et al.'s approach represents the class of representation-level intervention this project's own results (C05-C07 negative results for post-hoc corrections) argue is likely necessary.
  - Why: cited as the nearest prior-art direction for the concrete next step this project's findings point toward, not as a baseline this project's numbers are compared against.
- **Claims affected**: none directly (referenced in problem.md's Key Insight and future-work framing, not proof of any claim)
- **Adopted elements**: none yet — a direction, not an implementation, in this project

## RW07: Ben-Israel et al., 2012 (snore-based AHI estimation)
- **DOI**: not specified in paper (Sleep journal)
- **Type**: bounds
- **Delta**: reports AUC 0.87 for AHI>=10 classification using hand-engineered snoring-signal features on (implicitly) a single-device setup; this project's task (binary event-window classification, bidirectional cross-device) is not the same task or metric, so no direct numeric comparison is drawn.
- **Claims affected**: none directly — cited as prior-art context in `paper/conference_101719.tex`'s introduction, not as an experimental baseline this ARA's experiments reproduce or compare against
- **Adopted elements**: none

## RW08: Balagopalan et al., 2021 (frozen SSL features vs. fine-tuned/feature-engineered baselines, Alzheimer's prediction)
- **DOI**: not specified in paper (Interspeech)
- **Type**: bounds
- **Delta**: establishes, in a different clinical-audio domain, that frozen HuBERT features can outperform hand-engineered feature baselines — a precedent this project's `classical` vs. SSL-encoder comparison (E01, all SSL encoders outperform `classical`'s 0.4988 cross-device BA) is consistent with, though independently re-derived rather than assumed.
- **Claims affected**: C01 (supporting precedent, not proof)
- **Adopted elements**: none (methodological precedent only)

## Additional cited background (no specific technical delta against this project's methods; captured for citation-footprint completeness)
- Benjafield et al. 2019, Young et al. 2002, Caples et al. 2005, AASM 2009, Statista 2024 — OSA prevalence/burden and PSG-cost background, motivating the acoustic-screening problem framing generally, not any specific method in this ARA.
- Perez-Padilla et al. 1993, Fiz et al. 1996, Karunajeewa et al. 2008, Dafna et al. 2013, Cavusoglu et al. 2009, Nakano et al. 2012, Rossi et al. 2023, Shi et al. 2018, Bsoul et al. 2011 — earlier acoustic/snoring-based OSA screening work (hand-engineered features through early deep learning), establishing the field this project's SSL-encoder approach follows; none contribute a technical component reused in this codebase.
- Yang et al. 2021 (SUPERB benchmark) — cited as the general-domain benchmark on which HuBERT/WavLM/Wav2Vec2 report strong performance, motivating their selection as candidate encoders here.
- Mesaros et al. 2021, Perna & Tagarelli 2019 — sound-event-detection and respiratory-sound-classification precedents cited in a currently-commented-out portion of `paper/conference_101719.tex` (not part of the compiled manuscript's active text as of this ARA's compilation, but present in the source and worth preserving here since the compiler's rule is not to silently drop citation footprint).
