# Candidate Titles (internal — pick the one matching the evidence)

1. ~~Sleep-QuadNet: A Multi-Encoder Fusion Architecture for Device-Robust Sleep Apnea Screening~~ — rejected; the evidence does not support "device-robust" or that fusion is the contribution.
2. ~~Device-Invariant Representation Learning for Acoustic Sleep Apnea Detection~~ — rejected; no invariance mechanism was built or evaluated.
3. **On the Cross-Device Fragility of Self-Supervised Acoustic Representations for Sleep Apnea Screening, and a Simple Mitigation** — selected. States the actual finding (fragility, not robustness), names the actual mitigation evaluated (pooled training), and does not overclaim a novel architecture.
4. ~~A Comprehensive Benchmark of Foundation Models for Sleep Apnea Detection~~ — rejected; "comprehensive" and "foundation models" as headline overstate the framing relative to the device-shift focus.
5. ~~Toward Deployable Acoustic OSA Screening: Lessons from Cross-Device Failure~~ — reasonable alternative, more clinical framing; not selected because "deployable" is not demonstrated (Limitations, §8).

**Selected title:** *On the Cross-Device Fragility of Self-Supervised Acoustic Representations for Sleep Apnea Screening, and a Simple Mitigation*

---

# Abstract

**Problem.** Acoustic screening for obstructive sleep apnea (OSA) from consumer microphones is a promising low-cost alternative to polysomnography, but almost all prior work reports only matched-device accuracy. **Gap.** Whether representations learned or extracted on one recording device transfer to a different device — the realistic deployment scenario — is rarely tested directly, and no prior work on this task has, to our knowledge, quantified the acoustic domain shift between devices at the signal level or examined it in the representation's own embedding space. **Approach.** Using a public dataset of two devices (a clinical bedside recorder and a consumer smartphone) recording the same subjects concurrently in the same room (N=41–50), we (i) directly measure acoustic domain shift between devices via six window-level signal statistics, (ii) benchmark 14 audio representations — a handcrafted baseline, five frozen self-supervised speech encoders, a health-acoustic foundation model, a clinical severity feature, and eight fusions — across four classifiers and five evaluation protocols (matched-device, bidirectional cross-device, and pooled-device training), under subject-disjoint 5-fold cross-validation with paired-bootstrap significance testing, and (iii) visualize what the strongest encoder's embedding space actually separates. **Results.** Devices differ with near-maximal effect size on every acoustic statistic tested (|Cliff's δ| = 0.54–1.00, p < 1e-30). Every representation shows a real, mostly significant matched-to-cross-device accuracy drop (mean 4.7 balanced-accuracy points). In the strongest encoder's frozen embedding space, recording device is a substantially more separable signal than the clinical label itself (silhouette 0.145 vs. 0.007). No representation — including a 3,840-dimensional five-encoder fusion — is confirmed significantly better than the others once evaluated under the project's own validation-selected methodology. Of two post-hoc domain-adaptation corrections, one (PCA) was salvageable after correcting an implementation-scope bug; the other (CORAL) failed regardless of scope. Simply pooling both devices' data at training time, with no architectural change, closed the great majority of the observed gap for every representation tested. Three further candidate interventions (label-quality filtering, the health-acoustic foundation model, a clinical severity feature) did not help and are reported as negative results. **Conclusion.** The central obstacle to acoustic OSA screening across devices is not representation choice but device identity itself being trivially separable in current frozen encoders; among the interventions tested here, training-data composition — not encoder or fusion sophistication — is the most effective lever available without new model training.

---

# 1. Introduction

Obstructive sleep apnea affects an estimated one billion adults worldwide, with up to 80% of moderate-to-severe cases undiagnosed, largely because the diagnostic gold standard — attended polysomnography — is expensive, resource-intensive, and unscalable for population screening [1,2,4]. Passive acoustic monitoring from consumer microphones has been proposed as a low-cost screening alternative for three decades [5–12], and the field has recently shifted from hand-engineered spectral features to frozen self-supervised speech encoders (HuBERT, WavLM, Wav2Vec 2.0, Data2Vec) [13–16], which capture richer paralinguistic structure without task-specific labels.

A structural weakness runs through nearly all of this literature: deployment necessarily involves heterogeneous recording hardware — a clinical device in one setting, a patient's own smartphone in another — yet almost every reported result is a matched-device evaluation, where the same device (often the same recording session) supplies both training and test data. This conflates "the model works" with "the model works on this specific microphone," and the difference is not a minor technicality: recording hardware differs in frequency response, self-noise, gain characteristics, and typical placement, all of which alter the acoustic signal independent of the underlying physiology. Whether the field's move to self-supervised representations has actually solved this problem, made it worse, or left it untouched has — to our knowledge — not been directly measured on this task.

This paper measures it directly, using a dataset with genuinely concurrent, time-aligned dual-device recordings (Section 3), which is the structural precondition for isolating device effects from confounds like different subjects, different sessions, or different recording environments. Our contributions:

1. **Direct, signal-level quantification of acoustic device shift** (Section 6.1) — not inferred from downstream accuracy alone, but measured on the raw audio via six standard acoustic statistics with a non-parametric effect-size test.
2. **A subject-disjoint, statistically validated benchmark** across 14 representations × 4 classifiers × 5 protocols (including a pooled-device training protocol not evaluated in prior cross-device acoustic-screening work), with paired-bootstrap significance testing applied to every comparative claim rather than raw point estimates.
3. **A representation-space explanation for the cross-device gap** (Section 6.1) — to our knowledge the first visualization, for this task, of what a frozen SSL encoder's embedding space actually separates most easily: device identity, not the clinical label.
4. **A methodological finding about evaluating "best representation" claims**: two legitimate measurement conventions disagree on which encoder ranks highest, and the more rigorous one finds no representation significantly better than any other (Section 6.2) — a caution against the common practice of reporting only a point-estimate leaderboard.
5. **A distinction between a correctable domain-adaptation implementation bug and a genuine method limitation** (Section 6.3), and the identification of pooled-device training as a simple, architecture-free mitigation that closes most of the measured gap (Section 6.3) — the paper's most actionable finding.
6. **Three honestly-reported negative results** (Section 6.5) and a first, explicitly labeled-as-preliminary attempt at a non-annotation-privileged severity target (Section 6.6).

---

# 2. Related Work

**Acoustic OSA and respiratory-sound screening.** Perez-Padilla et al. [7] and Fiz et al. [8] characterized snoring-sound acoustic properties; Karunajeewa et al. [9], Dafna et al. [10], Ben-Israel et al. [11] (AUC 0.87 for AHI≥10), and Cavusoglu et al. [12] developed hand-engineered-feature classifiers for apnea/hypopnea-related sound events. Nakano et al. [18] and Rossi et al. [19] extended this toward deep learning; Shi et al. [20] and Bsoul et al. [21] demonstrated smartphone-specific feasibility. None of this prior work evaluates a model trained on one device against a *different*, held-out recording device — the gap this paper addresses.

**Self-supervised speech representations.** Wav2Vec 2.0 [13] and HuBERT [14] established masked/contrastive self-supervised pretraining as a strong general-purpose speech feature extractor, consistently near the top of the SUPERB benchmark [22]; WavLM [15] adds a denoising objective explicitly targeting noise/channel robustness; Data2Vec [16] unifies the masked-prediction objective across modalities. Balagopalan et al. [23] showed frozen HuBERT features can outperform hand-engineered baselines in a different clinical-audio task (Alzheimer's prediction from speech) — a precedent this paper's classical-vs-SSL comparison (Section 6.2, Table 3) is consistent with.

**Domain adaptation and cross-device robustness.** Post-hoc statistical correction of frozen features — dimensionality reduction, covariance alignment — is a standard cheap alternative to encoder retraining; we evaluate two such techniques directly (Section 6.3). Moummad et al. [24] applied domain-adversarial training for cross-device acoustic monitoring in a different acoustic-sensing task; we do not implement this here (Section 9) but our negative findings for post-hoc correction (Section 6.3) motivate it as the more likely next step.

**Health-acoustic foundation models.** HeAR [25] is a masked-autoencoder pretrained on ~174,000 hours of health acoustics (coughs, breathing) specifically, rather than general speech. We test, to our knowledge for the first time on this task, whether this domain-matched pretraining transfers better than general-purpose speech encoders (Section 6.5).

**Dataset.** All experiments use the public dual-device dataset of Tao et al. [17] — synchronized Type-I PSG, bedside clinical-recorder audio, and smartphone audio from 50 subjects, and the only public dataset we are aware of with genuinely concurrent, time-aligned dual-device audio for this task.

---

# 3. Materials and Dataset

## 3.1 Population and Devices

| Property | Value |
|---|---|
| Source | Tao et al., 2025 [17], public dataset |
| Subjects (total) | 50 |
| Subjects with reliable dual-device coverage (used here) | 41 |
| Subjects excluded (single-device-only or unreliable alignment) | 9 (5 smartphone-only, 2 recorder-only, 2 excluded for alignment quality) |
| Device 1 | Bedside clinical digital recorder (Newamy V03) |
| Device 2 | Consumer smartphone (OPPO Reno8) |
| Native sample rate (both devices) | 8,000 Hz |
| Recording setting | Concurrent, same room, same subject, same night |
| Reference annotation | Type-I polysomnography (PSG), clinician-scored apnea/hypopnea events |

Both devices are general-purpose ambient microphones recording the same acoustic environment concurrently — this is genuine device/transducer-hardware shift (differing frequency response, self-noise floor, gain, and placement conventions), not a difference in anatomical recording site, patient population, or recording session. This distinction matters for interpreting every result in this paper (Section 6.1 quantifies it directly) and is stated explicitly because it is not always made explicit in the domain-shift literature.

## 3.2 Preprocessing

Both devices' native 8,000 Hz audio (Nyquist 4,000 Hz) is upsampled to 16,000 Hz only for compatibility with SSL encoders pretrained at that rate — this adds no new spectral information beyond the original capture bandwidth. Audio is peak-normalized; a subset of experiments additionally apply a 20–4,000 Hz order-4 Butterworth bandpass. Windows are constructed around PSG-annotated apnea/hypopnea events (positive) and duration-matched non-event segments (negative), aligned across devices via a fitted device-clock correction (smartphone as reference clock; recorder timestamps corrected with a piecewise-linear drift model, verified against the raw data during pipeline development).

## 3.3 Labels and Class Balance

Binary window-level label: apnea/hypopnea event present vs. absent, from PSG annotation. Exact positive/negative counts and per-subject distribution are recorded in `metadata/dataset_manifest_aligned.csv`; the manifest construction balances negatives against positives per subject rather than using the raw, highly imbalanced continuous-recording class ratio (Section 3.2, `analysis/dataset_audit.md` for the full accounting).

---

# 4. Proposed Method (Evaluation Framework, Not a Novel Architecture)

We emphasize at the outset: **this paper's primary contribution is not a novel model architecture.** It is a rigorous evaluation framework and the empirical/representation-level findings it produces (Sections 6–7). Where prior conventions in this space might reach for a fusion architecture as the headline "proposed method," our own evidence (Sections 6.2, 6.4) does not support that framing, and we do not adopt it.

## 4.1 Representations Compared

Fourteen representations, Table 1.

| # | Representation | Dim. | Type |
|---|---|---|---|
| 1 | `classical` | 52 | Handcrafted DSP (MFCC-family + spectral statistics) |
| 2 | `hubert` | 768 | Frozen SSL, `facebook/hubert-base-ls960` |
| 3 | `wavlm` | 768 | Frozen SSL, `microsoft/wavlm-base` |
| 4 | `wavlm_large` | 1024 | Frozen SSL, `microsoft/wavlm-large` |
| 5 | `wav2vec2` | 768 | Frozen SSL, `facebook/wav2vec2-base` |
| 6 | `data2vec_audio` | 768 | Frozen SSL, `facebook/data2vec-audio-base-960h` |
| 7 | `data2vec_spectrogram` | 768 | Frozen SSL (vision variant on rendered Mel-spectrogram images) |
| 8 | `hear` | 512 | Frozen health-acoustic foundation model, `google/hear` |
| 9 | `odi_hb` | 2 | Per-subject clinical severity feature (Oxygen Desaturation Index + Hypoxic Burden, from raw SpO₂) |
| 10–14 | `data2vec_fusion`, `full_fusion`, `full_fusion_v2`, `full_fusion_plus_hear`, `hubert_odi_hb` | 770–3,840 | Concatenations of the above, no learned fusion weighting |

Fusions are simple concatenation — deliberately the least sophisticated combination strategy, so that any fusion-vs-single-encoder difference is attributable to the encoders' own signal content, not a fusion mechanism's capacity (per Sections 6.2/6.4's findings, this choice turned out to matter).

## 4.2 Classifiers

Four downstream classifiers, each with 2 validation-selected hyperparameter candidates: RBF-kernel SVM, Random Forest, XGBoost, and a shallow (2 hidden layers, 256/128 units) MLP. Classifiers are deliberately shallow so any observed effect is attributable to representation choice, not classifier capacity.

## 4.3 Domain Adaptation Methods

**PCA.** Fold-local dimensionality reduction (384/768/1536-D targets), fit in two stages: a tuning-stage fit on source-device training data only (hyperparameter selection), and a refit-stage fit that produces the transform actually applied to the test set. **CORAL.** Covariance whitening-and-recoloring of source-device features toward the target device's second-order statistics, fit using unlabeled target-device validation subjects only (never test data or test labels). See Section 6.3 for the scope bug found and fixed in the PCA implementation.

## 4.4 Additional Signals

**Clinical severity feature.** Per-subject Oxygen Desaturation Index and Hypoxic Burden, computed from raw SpO₂ (3% desaturation threshold below a 100 s rolling-max baseline, ≥8 s duration, sleep-time-corrected), validated against PSG-annotated event counts (Pearson r=0.83 ODI, r=0.61 hypoxic burden). **Label-quality filter.** Every PSG-annotated positive event cross-referenced against objective SpO₂ desaturation timing (45 s lag tolerance); 82.1% of 13,455 annotated events have a corroborating desaturation (OSA 97.0%, hypopnea 74.5%). **HeAR.** Benchmarked identically to the other encoders, alone and fused. **Sliding-window severity.** A non-annotation-privileged target: the whole night binned into fixed 5-minute clock epochs (independent of where annotated events fall), each labeled by its SpO₂-derived desaturation rate.

---

# 5. Experimental Protocol

## 5.1 Cross-Validation and Subject Independence

Subjects are assigned to 5 disjoint folds. For fold *f*, the test set is fold *f*'s subjects, validation is fold (*f*+1 mod 5), and the remainder train. **A runtime assertion checks all three subject sets are pairwise disjoint on every single fold-run** (not only at fold-file construction), a discipline adopted directly after an earlier, deprecated version of this benchmark was found to have subject-level cross-device leakage (Section 8, `analysis/dataset_audit.md`). Hyperparameter selection and model fitting are confined to training-device data and held-out validation subjects in every protocol; test-device data is never used for fitting or selection, including in the domain-adaptation methods.

## 5.2 Device Protocols

Five protocols: `R→R`, `S→S` (matched-device); `R→S`, `S→R` (bidirectional cross-device zero-shot transfer); `(R+S)→(R+S)` (pooled: train and test on both devices combined). The pooled protocol trains and tests on pooled data — it does **not** provide a pooled-train/single-device-test cell; we do not estimate or interpolate that missing cell (Figure 4 marks it N/A explicitly, per the "never fabricate" rule governing this artifact).

## 5.3 Statistical Testing

Every comparative claim uses a paired, subject-level bootstrap (2,000 resamples): per-subject metric scores on an identical, aligned test set are differenced between two arms and resampled with replacement over subjects — not windows, which would treat highly correlated same-subject clips as independent (pseudo-replication). A 95% confidence interval excluding zero is the significance criterion used throughout. The domain-shift acoustic-statistics test (Section 6.1) is the one exception: it is a **window-level** Mann-Whitney U test, explicitly labeled as such, because it characterizes the raw signal distribution rather than a subject-level classifier outcome, and is not used to support any classifier-performance claim.

## 5.4 Compute

All classifier fitting and encoder inference runs on GPU with an explicit hard-failure check (no silent CPU fallback) throughout the pipeline. Full environment and hardware details: `REPRODUCIBILITY.md`.

---

# 6. Results

## 6.1 RQ1 — How severe is cross-device domain shift, and is it visible in the representation itself?

**Signal level.** Table 2 and Figure 3 report six window-level acoustic statistics (RMS energy, spectral centroid, bandwidth, rolloff, zero-crossing rate, spectral flatness) on a stratified sample of 600 windows per device. Every statistic differs between devices with a two-sided Mann-Whitney U p-value below 1e-30, and five of six show a Cliff's delta magnitude above 0.95 (out of a maximum of 1.0) — near-complete distributional separation, not a subtle shift.

**Table 2. Domain-shift audio statistics, Recorder vs. Smartphone (n=600/device, window-level)**

| Statistic | Recorder mean (SD) | Smartphone mean (SD) | \|Cliff's δ\| | p-value |
|---|---|---|---|---|
| RMS energy | 0.0091 (0.0109) | 0.0194 (0.0152) | 0.54 | 2.3×10⁻³⁰ |
| Spectral centroid (Hz) | 985.3 (105.4) | 439.0 (169.1) | 0.98 | <10⁻⁹⁰ |
| Spectral bandwidth (Hz) | 1005.4 (58.1) | 680.7 (138.4) | 0.98 | <10⁻⁹⁰ |
| Spectral rolloff (Hz) | 2154.4 (209.7) | 816.4 (451.2) | 0.99 | <10⁻⁹⁰ |
| Zero-crossing rate | 0.0745 (0.0191) | 0.0225 (0.0134) | 0.97 | <10⁻⁹⁰ |
| Spectral flatness | 3.0×10⁻⁴ (2.2×10⁻⁴) | 2.5×10⁻⁵ (2.5×10⁻⁵) | 1.00 | <10⁻⁹⁰ |

*Source: `scripts/analyze_domain_shift.py`; full data in `tables/domain_shift_audio_stats.csv`, `analysis/domain_shift_summary.json`.*

**Representation level.** Figure 5 projects cached HuBERT embeddings (n=2,400 stratified windows) onto their first two principal components, colored once by device and once by class label. Silhouette score (computed in the full 10-component PCA space, not the 2D plot alone) is **0.145 by device** vs. **0.007 by class label** — device identity is nearly 20× more separable than the clinical label in this frozen representation. This is, to our knowledge, the first direct evidence for *why* cross-device transfer fails on this task: the encoder's own feature space encodes acquisition device far more saliently than task-relevant signal.

## 6.2 RQ2 — How well do current representations transfer, and which is "best"?

Table 3 (full data: `tables/table1_main_benchmark_summary.md` in the accompanying ARA, `MASTER_RESULTS.csv`) reports matched- and cross-device balanced accuracy for all 14 representations (1,120 fold-runs, `R→R/S→S/R→S/S→R` protocols). Every real representation (i.e., excluding the deliberately degenerate `odi_hb` baseline) shows a positive matched-minus-cross gap (Figure 6), confirmed statistically significant for `full_fusion` specifically (`R→R` vs. `R→S`: +5.9 pts, 95% CI [3.3, 8.4], p<0.001; `S→S` vs. `S→R`: +6.0 pts, p<0.001).

**Which representation is best is methodology-dependent.** By raw point estimate averaged across all 4 classifiers, HuBERT ranks highest cross-device (0.5466). Applying the project's own more rigorous, pre-specified methodology — select the primary classifier by validation performance (RBF-SVM), then select the best single encoder by validation performance under that classifier — instead selects **WavLM-large** (validation BA 0.625 vs. HuBERT's 0.594). A paired bootstrap under this selection finds `full_fusion` statistically indistinguishable from WavLM-large in both cross-device directions (`R→S`: p=0.217; `S→R`: p=0.685). We report both conventions rather than the more publishable-sounding point estimate alone, because they disagree, and the disagreement is itself informative about how sensitive "best representation" claims are to selection methodology.

**Table 3 (abbreviated; full table in accompanying materials). Cross-device balanced accuracy, selected representations**

| Representation | Cross BA | Matched BA | Gap |
|---|---|---|---|
| `hubert_odi_hb` | 0.547 | 0.586 | 0.038 |
| `hubert` | 0.547 | 0.584 | 0.038 |
| `wavlm_large` | 0.541 | 0.611 | 0.071 |
| `full_fusion_v2` | 0.536 | 0.580 | 0.044 |
| `full_fusion` | 0.531 | 0.584 | 0.053 |
| `classical` (baseline) | 0.499 | 0.551 | 0.052 |
| `hear` | 0.504 | 0.579 | 0.075 |
| `odi_hb` (degenerate baseline) | 0.496 | 0.496 | 0.000 |

## 6.3 RQ3 — Does pooled-device training or post-hoc domain adaptation improve cross-device generalization?

**PCA: a correctable bug, not a fundamental limitation.** An initial PCA run, fit on source-device data only, collapsed to a degenerate all-one-class predictor in every cross-device configuration tested (43/60 audited combinations at exactly 0.500 balanced accuracy); ROC-AUC remained modestly above chance throughout (0.50–0.59), indicating the underlying ranking signal survived compression even though the fixed decision threshold did not transfer. Comparing PCA's fit scope against CORAL's (already using unlabeled target-device validation data) revealed the discrepancy; extending PCA's refit to the same scope resolves the collapse (paired bootstrap, 60 combinations: mean +0.030 BA, 30/60 significant positive, **0/60 significant negative**).

**CORAL: fails regardless of scope.** Evaluated at full scope after the above fix (4 representations × 4 classifiers × 2 protocols = 160 combinations), CORAL-aligned balanced accuracy is lower than uncorrected features for every representation tested (Table 4), with the aligned classifier's specificity dominating sensitivity (0.78–0.99 vs. 0.01–0.29) — collapse toward the majority-safe class, not genuine domain bridging.

**Table 4. CORAL-aligned vs. uncorrected cross-device balanced accuracy (mean over classifiers/folds)**

| Representation | Uncorrected | CORAL-aligned | Δ |
|---|---|---|---|
| `hubert` | 0.5466 | 0.5293 | −0.0173 |
| `wavlm_large` | 0.5406 | 0.5255 | −0.0151 |
| `full_fusion` | 0.5311 | 0.5196 | −0.0115 |
| `data2vec_fusion` | 0.5347 | 0.5167 | −0.0180 |

**Pooled-device training: the paper's strongest positive finding.** The `(R+S)→(R+S)` protocol was evaluated for 5 representative representations (100 combinations). Pooled-device balanced accuracy is statistically indistinguishable from matched-device performance and dramatically exceeds cross-device performance, for every representation tested (Table 5, Figure 4).

**Table 5. Cross- vs. matched- vs. pooled-device balanced accuracy**

| Representation | Cross BA | Matched BA | Pooled BA | Gap closed |
|---|---|---|---|---|
| `wavlm_large` | 0.541 | 0.611 | **0.613** | 101% |
| `full_fusion_plus_hear` | 0.529 | 0.586 | 0.594 | 115% |
| `full_fusion` | 0.531 | 0.584 | 0.593 | 118% |
| `data2vec_fusion` | 0.535 | 0.583 | 0.588 | 111% |
| `classical` | 0.499 | 0.551 | 0.563 | 123% |

*"Gap closed" = (pooled − cross) / (matched − cross); values above 100% indicate pooled training modestly exceeded matched-device performance, plausibly from the larger effective training set. No architectural change, no inference-time cost — a training-data composition change alone.*

## 6.4 RQ4 — Which components contribute?

Leave-one-encoder-out ablation (6 variants × 4 classifiers × 4 protocols × 5 folds; Figure 8). By raw point estimate, most single-encoder removals numerically exceed `full_fusion`. A paired bootstrap across 89 evaluable combinations tells a different story: mean difference −0.0003, only **7/89** individually significant (4 positive, 3 negative). **We do not have evidence that any specific encoder in the fusion is individually necessary**; the fusion's encoders carry substantially redundant, not complementary, information under naive concatenation. No ablation isolates the contribution of a learned fusion mechanism, attention, or a domain-specific projection layer, because none of these components exist in the evaluated system (Section 4) — we do not claim evidence for components that were not built.

## 6.5 RQ5 — Negative results and failure modes

Three further candidate improvements were tested and did not help (Table 6). Each is reported in full per this project's stated verification discipline.

**Table 6. Negative results**

| Intervention | Result | Significance |
|---|---|---|
| SpO₂-corroboration training filter | Balanced accuracy reduced in 30/32 tested combinations (2/32 marginally positive) | 12/32 significantly negative, 0/32 significantly positive |
| HeAR (health-acoustic model) | Weakest of 14 representations alone (0.504 BA); fusing it into `full_fusion` reduces the point estimate | Fusion-level effect not significant (`R→S`: p=0.256; `S→R`: p=0.730) |
| ODI/Hypoxic-Burden as feature | Chance-level alone (0.496 BA); fused, statistically indistinguishable from HuBERT alone in 7/8 tested combinations | 1/8 marginally significant, remainder not |

The corroboration-filter finding is plausibly explained by AASM annotation convention: hypopnea events lacking a corroborating desaturation are typically scored on nasal flow or chest effort rather than SpO₂, so they are likely real, differently-scored positives rather than annotation noise; removing them shrinks and re-imbalances the training set for no compensating quality gain. The ODI/Hypoxic-Burden null result was predicted before testing — a value constant across all of a subject's windows cannot add per-window discriminative signal to a per-window classifier — and is treated as a confirmed, not merely observed, prediction; its subject-level clinical validity (Pearson r=0.83 vs. PSG-annotated event counts) is not in question, only its use as a raw per-window feature.

**Error-mode characterization.** Aggregated confusion counts (`tn`/`fp`/`fn`/`tp`, stored per fold-run in every `completion.json`) show, consistent with the CORAL result above, that domain-adaptation failure modes systematically favor specificity over sensitivity — models under distribution shift tend toward the majority-safe (negative) prediction rather than toward balanced or false-positive-dominant errors. We do not have per-subject or per-SNR stratified error analysis (`MISSING_EXPERIMENTS.md`, Tier 2) and do not claim it here.

## 6.6 Preliminary: A Non-Annotation-Privileged Severity Target

All windows above are centered on PSG-annotated events — an *annotation-privileged* framing. As a first, explicitly preliminary step toward a target that does not depend on expert event annotation, the whole night was instead binned into fixed 5-minute clock epochs (6,013 epochs, 50 subjects), labeled by SpO₂-derived desaturation rate (binarized severe-vs-not for this first pass), and classified from HuBERT features under the same subject-disjoint protocol (80/80 combinations complete). Matched-device balanced accuracy: 0.531 (SD 0.049, n=40) — a real, modest positive signal. Cross-device balanced accuracy: **0.486, below chance** (SD 0.074, n=40). No significance test has yet been run for this specific comparison (`MISSING_EXPERIMENTS.md`, Tier 1). This result is directionally consistent with, and reinforces, Section 6.1–6.2's central finding, though it should not be over-read given its preliminary, single-encoder, single-framing status.

---

# 7. Discussion

Three findings together reframe how this problem should be approached. First, the device-shift problem is not subtle or inferred — it is directly measurable in the raw signal (Table 2) and directly visible in the representation's own embedding geometry (Figure 5), where device separates roughly 20× more cleanly than the clinical label. This is, to our knowledge, a novel piece of mechanistic evidence for this task: prior work has reported cross-device accuracy drops without examining whether the representation itself is the bottleneck. Second, architectural complexity is not the lever that matters here: a 3,840-dimensional five-encoder fusion is not confirmed significantly better than a single well-chosen encoder, at several times the inference cost, and the ablation finds no evidence that any individual encoder is necessary. Third, and most actionably, post-hoc statistical correction is not a monolithic category — PCA's apparent failure was a correctable implementation-scope bug, while CORAL's failure persisted after applying the identical correct scope — and training-data composition (pooling both devices) outperforms both corrections by a wide margin, at zero inference-time cost. For any deployment where both device types are available during data collection, this is a low-cost, high-yield, immediately actionable recommendation, well ahead of representation-level interventions (contrastive device-invariance training, domain-adversarial pretraining) that this paper's negative results otherwise motivate as necessary next steps.

We are explicit about what this paper does **not** show. It does not show that any encoder is inherently device-invariant — the pooled-training result is a training-data effect, not a representation property, and Figure 5's embedding analysis (on a matched-device-agnostic-trained encoder) still shows device dominating the frozen feature space. It does not show generalization beyond this specific device pair, population, or annotation protocol (Section 8). It does not demonstrate a deployable clinical system.

---

# 8. Limitations

- **Single cohort, single device pair.** N=41–50 subjects, one specific recorder/smartphone pair. Generalization to other hardware, populations, or annotation protocols is untested (`MISSING_EXPERIMENTS.md`, Tier 1).
- **GPU-classifier numerical equivalence unverified.** cuML's tree/SVM implementations and a from-scratch PyTorch MLP are functionally, not verified bit-identical, substitutes for CPU/scikit-learn analogues.
- **The domain-shift acoustic-statistics test (Section 6.1, Table 2) is window-level**, not subject-level — it characterizes the aggregate signal distribution, and its p-values should not be read with the same subject-independence guarantee as the classifier-performance significance tests elsewhere in this paper (Section 5.3).
- **The sliding-window severity result (Section 6.6) is a first cut**: binary framing only, no significance test yet, no device-invariance mechanism.
- **No representation-level device-invariance mechanism was built or evaluated** (contrastive training, domain-adversarial objectives) — every finding in this paper concerns existing frozen representations and post-hoc corrections, not a new architecture.
- **No external dataset or independent device-pair validation.**
- **Provenance.** An earlier version of this project's headline numbers traced to a script with confirmed subject-level cross-device leakage; every number in this paper comes from an independently re-audited pipeline with a runtime leakage assertion on every fold-run, adopted directly in response (`analysis/dataset_audit.md`).

---

# 9. Conclusion

We directly measured acoustic domain shift between two concurrently-recording devices at both the signal level and the representation level, and found it severe and mechanistically visible in a standard frozen encoder's embedding space. Across 14 representations, 4 classifiers, and 5 protocols, no representation — including large multi-encoder fusion — is confirmed significantly more cross-device-robust than the others. Of two post-hoc domain-adaptation corrections tested, one was salvageable after fixing an implementation bug and one was not; pooled-device training, requiring no new model, closed most of the measured gap for every representation tested and is this paper's most actionable finding. Three further candidate interventions did not help and are reported as such. Future work should pursue representation-level interventions this paper's negative results motivate as necessary — paired-recording contrastive device-invariance training (exploiting this dataset's unique time-aligned dual-device structure), domain-adversarial continued pretraining, and validation on an independent device pair and cohort.

---

*Figures referenced: Figure 1 (`figures/fig01_problem_setting.png`, conceptual), Figure 2 (`figures/fig02_architecture.png`, as-implemented), Figure 3 (`figures/fig03_domain_shift.png`, measured), Figure 4 (`figures/fig04_transfer_matrix.png`, measured), Figure 5 (`figures/fig05_embedding_space.png`, measured), Figure 6 (`figures/fig06_generalization_gap.png`, measured), Figure 7 (`figures/fig07_representation_comparison.png`, measured), Figure 8 (`figures/fig08_ablation.png`, measured). Full details: `FIGURE_INVENTORY.md`. References list: see accompanying `ara/logic/related_work.md` (25 entries, IEEE-style numbering matches this manuscript's in-text citations).*
