# Sleep-QuadNet Workshop Revision — Experiment Execution Plan

## Goal

Revise the current Sleep-QuadNet work for the **IEEE HealthCom 2026 workshop: AI-Driven e-Health Systems — From Intelligent Algorithms to Real-World Clinical Deployment**.

The revised paper should **not** be positioned primarily as "a new 3840-D fusion architecture."  
The central scientific question should be:

> **How robust are self-supervised acoustic representations to microphone/device shift in sleep-apnea respiratory-event detection, and when is multi-encoder fusion worth its computational cost?**

This plan uses **only the existing synchronized dual-device dataset** and **drops the external 305-patient PSG-Audio validation idea**.

---

# 1. Fixed Scope

## Dataset

Use only the current synchronized clinical dataset:

- 50 subjects
- Professional/clinical bedside recorder domain: **R**
- Smartphone domain: **S**
- PSG-derived respiratory-event labels already used in the current pipeline
- Same synchronized respiratory events available across both audio devices

Do **not** add another PSG dataset in this revision.

## Task

Keep the main task as:

> **Binary respiratory-event classification from audio windows: normal/non-event vs apnea/hypopnea event**

For the revised manuscript, avoid overstating this as direct patient-level OSA diagnosis unless a separate patient-level experiment is actually performed.

Preferred terminology:

> **Acoustic respiratory-event detection for OSA pre-screening**

---

# 2. First Action — Audit the Existing Code Before Running Anything

Before modifying the pipeline, inspect the existing repository and produce:

`results/audit/current_pipeline_audit.md`

The audit must report:

1. Exact dataset path(s)
2. Exact patient IDs available
3. Number of unique subjects
4. Device names and internal labels
5. Exact audio sampling rate before preprocessing
6. Exact sampling rate after preprocessing
7. Exact window/clip extraction logic
8. Fixed or variable clip duration
9. Window length
10. Hop/overlap, if applicable
11. Positive-label rule
12. Negative/normal-window sampling rule
13. Number of windows per class
14. Number of windows per subject
15. Current train/validation/test split logic
16. Whether splits are truly patient-disjoint
17. Random seeds
18. Existing preprocessing
19. Existing encoders
20. Existing downstream classifiers
21. Existing metrics
22. Existing output files/checkpoints/features
23. Whether SSL features are cached
24. GPU/CPU memory requirements currently observed
25. Any mismatch between code and manuscript

### Critical rule

Do **not** guess missing methodological details.

If window duration, overlap, denoising, normalization, or split rules are unclear, recover them from the current code/configuration and document them explicitly.

---

# 3. Reproducible Master Split

This is the highest-priority methodological correction.

## Required evaluation design

Create a **subject-disjoint 5-fold evaluation**.

Use one fixed fold assignment file:

`metadata/subject_folds_5cv.csv`

Suggested columns:

```text
subject_id,fold
P001,0
P002,3
...
```

### Rules

- A subject must appear in only one fold.
- The **same subject folds must be used for Recorder and Smartphone data**.
- No windows from one subject may appear in both training and testing.
- Hyperparameter selection must not use test subjects.
- Save every split to disk.
- Use fixed seeds.
- Do not regenerate folds separately for individual models.

## Fold execution

For each fold:

- Test = one subject fold
- Remaining subjects = train/validation
- Validation must also be subject-disjoint

If the existing code already has a clean subject-level validation strategy, retain it and document it.

---

# 4. Core Device Protocols

Run all main representations under the same four device conditions:

1. **R → R**
2. **S → S**
3. **R → S**
4. **S → R**

Also retain:

5. **(R + S) → (R + S)**

where useful for the deployment/recovery analysis.

## Important

For cross-device experiments:

- Training and testing subjects must remain patient-disjoint.
- Use identical test-subject identities where logically possible across matched and cross-device comparisons.
- Do not allow paired recordings from a held-out subject to enter training through the opposite device.

---

# 5. Representation Set

Use the representations already central to the paper.

## A. Handcrafted baseline

Current handcrafted acoustic representation.

Record exact:

- filtering
- FFT settings
- frame size
- hop size
- MFCC count
- spectral descriptors
- aggregation procedure

## B. Individual SSL encoders

Run:

- HuBERT
- WavLM
- Wav2Vec2

Use the same pooling strategy currently implemented.

## C. Data2Vec branches

The existing paper combines waveform and spectrogram features.  
For the revision, separate them into:

1. **Data2Vec-Audio**
2. **Data2Vec-Spectrogram**
3. **Data2Vec Audio + Spectrogram**

This is a required ablation.

## D. Full fusion

Current full multi-encoder fusion:

- HuBERT
- WavLM
- Wav2Vec2
- Data2Vec Audio
- Data2Vec Spectrogram

Use the existing implementation first.

Do **not** introduce a new gated/attention fusion model until all reviewer-driven experiments below are complete.

---

# 6. Experiment P0-A — Establish the Device Generalization Gap

## Objective

Quantify how much performance degrades purely because of device shift.

For every representation and classifier, calculate:

```text
Matched Recorder performance: R→R
Matched Smartphone performance: S→S
Cross-device performance: R→S
Cross-device performance: S→R
```

Define:

```text
Gap_R_to_S = Metric(R→R) - Metric(R→S)

Gap_S_to_R = Metric(S→S) - Metric(S→R)
```

Also report:

```text
Mean_Cross_Device = mean(R→S, S→R)
Mean_Matched = mean(R→R, S→S)
Mean_Device_Gap = Mean_Matched - Mean_Cross_Device
```

## Primary metrics

- F1-score
- Balanced Accuracy
- ROC-AUC
- MCC
- Sensitivity
- Specificity

## Output

Create:

```text
results/P0_device_gap/
    fold_metrics.csv
    subject_level_predictions.csv
    protocol_summary.csv
    device_gap_summary.csv
    confusion_matrices/
    roc_curves/
```

---

# 7. Experiment P0-B — Full SSL Fusion Ablation

## Objective

Determine which encoder actually contributes to cross-device robustness.

Run:

1. HuBERT
2. WavLM
3. Wav2Vec2
4. Data2Vec-Audio
5. Data2Vec-Spectrogram
6. Data2Vec Audio + Spectrogram
7. Full Fusion
8. Full Fusion − HuBERT
9. Full Fusion − WavLM
10. Full Fusion − Wav2Vec2
11. Full Fusion − Data2Vec-Audio
12. Full Fusion − Data2Vec-Spectrogram
13. Full Fusion − complete Data2Vec branch

### Classifier strategy

To control experiment count:

#### Stage 1

Run all ablations with the **single strongest and most stable downstream classifier** identified from the current baseline.

Prefer:

- SVM-RBF if robustness is the priority, or
- the classifier that wins on mean cross-device BA using validation only.

#### Stage 2

For only the top 3 representations, run the broader classifier set:

- RF
- XGBoost
- SVM-RBF
- MLP

## Output

```text
results/P0_ablation/
    ablation_fold_metrics.csv
    ablation_summary.csv
    encoder_contribution.csv
```

Required figure:

```text
figures/encoder_ablation_cross_device.pdf
```

Suggested plot:

- x-axis: model/ablation
- y-axis: mean cross-device F1 or BA
- error bars: 95% CI

---

# 8. Experiment P0-C — Statistical Robustness

## Objective

Determine whether Sleep-QuadNet/full fusion improvements over Data2Vec and the strongest single SSL encoder are statistically meaningful.

## Required reporting

For each principal model:

- mean
- standard deviation
- 95% confidence interval

Report at minimum for:

- F1
- Balanced Accuracy
- ROC-AUC
- MCC

## Statistical unit

**Subject**, not individual audio window.

Windows from the same patient are correlated and must not be treated as independent samples for significance claims.

## Recommended comparisons

1. Full Fusion vs Data2Vec Audio+Spectrogram
2. Full Fusion vs best individual SSL encoder
3. Best single encoder vs handcrafted baseline
4. R→R vs R→S device degradation
5. S→S vs S→R device degradation

## Preferred method

Use a **subject-level paired bootstrap** for performance difference confidence intervals.

If using a significance test, ensure it is appropriate for paired subject-level results.

Save:

```text
results/P0_statistics/
    bootstrap_differences.csv
    confidence_intervals.csv
    significance_summary.csv
```

Do not report only p-values.

Always report:

```text
difference + 95% CI + p-value (if used)
```

---

# 9. Experiment P0-D — Computational Cost / Deployment Trade-off

## Objective

Directly answer:

> Is the performance benefit of full multi-encoder fusion worth the extra computation?

Measure for:

- Handcrafted
- HuBERT
- WavLM
- Wav2Vec2
- Data2Vec-Audio
- Data2Vec-Spectrogram
- Data2Vec Audio+Spectrogram
- Full Fusion

## Required measurements

Use the same hardware and batch settings.

Record:

1. Feature dimension
2. Number of encoders/branches
3. Feature-extraction time per clip
4. End-to-end inference time per clip
5. Clips/second
6. Real-Time Factor, if meaningful
7. Peak GPU memory
8. Peak CPU RAM if extraction is CPU-heavy
9. Cached feature size on disk
10. Downstream classifier inference time
11. Total deployable pipeline latency

## Repeat timing

- Warm up before measurement
- Measure multiple runs
- Report mean ± SD

## Output

```text
results/P0_efficiency/
    runtime_benchmark.csv
    memory_benchmark.csv
    representation_size.csv
```

Required figure:

```text
figures/performance_vs_latency.pdf
```

Preferred graph:

- x-axis: inference latency
- y-axis: mean cross-device F1 or BA
- label each representation

This figure is central to the workshop submission.

---

# 10. Experiment P1-A — Acoustic Characterization of Device Shift

## Objective

Show that "Recorder" and "Smartphone" represent a measurable acoustic domain shift rather than merely different names for recording sources.

Use synchronized/paired clips where available.

## Analyze

### Frequency-domain

- Power Spectral Density
- Mean spectrum
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral flatness
- Band-wise energy

Suggested bands after 16 kHz resampling:

```text
0–500 Hz
500–1000 Hz
1–2000 Hz
2–4000 Hz
4–8000 Hz
```

Explicitly document:

```text
fs = 16 kHz
Nyquist frequency = 8 kHz
```

Also document the resampling implementation and anti-aliasing behavior.

### Amplitude/noise-domain

Where possible calculate:

- RMS
- signal energy
- dynamic range
- estimated noise floor
- SNR or a clearly defined proxy

Do not claim absolute SNR if no valid noise-reference method exists.

## Paired analysis

For synchronized Recorder and Smartphone clips, calculate paired differences.

## Output

```text
results/P1_device_acoustics/
    spectral_statistics.csv
    paired_device_statistics.csv
    band_energy.csv
```

Required figures:

```text
figures/mean_psd_recorder_vs_smartphone.pdf
figures/device_band_energy.pdf
```

Optional:

```text
figures/device_embedding_projection.pdf
```

if a clean PCA/UMAP device-domain visualization is useful.

---

# 11. Experiment P1-B — Preprocessing Ablation

## Objective

Determine how much of the device gap is removed by simple signal preprocessing.

Use a compact representation/model combination first.

Compare:

1. Raw/resampled only
2. + peak normalization
3. + current filtering
4. + normalization + filtering
5. + denoising, **only if a defensible denoising method already exists or can be added without changing the scientific scope**

Do not add arbitrary denoising simply to satisfy a reviewer.

If denoising is tested, document:

- algorithm
- parameters
- whether trained or deterministic
- whether it can distort apnea/snore acoustic content

## Output

```text
results/P1_preprocessing/
    preprocessing_ablation.csv
```

Required conclusion:

> Which preprocessing operation reduces cross-device degradation, and by how much?

---

# 12. Experiment P1-C — Dimension-Controlled Fusion

## Objective

Test whether fusion gains are caused merely by increasing representation dimensionality.

Current full fusion dimension is much larger than an individual SSL encoder.

Compare:

1. Best individual SSL encoder — native dimension
2. Data2Vec fusion — native dimension
3. Full Fusion — native dimension
4. Full Fusion → PCA 1536-D
5. Full Fusion → PCA 768-D
6. Optional Full Fusion → PCA 384-D

## Critical leakage rule

PCA must be fitted **only on training data within each fold**.

Never fit PCA before the subject-level split.

## Output

```text
results/P1_dimension_control/
    dimension_control_fold_metrics.csv
    dimension_control_summary.csv
```

Required figure:

```text
figures/performance_vs_feature_dimension.pdf
```

---

# 13. Experiment P1-D — Explicit Domain Adaptation Baseline

## Objective

Test whether simple domain alignment can compete with brute-force multi-encoder fusion.

Preferred first baseline:

### CORAL

Apply feature covariance alignment between source and target domains.

Run initially on:

- best individual SSL encoder
- Data2Vec fusion
- Full Fusion

For:

- R→S
- S→R

If the method is transductive or uses unlabeled target-domain data, state this clearly.

Do not compare it unfairly against strict zero-target-data conditions without marking the difference.

## Output

```text
results/P1_domain_adaptation/
    coral_results.csv
```

Optional only after CORAL:

- DANN
- other adversarial adaptation

Do not expand into many domain adaptation algorithms unless the first result is scientifically useful.

---

# 14. Optional P2 — Lightweight Fusion Upgrade

Run this **only if the reviewer-driven experiments above are complete** and there is time.

## Motivation

The current full model is feature concatenation.

A lightweight upgrade may provide genuine algorithmic novelty.

Possible design:

1. Project every SSL representation to a common dimension
2. Learn encoder weights
3. Fuse via weighted sum or compact gated fusion

Example conceptual form:

```text
h_i = Projection_i(z_i)

alpha_i = softmax(g(h_i))

z_fused = sum(alpha_i * h_i)
```

Potential questions:

- Are learned weights device dependent?
- Does Smartphone audio rely more strongly on noise-robust SSL features?
- Can compact fusion match or exceed 3840-D concatenation?
- Does it reduce latency/memory downstream?

Do **not** implement this before establishing the full baseline and ablations.

---

# 15. Classifier Policy

The paper currently evaluates:

- Random Forest
- XGBoost
- RBF-SVM
- MLP

Do not run every classifier for every experimental variation if this causes a combinatorial explosion.

Use this strategy:

## Main benchmark

All representations × all 4 classifiers.

## Ablations

Use one primary classifier selected using validation performance.

## Final comparison

For top 3 representations, repeat all classifiers.

## Hyperparameter fairness

Use:

- identical tuning budget
- identical folds
- validation-only hyperparameter selection
- no test-set tuning

Save all hyperparameters:

```text
results/configs/
```

---

# 16. Result Tables Required for the Revised Paper

## Table 1 — Dataset and evaluation protocol

Include:

- device
- train subjects
- validation subjects
- test subjects
- train clips
- validation clips
- test clips
- positive/negative counts
- patient-disjoint: yes/no

## Table 2 — Main matched/cross-device benchmark

Columns:

```text
Representation
Classifier
R→R
S→S
R→S
S→R
Mean Cross-Device
Mean Device Gap
```

Primary metric:

- Balanced Accuracy or F1

Add secondary metrics in supplementary/results files.

## Table 3 — Fusion ablation

Columns:

```text
Configuration
Dimension
R→S
S→R
Mean Cross-Device
95% CI
```

## Table 4 — Deployment cost

Columns:

```text
Representation
Dimension
Latency
Peak GPU Memory
Feature Storage
Cross-Device F1
Cross-Device BA
```

## Table 5 — Statistical comparison

Columns:

```text
Comparison
Metric Difference
95% CI
p-value
Interpretation
```

---

# 17. Figures Required

Create publication-ready vector PDF wherever possible.

## Figure 1 — Revised system overview

Focus on:

```text
Same patient
    ↓
Synchronized clinical recorder + smartphone
    ↓
Device-specific acoustic shift
    ↓
SSL representation
    ↓
Respiratory-event classifier
    ↓
Cross-device robustness evaluation
```

Do not make the 3840-D concatenation itself the central visual message.

## Figure 2 — Device acoustic shift

Recorder vs Smartphone PSD / band energy.

## Figure 3 — Cross-device robustness

Grouped model comparison:

- R→R
- S→S
- R→S
- S→R

## Figure 4 — Ablation

Contribution of each SSL branch.

## Figure 5 — Performance vs computational cost

Cross-device F1/BA vs latency.

If page limits are strict, combine or move secondary figures to supplementary material.

---

# 18. Logging Requirements

Every experiment must save:

```text
run_id
timestamp
git_commit
seed
fold
train_subjects
val_subjects
test_subjects
device_train
device_test
representation
classifier
hyperparameters
preprocessing
feature_dimension
metrics
runtime
hardware
```

Save to:

```text
results/master_experiment_log.csv
```

Never overwrite previous runs.

---

# 19. Suggested Repository Structure

```text
project/
│
├── configs/
│   ├── base.yaml
│   ├── encoders/
│   ├── classifiers/
│   └── experiments/
│
├── metadata/
│   ├── subject_folds_5cv.csv
│   └── dataset_manifest.csv
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── statistics/
│   └── benchmarking/
│
├── cached_features/
│
├── results/
│   ├── audit/
│   ├── P0_device_gap/
│   ├── P0_ablation/
│   ├── P0_statistics/
│   ├── P0_efficiency/
│   ├── P1_device_acoustics/
│   ├── P1_preprocessing/
│   ├── P1_dimension_control/
│   ├── P1_domain_adaptation/
│   ├── configs/
│   └── master_experiment_log.csv
│
├── figures/
│
└── scripts/
    ├── audit_pipeline.py
    ├── create_subject_folds.py
    ├── run_main_benchmark.py
    ├── run_ablation.py
    ├── run_statistics.py
    ├── benchmark_efficiency.py
    ├── analyze_device_acoustics.py
    ├── run_preprocessing_ablation.py
    ├── run_dimension_control.py
    └── run_coral.py
```

Adapt this to the existing repository instead of restructuring working code unnecessarily.

---

# 20. Recommended Execution Order

## Phase 0 — Audit

- [ ] Audit current code and dataset
- [ ] Confirm subject IDs
- [ ] Confirm windowing/clip extraction
- [ ] Confirm labels
- [ ] Confirm current preprocessing
- [ ] Confirm current SSL feature extraction
- [ ] Confirm current split logic
- [ ] Write `current_pipeline_audit.md`

## Phase 1 — Fix evaluation

- [ ] Create fixed 5-fold subject-level splits
- [ ] Verify no subject leakage
- [ ] Generate dataset manifest
- [ ] Re-run one cheap baseline end-to-end
- [ ] Confirm fold output consistency

## Phase 2 — Main benchmark

- [ ] Handcrafted
- [ ] HuBERT
- [ ] WavLM
- [ ] Wav2Vec2
- [ ] Data2Vec-Audio
- [ ] Data2Vec-Spectrogram
- [ ] Data2Vec Audio+Spectrogram
- [ ] Full Fusion
- [ ] All main device protocols
- [ ] Main classifier comparison

## Phase 3 — Reviewer-critical ablations

- [ ] Remove HuBERT
- [ ] Remove WavLM
- [ ] Remove Wav2Vec2
- [ ] Remove Data2Vec Audio
- [ ] Remove Data2Vec Spectrogram
- [ ] Remove whole Data2Vec branch
- [ ] Compute contribution deltas

## Phase 4 — Reliability

- [ ] Subject-level bootstrap
- [ ] 95% CIs
- [ ] Principal pairwise comparisons
- [ ] Device degradation significance

## Phase 5 — Deployment

- [ ] Runtime
- [ ] GPU memory
- [ ] feature storage
- [ ] clips/sec
- [ ] performance-vs-latency figure

## Phase 6 — Device acoustics

- [ ] PSD
- [ ] band energies
- [ ] spectral descriptors
- [ ] paired device comparison
- [ ] Nyquist/resampling documentation

## Phase 7 — Additional strengthening

- [ ] Preprocessing ablation
- [ ] PCA dimension-control
- [ ] CORAL domain adaptation

## Phase 8 — Optional novelty upgrade

- [ ] Lightweight learnable fusion only if needed
- [ ] Compare against original concatenation
- [ ] Re-run efficiency analysis

---

# 21. Minimum Experiment Set Before Rewriting the Paper

Do not start the final workshop rewrite until the following are finished:

- [ ] Correct subject-disjoint 5-fold setup
- [ ] Main R→R / S→S / R→S / S→R benchmark
- [ ] Data2Vec audio/spectrogram separation
- [ ] Leave-one-encoder-out fusion ablation
- [ ] Subject-level 95% confidence intervals
- [ ] Full Fusion vs Data2Vec statistical comparison
- [ ] Full Fusion vs best single SSL statistical comparison
- [ ] Inference latency/memory benchmark
- [ ] Device PSD/frequency characterization
- [ ] Exact window/split methodology documented

Strongly preferred:

- [ ] PCA dimension-control
- [ ] Preprocessing ablation
- [ ] CORAL domain adaptation

---

# 22. Decision Rules

## Keep Full Fusion as the headline model only if:

- it gives a reproducible cross-device improvement,
- the confidence interval supports a meaningful gain,
- ablation shows complementary contributions,
- and the performance gain can be justified against computational overhead.

## If Data2Vec performs nearly as well at much lower cost:

Do **not** hide this.

Reframe the paper around:

> **accuracy–efficiency trade-offs for real-world deployment**

This is highly suitable for the target workshop.

## If one encoder dominates:

Reframe around:

> **which SSL pretraining strategy yields the strongest device-invariant respiratory-audio representation**

Again, this remains a publishable and coherent result.

## If fusion gains disappear under corrected CV:

Do not force the old claim.

The revised contribution becomes the rigorous finding that:

> naive multi-encoder concatenation does not necessarily provide clinically meaningful robustness under strict patient- and device-disjoint evaluation.

That is scientifically stronger than an overstated positive result.

---

# 23. Manuscript Positioning After Experiments

The revised paper should answer:

### RQ1
How large is microphone-induced degradation in respiratory-event classification?

### RQ2
Which self-supervised representation is most robust to cross-device domain shift?

### RQ3
Does multi-encoder fusion provide statistically meaningful robustness beyond strong single/fused Data2Vec baselines?

### RQ4
What computational cost is paid for this robustness?

### RQ5
Can simple signal preprocessing or domain alignment reduce the hardware gap?

---

# 24. Revised Contribution Style

Avoid claims such as:

```text
dominates
seamlessly generalizes
proves
exceptional
formidable
device-agnostic
```

Prefer:

```text
improves
reduces the observed device gap
demonstrates within the evaluated hardware
suggests
supports feasibility
provides a controlled cross-device benchmark
```

---

# 25. Final Success Criteria

The revision is ready for the workshop when it can make all of the following statements with experimental evidence:

1. The exact subject-disjoint evaluation protocol is reproducible.
2. Device shift is quantitatively characterized.
3. Cross-device performance is reported separately in both directions.
4. Full fusion is compared fairly with all constituent representations.
5. The contribution of every fusion branch is measured.
6. Performance improvements include subject-level uncertainty estimates.
7. Computational overhead is reported.
8. Feature dimensionality is controlled or explicitly analyzed.
9. Device frequency/noise characteristics are documented.
10. The final recommendation distinguishes:
   - maximum-performance configuration,
   - resource-efficient configuration,
   - strongest zero-target-domain configuration.
11. Conclusions are limited to the two evaluated hardware domains and current 50-subject corpus.
12. The paper is framed as a **real-world deployment robustness study**, not merely a feature-concatenation architecture.

---

# 26. Immediate Coding-Agent Instruction

Use the following as the execution instruction for the current repository:

> Inspect the existing Sleep-QuadNet repository before changing anything. Preserve the existing working data pipeline and first generate a complete audit of subject IDs, device labels, clip/window extraction, preprocessing, representation extraction, classifiers, metrics, current split logic, seeds, and cached outputs. Then implement a fixed subject-disjoint 5-fold evaluation shared identically across Recorder and Smartphone domains. Build the reviewer-driven experiments in this order: main matched/cross-device benchmark, Data2Vec audio-vs-spectrogram ablation, leave-one-encoder-out full-fusion ablation, subject-level confidence intervals/bootstrap comparisons, computational latency/memory benchmark, recorder-vs-smartphone spectral characterization, preprocessing ablation, PCA dimension-controlled fusion, and finally CORAL as a lightweight domain-adaptation baseline. Reuse cached SSL embeddings whenever scientifically valid. Never fit scalers, PCA, feature selection, alignment, or any learned preprocessing on test subjects. Save every fold prediction, subject-level prediction, configuration, runtime measurement, and metric to structured CSV/JSON files. Do not introduce a new fusion architecture until the reviewer-critical experiments are complete. At every phase, update a master experiment log and produce publication-ready summary tables/figures.

---

# 27. First Run

Start with:

```text
1. Audit existing code
2. Create fixed subject folds
3. Run ONE cheap baseline over all four device protocols
4. Verify leakage-free behavior
5. Only then launch SSL experiments
```

The first milestone is **not a higher score**.

The first milestone is:

> **A completely reproducible, patient-disjoint cross-device evaluation pipeline that every subsequent experiment uses unchanged.**
