# Claude Code Master Prompt — Refactor Current Sleep-QuadNet Project into a Paired, Physiology-Guided, Device-Invariant Study

## Role
Act as a senior research engineer and paper-reproduction auditor. Work INSIDE the current repository. Do not delete, overwrite, or silently mutate trusted previous results. Preserve the existing Sleep-QuadNet / cross-device benchmark as a baseline branch and build the new work in a separate, clearly named module/output tree.

The goal is NOT to maximize a headline number. The goal is to rigorously test whether simultaneous Recorder (R) and Smartphone (S) recordings of the SAME physiological event can be used to learn representations that preserve respiratory physiology while suppressing acquisition-device shortcuts, and whether this improves clinically meaningful night-level sleep-apnea screening.

## Non-negotiable principles
1. Subject-disjoint evaluation only.
2. Same-subject Recorder/Smartphone pairs must always remain in the same fold.
3. No test labels or test-device statistics may be used in model selection.
4. Never fabricate missing results, missing plots, missing citations, or missing metadata.
5. Preserve negative results.
6. No silent CPU fallback for numerical/model computation. If required GPU kernels/backends are unavailable, HARD FAIL and report the blocker.
7. CPU may be used only for unavoidable OS orchestration, file listing, metadata reading, process launch, and disk I/O. All feature computation, transforms, training, inference, classifiers, statistics that can reasonably be GPU-resident must use CUDA/GPU implementations.
8. Do not use the PSG-Audio dataset. It is not currently available.
9. Do not claim universal device invariance or unseen-device generalization. This dataset contains one paired Recorder/Smartphone acquisition setting.

---

# A. FIRST: AUDIT THE CURRENT PROJECT AND DATA

Before changing training code, inspect:
- raw data directories;
- current manifests;
- synchronization/alignment code;
- patient/subject IDs;
- Recorder and Smartphone file mapping;
- PSG annotations;
- SpO2 signal availability;
- apnea subtype annotations (OA, CA, MA, hypopnea);
- current sampling rates/channels/codecs found in the actual raw files;
- resampling/normalization/filtering code;
- event-window extraction;
- negative-window sampling;
- fold files;
- cached embeddings;
- classifiers;
- Sleep-QuadNet/full_fusion code;
- PCA/CORAL code;
- existing result tables;
- significance-testing code;
- previous leakage safeguards;
- previous manuscript/figures.

Do not trust prose descriptions when they conflict with executable code or raw-file metadata.

Create:
- `paired_physio_device/AUDIT_REPORT.md`
- `paired_physio_device/DATA_INTEGRITY_REPORT.md`
- `paired_physio_device/audit/raw_audio_inventory.csv`
- `paired_physio_device/audit/pair_inventory.csv`

The raw audio inventory must contain:
`subject_id, device, source_file, original_sr, channels, format, duration_sec, processed_sr, alignment_status`

The pair inventory must contain:
`subject_id, paired_event_id, event_type, event_start, event_end, recorder_file, smartphone_file, pair_alignment_error_ms`

If the raw-file sampling rates differ from the current manuscript, document the discrepancy and correct code/manuscript assumptions. Do not rewrite source data.

---

# B. REFRAME THE SCIENTIFIC TASKS

Use TWO explicit tasks.

## Task 1 — Mechanistic event-level task
Name it:
**PSG-annotated sleep respiratory event classification**

Primary binary target:
`respiratory event vs non-event`

Secondary subtype analysis:
- Normal
- Obstructive apnea (OA)
- Central apnea (CA)
- Mixed apnea (MA)
- Hypopnea

Do NOT call this full OSA screening because the event windows are annotation-centered.

## Task 2 — Clinically relevant night-level screening
No PSG event timestamps may be required at inference.

Input:
full-night Recorder or Smartphone audio

Processing:
fixed windows (start with 30 s and 60 s; choose final settings using validation subjects only)

Output:
- primary: moderate-to-severe OSA screening if cohort distribution supports `AHI >= 15`
- secondary: severe OSA `AHI >= 30` if class counts are sufficient
- continuous AHI regression if label quality permits
- secondary physiological targets: ODI and hypoxic/desaturation burden

Before defining thresholds, audit subject-level AHI distribution and report counts. Do not create a binary task with unusably small class counts.

---

# C. REPLACE SLEEP-QUADNET AS THE PROPOSED ARCHITECTURE

Keep Sleep-QuadNet/full_fusion only as a BASELINE.

Implement a new paired physiology-guided device-disentangled model. Temporary research name:
**PairPhysNet**

Do not use the final name in the paper until the complete results support the method.

## C1. Shared encoder
Choose ONE primary audio backbone using validation-only baseline evidence:
prefer WavLM-large or HuBERT if already validated.

For paired same-event audio:
- `x_R`: Recorder
- `x_S`: Smartphone

Shared weights:
`h_R = Encoder(x_R)`
`h_S = Encoder(x_S)`

Implement a configurable pooling module:
1. mean pooling baseline
2. mean+std statistical pooling
3. attentive statistical pooling

Do not concatenate five large pretrained encoders in the proposed method.

## C2. Physiology-content branch
`c_R = PhysProjector(Pool(h_R))`
`c_S = PhysProjector(Pool(h_S))`

Default projection dimension: 256.
L2-normalize for paired contrastive training.

The respiratory-event classifier and night-level aggregator must consume the physiology-content representation `c`, not the device-style representation.

## C3. Device-style branch
`d_R = DeviceProjector(Pool(h_R))`
`d_S = DeviceProjector(Pool(h_S))`

A device classifier on `d` should correctly distinguish:
- 0 Recorder
- 1 Smartphone

## C4. Device adversarial head on physiology content
Add:
`c -> GradientReversal -> DeviceClassifier`

The purpose is to reduce device identity recoverability from `c`.

## C5. Disentanglement / orthogonality
Implement a stable disentanglement loss between physiology-content and device-style representations, e.g. normalized cross-covariance or cosine/orthogonality loss.

Do not use a numerically unstable raw dot-product penalty without normalization.

## C6. Paired same-event alignment
The simultaneous R/S views of the SAME event are natural positive pairs.

Implement a standard NT-Xent / InfoNCE paired contrastive objective on `c_R, c_S`.

Positive:
same `paired_event_id` across R/S.

Negatives:
other events in the batch, while avoiding false negatives where practical (e.g. same event pair or duplicated augmented view).

Keep temperature configurable and tune only on validation data.

---

# D. PHYSIOLOGY-GUIDED SUPERVISION

## D1. Stop using ODI/Hypoxic Burden as repeated per-window features
The existing subject-level ODI/HB values are constant across a subject's event windows and should NOT be treated as peer "representations" beside HuBERT/WavLM.

Keep the old result as a negative baseline, but remove it from the core representation leaderboard.

## D2. Event-local SpO2 auxiliary targets
From raw SpO2, derive event-local physiological targets with definitions stored explicitly in code/config:
- desaturation occurrence probability;
- desaturation amplitude;
- event-to-nadir delay;
- event-associated desaturation area;
- optional future SpO2 delta curve summary.

First generate event-aligned SpO2 traces to empirically characterize timing from event end to nadir by subtype.

Do NOT use an arbitrary fixed lag without auditing the actual dataset distribution.

Add auxiliary heads from physiology-content representation `c`.

Example:
- classification head for `>=3% desaturation`
- regression head for desaturation amplitude
- regression head for event-associated desaturation area

Do not claim exact SpO2 prediction unless the metrics support it.

## D3. Hypoxic burden terminology audit
Inspect the current hypoxic-burden formula.

If it does not match a recognized event-associated whole-night hypoxic-burden formulation, rename it conservatively in code/results, e.g.:
`oximetry_desaturation_burden`

Do not silently call a custom metric "Hypoxic Burden."

## D4. Night-level physiological targets
At night level, predict:
- AHI
- ODI
- hypoxic/desaturation burden
- screening class

Use SpO2/HB as subject/night-level targets, not repeated event inputs.

---

# E. TRAINING OBJECTIVES / ABLATIONS

Implement the following controlled variants with identical splits.

### A0 — Frozen baseline
Frozen validated encoder + shallow GPU classifier.

### A1 — CE only
Fine-tuned/partially fine-tuned encoder + respiratory event classification only.

### A2 — CE + Paired Contrastive
`L = L_event + lambda_pair * L_pair`

### A3 — CE + Device Adversarial
`L = L_event + lambda_adv * L_adv_device`

### A4 — CE + Pair + Device Adversarial
`L = L_event + lambda_pair*L_pair + lambda_adv*L_adv_device`

### A5 — Full PairPhysNet
`L = L_event
   + lambda_pair*L_pair
   + lambda_adv*L_adv_device
   + lambda_dis*L_disentangle
   + lambda_spo2*L_spo2_aux`

If appropriate, include the device-style classification objective on `d`.

Tune lambda values using validation subjects only. Do not tune using test performance.

Record every chosen value in config files.

---

# F. POOLED-DEVICE CONTROLS — MANDATORY

The old `(R+S)->(R+S)` result is confounded by increased training data.

Run:

1. `R -> R`
2. `S -> S`
3. `R -> S`
4. `S -> R`
5. `(R+S balanced N total) -> R`
6. `(R+S balanced N total) -> S`
7. `(R+S full) -> R`
8. `(R+S full) -> S`
9. `(R+S full) -> (R+S)` only as a secondary diagnostic

For equal-data control:
- `R-only`: N
- `S-only`: N
- `R+S-balanced`: N TOTAL, approximately N/2 + N/2
- `R+S-full`: all available examples

Only claim device-diversity benefit if the equal-data mixed-device condition improves over single-device training.

---

# G. DEVICE-SHIFT MECHANISTIC ANALYSIS

Use exact paired same-event R/S recordings wherever possible.

For each pair calculate:
- pre-normalization RMS / loudness proxy;
- post-normalization RMS;
- PSD;
- frequency-wise paired difference;
- log-spectral distance;
- spectral centroid;
- bandwidth;
- rolloff;
- ZCR;
- flatness;
- cross-correlation lag;
- residual alignment error;
- magnitude-squared coherence where appropriate;
- embedding cosine similarity;
- normalized embedding Euclidean distance;
- paired prediction probability difference `|p_R-p_S|`.

Use subject-level aggregation/bootstrap for inferential claims where repeated windows exist.

Do not base major p-value claims on treating correlated windows as independent.

---

# H. DEVICE PROBE — MANDATORY

For each major representation:
- frozen HuBERT
- WavLM
- WavLM-large
- Wav2Vec2
- Data2Vec
- HeAR if available
- CE-only PairPhysNet encoder
- full PairPhysNet physiology-content branch

train a SUBJECT-DISJOINT GPU device classifier:
Recorder vs Smartphone.

Report:
- balanced accuracy
- AUROC
- confidence interval

The proposed method should ideally reduce device-probe predictability from physiology-content `c` while preserving/improving event performance.

Also verify that device-style branch `d` retains device information.

---

# I. EVENT-TYPE AND ERROR PHENOTYPE ANALYSIS

At minimum calculate binary detector recall/sensitivity separately for:
- OA
- CA
- MA
- Hypopnea

Create a protocol x subtype heatmap.

Build error phenotypes:
1. R correct + S correct
2. R correct + S wrong
3. R wrong + S correct
4. R wrong + S wrong
5. false negative with strong desaturation
6. false negative with weak/no desaturation
7. false positive during snoring/hard negative, only if identifiable from available annotations/data

Never invent clinically unsupported hard-negative labels.

---

# J. QUALITATIVE CASE ATLAS — MANDATORY

Create scientifically selected, non-cherry-picked paired cases.

For each case show:
- Recorder waveform
- Recorder log-Mel spectrogram
- Smartphone waveform
- Smartphone log-Mel spectrogram
- PSG event start/end and subtype
- SpO2 trace aligned around event
- baseline SpO2
- nadir
- desaturation amplitude
- event-associated desaturation area
- model probability on R
- model probability on S
- baseline model probability and proposed model probability if available
- attention/importance overlay if method supports it

Select representative cases by a PREDEFINED rule:
- median-confidence concordant correct pair;
- median disagreement among R-correct/S-wrong;
- median disagreement among R-wrong/S-correct;
- median-confidence both-wrong case.

Save the selection rule and selected IDs in a machine-readable JSON.

Do not manually pick visually dramatic examples after seeing the plots.

---

# K. NIGHT-LEVEL SCREENING MODEL

Construct full-night annotation-free inference.

1. Split entire night into fixed windows.
2. Extract physiology-content representation `c_t`.
3. Aggregate:
   - attention MIL baseline
   - Transformer or temporal attention model
4. Predict subject/night endpoints.

Primary if cohort supports:
`P(AHI >= 15)`

Secondary:
- `P(AHI >= 30)`
- AHI regression
- ODI regression
- hypoxic/desaturation burden regression

No PSG event timestamps are allowed as model input at inference.

Evaluate:
`R->R, S->S, R->S, S->R, R+S->R, R+S->S`

Also measure paired SAME-PATIENT screening consistency:
`|P_R - P_S|`
and, for continuous predictions:
`|AHI_hat_R - AHI_hat_S|`

---

# L. METRICS

## Event level
- balanced accuracy
- macro F1
- AUROC
- PR-AUC
- sensitivity
- specificity
- PPV
- NPV where valid
- MCC
- Brier score/calibration if probabilities are available

## Night-level screening
- AUROC
- PR-AUC
- sensitivity/specificity
- sensitivity at fixed specificity if clinically sensible
- PPV/NPV
- calibration curve
- Brier score

## Night-level regression
- MAE
- RMSE
- Spearman rho
- Pearson r if assumptions are acceptable
- Bland-Altman bias / limits of agreement

Always report 95% subject-level confidence intervals for headline metrics.

---

# M. STATISTICAL PLAN

Predefine PRIMARY hypotheses:
H1: matched-device performance > cross-device performance.
H2: full PairPhysNet improves R->S over CE-only baseline.
H3: full PairPhysNet improves S->R over CE-only baseline.
H4: PairPhysNet reduces device-probe performance in physiology-content representation.
H5: PairPhysNet reduces same-event R/S embedding distance and prediction disagreement.
H6: if night-level screening works, PairPhysNet improves cross-device night-level screening consistency/performance.

Use paired subject-level bootstrap where appropriate.

Use multiple-comparison control for secondary/exploratory families:
Holm or Benjamini-Hochberg FDR.

Do not color a figure "significant" merely because at least one of many comparisons has p<0.05.

---

# N. REQUIRED FIGURES

Generate publication-quality vector PDF/SVG + 600 dpi PNG.

1. Clinical problem / simultaneous paired acquisition.
2. PairPhysNet architecture.
3. Paired raw-domain shift: PSD/spectral difference/coherence.
4. Paired physiological case: R waveform+spec, S waveform+spec, PSG event, SpO2, event-associated desaturation.
5. Device shortcut analysis: PCA/UMAP only as exploratory + quantitative device probe.
6. Direction-specific R->R/R->S and S->S/S->R forest plot with CIs.
7. Baseline vs PairPhysNet cross-device performance.
8. Device-probe before/after + physiology/device branch comparison.
9. Paired embedding-distance distributions.
10. Paired prediction-consistency scatter `p_R vs p_S`.
11. Event subtype x protocol heatmap.
12. Correct/misclassified paired-case atlas.
13. Equal-data pooled-training control.
14. Night-level ROC/PR.
15. PSG AHI vs predicted AHI and Bland-Altman if regression is successful.
16. Supported-vs-not-supported prediction summary figure for the final paper.

Figures must be generated only from real result artifacts.

---

# O. WHAT THE FINAL PAPER MAY AND MAY NOT CLAIM

Potentially supported if results confirm:
- paired acquisition-device shift exists;
- device identity is recoverable from frozen SSL embeddings;
- paired training reduces device-specific information;
- paired training improves R<->S transfer;
- audio encodes event-associated physiological consequence;
- whole-night audio can screen moderate/severe OSA, if validated.

Not supported unless separately demonstrated:
- universal microphone invariance;
- arbitrary unseen-device generalization;
- tracheal-audio transfer;
- PSG-equivalent diagnosis;
- generalization to new hospitals/populations;
- cardiovascular-risk prediction;
- exact SpO2 reconstruction;
- reliable apnea subtype diagnosis if subtype metrics are weak.

---

# P. GPU EXECUTION

Use the project skill `/gpu-only-4way`.

Before any training:
- inspect all 4 GPUs;
- verify CUDA;
- verify VRAM;
- create a GPU job plan;
- hard fail on CPU fallback.

Parallelize safely:
- feature-extraction shards across GPUs;
- folds/experiments as independent jobs;
- heavy model variants one GPU each when feasible;
- use DDP only when one job genuinely requires multiple GPUs;
- never oversubscribe VRAM.

Write:
`paired_physio_device/GPU_EXECUTION_PLAN.md`
and continuously update:
`paired_physio_device/GPU_JOB_STATUS.csv`

---

# Q. OUTPUT TREE

Create:

paired_physio_device/
├── AUDIT_REPORT.md
├── DATA_INTEGRITY_REPORT.md
├── EXPERIMENT_PLAN.md
├── GPU_EXECUTION_PLAN.md
├── configs/
├── audit/
├── manifests/
├── scripts/
├── models/
├── logs/
├── checkpoints/
├── results/
│   ├── event/
│   ├── device_probe/
│   ├── physiology/
│   ├── pooled_controls/
│   ├── screening/
│   └── statistics/
├── figures/
│   ├── main/
│   ├── supplementary/
│   └── case_atlas/
├── tables/
└── artifacts/

Maintain:
`paired_physio_device/results/MASTER_RESULTS.csv`

Minimum columns:
`experiment_id, fold, seed, task, train_device, test_device, model_variant, backbone, pooling, window_sec, normalization, lambda_pair, lambda_adv, lambda_dis, lambda_spo2, n_train_subjects, n_val_subjects, n_test_subjects, n_train_samples, n_test_samples, BA, F1, AUC, PRAUC, sensitivity, specificity, MCC, Brier, runtime_sec, gpu_id, config_hash, git_commit`

---

# R. STAGED EXECUTION

Stage 0: Audit and freeze existing baseline artifacts.
Stage 1: Verify/reproduce trusted existing baselines only where needed.
Stage 2: Build exact paired-event dataloader and alignment validation.
Stage 3: Run paired acoustic/device-shift analysis.
Stage 4: Run device probes.
Stage 5: Implement PairPhysNet and unit tests.
Stage 6: Run A1–A5 model variants.
Stage 7: Run equal-data pooled controls.
Stage 8: Run subtype/error/case-atlas analyses.
Stage 9: Build night-level screening task.
Stage 10: Run statistical testing and multiple-comparison correction.
Stage 11: Run `/results-audit`.
Stage 12: If and only if audit passes, run `/publication-figures`.
Stage 13: If figures/tables are finalized, run `/journal-paper`.

At each stage:
- checkpoint;
- log exact command/config;
- do not overwrite old results;
- stop and report any data-integrity issue before continuing.

---

# S. COMPLETION ARTIFACT

At the end create:
`paired_physio_device/artifacts/PAIRPHYSNET_RESEARCH_ARTIFACT.md`

It must summarize:
- data audit;
- exact task definitions;
- architecture;
- losses;
- device probe;
- paired signal analysis;
- event results;
- subtype results;
- physiology results;
- pooled controls;
- night-level screening;
- statistics;
- negative results;
- success/failure case analysis;
- supported claims;
- unsupported claims;
- recommended paper title;
- recommended abstract;
- recommended contribution bullets;
- exact figures/tables to publish;
- unresolved weaknesses;
- next experiments needed for a top-tier submission.

Do not write a triumphant conclusion if the proposed method fails. The artifact must reflect the actual evidence.
