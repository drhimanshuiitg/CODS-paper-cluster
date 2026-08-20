You are acting as a **senior ML researcher, scientific writer, statistical analyst, visualization expert, and hostile Q1 journal reviewer simultaneously**.

Your job is NOT merely to summarize my existing code or convert results into a paper.

Your task is to **reverse-engineer the entire research project from the code, experiment outputs, logs, saved predictions, tables, checkpoints, metadata, and existing documents**, determine what scientific story is genuinely supported by the experiments, and then construct a **publication-grade research artifact comparable to a strong Q1 journal / IEEE Transactions / ICASSP-quality manuscript**.

The final paper must be **evidence-driven, technically rigorous, visually strong, and reviewer-resistant**.

---

# 1. PROJECT LOCATIONS

Primary project/code/results directory:

`/home/pkdas/IEEE_healthcomm_workshop`

Dataset/storage directory:

`/scratch/pkdas/IEEE_healthcomm_workshop`

First inspect the entire project recursively.

Look for:

- Python scripts
- notebooks
- shell/SLURM scripts
- experiment configurations
- README files
- Markdown experiment plans
- CSV/TSV files
- JSON files
- TXT logs
- training logs
- TensorBoard logs
- checkpoints
- saved embeddings/features
- prediction files
- probability scores
- fold-wise predictions
- subject IDs
- dataset metadata
- confusion matrices
- metrics
- existing figures
- existing architecture drawings
- statistical analysis files
- ablations
- baselines
- model variants
- experiment folders
- previous manuscript text
- previous paper drafts
- experiment-plan `.md` files
- GPU experiment logs

Do NOT assume that filenames correctly describe the experiment.

Infer what was actually done from the implementation.

---

# 2. FIRST: REVERSE ENGINEER THE ENTIRE STUDY

Before writing anything, reconstruct the complete experiment.

Determine:

## Dataset

Identify:

- datasets used
- number of subjects
- number of recordings
- device/source of each recording
- microphone recordings
- tracheal recordings
- recorder recordings
- tracheal/device-specific data
- sampling rates
- recording durations
- labels
- class distributions
- number of clips/windows
- train/validation/test sizes
- subject-level distributions
- preprocessing
- segmentation
- normalization
- augmentation
- resampling
- feature extraction
- SSL model preprocessing
- exclusion criteria
- quality-control steps

Explicitly determine whether the datasets are:

- recordings of the same physiological phenomenon with different sensors,
- different populations,
- different recording environments,
- different devices,
- different anatomical acquisition positions,
- or some combination of these.

This distinction is CENTRAL to the scientific claims.

---

# 3. CHECK DATASET SIMILARITY AND DOMAIN SHIFT

Perform a serious analysis of whether the datasets/domains are actually comparable enough for the proposed robustness/generalization claims.

Where data permits, investigate differences in:

- sampling frequency
- bandwidth
- duration
- loudness
- RMS energy
- SNR proxies
- spectral centroid
- spectral bandwidth
- spectral rolloff
- zero-crossing rate
- log-Mel statistics
- MFCC distributions
- frequency-energy distributions
- embedding distributions
- clip lengths
- patient demographics if available
- class balance
- label definitions
- annotation protocols

If embeddings already exist, use them.

Otherwise, if computationally reasonable and already supported by the project, obtain representations using the same encoder being studied.

Create visualizations such as:

- PCA
- UMAP/t-SNE where appropriate
- distribution plots
- feature-density plots
- embedding-space domain visualization
- dataset/domain separability
- inter-domain distance matrix
- cosine similarity distributions
- MMD or another reasonable domain-distance estimate where scientifically appropriate

Do not introduce a metric merely because it looks impressive.

Explain what each analysis demonstrates.

The goal is to answer:

> Are these datasets sufficiently related to represent the same underlying task while being sufficiently different to constitute meaningful domain/device shift?

This must become part of the scientific motivation.

---

# 4. DETERMINE THE REAL PROBLEM STATEMENT

Do NOT blindly accept the current paper title or problem statement.

Based on the experiments that were actually performed, determine the **strongest scientifically defensible research question**.

Possible concepts may include:

- cross-device robustness
- cross-sensor robustness
- cross-domain generalization
- device-independent respiratory acoustic representation learning
- sensor/domain transfer
- robustness of self-supervised representations
- representation invariance
- robustness under acquisition mismatch
- transferability of nocturnal respiratory acoustic representations
- robustness of sleep-apnea-related acoustic representations

But choose ONLY what the evidence actually supports.

Distinguish carefully between:

- robustness
- generalization
- domain adaptation
- domain transfer
- cross-device transfer
- cross-dataset validation
- sensor invariance
- representation invariance
- multimodal fusion
- domain robustness

These terms are NOT interchangeable.

Construct:

1. Clinical/real-world problem
2. ML problem
3. Research gap
4. Scientific hypothesis
5. Research questions
6. Contributions
7. Claims supported by evidence

---

# 5. BUILD A CLAIM–EVIDENCE MATRIX

Before writing the manuscript, create a private working table:

| Claim | Evidence | Experiment | Figure/Table | Strength | Caveat |

Every important claim in the manuscript must map to actual evidence.

Categorize each as:

- Strongly supported
- Moderately supported
- Exploratory
- Unsupported

Remove or soften unsupported claims.

**Never fabricate an experiment, result, patient count, statistical test, architecture component, or numerical value.**

If something desirable for the paper was NOT evaluated, explicitly label it:

`NOT EVALUATED`

Do not silently imply that it was.

---

# 6. RECONSTRUCT THE MODEL ARCHITECTURE

Reverse-engineer the architecture directly from code.

Identify:

- input representation
- input dimensions
- encoders
- pretrained models
- frozen/trainable layers
- pooling
- projection layers
- fusion mechanism
- attention
- classifier
- dimensionality at each major stage
- loss functions
- optimization
- learning rates
- schedulers
- batch size
- epochs
- stopping criterion
- regularization
- parameter counts where obtainable
- training/inference flow

Trace tensor shapes wherever practical.

Produce a concise architecture specification such as:

`Input → preprocessing → encoder → representation → fusion → classifier → output`

But also describe all branches accurately.

---

# 7. GENERATE A PUBLICATION-QUALITY ARCHITECTURE FIGURE

If no good architecture figure exists, CREATE one.

Do not settle for an ugly raw Mermaid diagram if a more publication-quality vector-style figure can be generated.

Use Python/Graphviz/Matplotlib or an appropriate vector graphics workflow.

The architecture diagram should clearly show:

- each device/domain
- input respiratory/audio signal
- preprocessing
- SSL encoder(s)
- representations
- device-specific/shared components
- fusion mechanism
- classifier
- output
- training direction
- cross-domain testing setup if relevant

Prefer a professional academic visual style similar to figures in:

- IEEE Transactions
- Nature Machine Intelligence
- Medical Image Analysis
- Pattern Recognition
- IEEE JBHI
- ICASSP

Export:

- PDF/SVG if possible
- high-resolution PNG

Make text readable when placed in a two-column manuscript.

---

# 8. CREATE A SECOND FIGURE EXPLAINING THE SCIENTIFIC IDEA

Create a conceptual/problem-setting figure separate from the architecture.

For example:

**same clinical phenomenon → different acquisition devices/sensors → substantial acoustic/domain shift → conventional model degradation → proposed representation/fusion strategy → improved cross-domain robustness**

This figure should make the paper understandable within approximately 30 seconds.

Do not duplicate the architecture figure.

---

# 9. AUDIT ALL EXISTING RESULTS

Find every available experiment.

Construct a master results sheet containing:

- experiment ID
- training dataset
- test dataset
- sensor/device
- model
- feature
- classifier
- seed/fold
- accuracy
- precision
- recall
- specificity
- F1
- macro-F1
- weighted-F1
- ROC-AUC
- PR-AUC if available
- MCC if available
- sensitivity
- balanced accuracy
- confidence intervals if derivable
- any other existing metric

Do not mix subject-level and clip-level metrics.

Clearly identify evaluation unit.

Check for:

- inconsistent metrics
- suspicious results
- leakage
- duplicated samples
- subject overlap
- train-test contamination
- normalization using test data
- augmentation leakage
- inconsistent class mappings
- different evaluation populations
- different random seeds
- checkpoint selection using test performance
- accidental use of test labels
- inconsistent preprocessing between domains

If you detect a methodological problem, **do not hide it**.

Document it.

---

# 10. CHECK WHETHER THE CLAIMED ROBUSTNESS EXPERIMENT IS VALID

This is particularly important.

Determine whether the current experimental design truly establishes robustness.

For each experiment identify:

`Train domain → Test domain`

Examples:

- microphone → microphone
- microphone → tracheal
- tracheal → microphone
- recorder → another recording device
- pooled domains → unseen domain
- leave-one-device/domain-out
- in-domain → out-of-domain

Construct a **train-test transfer matrix** if possible.

A particularly strong figure would be a heatmap:

Rows = training domains  
Columns = testing domains  
Cell = Macro-F1 / AUROC / balanced accuracy

This allows readers to immediately see:

- in-domain performance
- cross-domain degradation
- strongest transfer directions
- asymmetric transfer
- whether the proposed model reduces the generalization gap

Define something like:

`Generalization Gap = In-domain performance – Cross-domain performance`

ONLY if scientifically justified.

If this can be calculated from existing results, calculate it.

---

# 11. CREATE ALL IMPORTANT FIGURES THAT ARE SUPPORTED BY EXISTING RESULTS

Search existing results and determine which visualizations are missing.

Potential figures include:

### Figure A
Study motivation / domain-shift illustration

### Figure B
Complete model architecture

### Figure C
Dataset/device characteristics

### Figure D
Cross-domain performance heatmap

### Figure E
In-domain vs cross-domain performance

### Figure F
Baseline vs proposed model

### Figure G
Generalization-gap comparison

### Figure H
Encoder/model comparison

### Figure I
Ablation study

### Figure J
Embedding visualization by domain

### Figure K
Embedding visualization by class

### Figure L
Confusion matrices

### Figure M
Per-class sensitivity/F1

### Figure N
ROC curves

### Figure O
Precision-recall curves when imbalance makes them useful

### Figure P
Subject-wise performance variation

### Figure Q
Domain-distance vs performance-degradation relationship, if data permits

### Figure R
Performance vs model complexity, if data permits

### Figure S
Statistical uncertainty / fold-wise variation

### Figure T
Error analysis

Do NOT make all figures automatically.

Create only figures that convey a meaningful scientific insight and are supported by existing data.

The paper should not look artificially inflated.

---

# 12. FIGURE QUALITY REQUIREMENTS

Every figure must:

- make one clear scientific point
- use publication-quality fonts
- use readable axis labels
- use consistent terminology
- avoid clutter
- use high resolution
- include units
- use sensible scales
- show uncertainty where appropriate
- not exaggerate tiny differences
- be interpretable in grayscale where feasible
- have a detailed paper-quality caption

Avoid generic Excel-style graphics.

---

# 13. STATISTICAL ANALYSIS

Where raw fold/subject predictions permit, determine whether stronger statistical analysis can be performed using EXISTING outputs.

Consider:

- 95% confidence intervals
- bootstrap confidence intervals
- paired tests
- Wilcoxon signed-rank test
- McNemar test
- DeLong test for AUC when appropriate
- permutation tests
- effect size
- fold/subject variability

Choose tests based on the data and experimental design.

Do not blindly compute p-values.

Explain:

- what is being compared
- why the test is valid
- unit of analysis
- number of samples
- assumptions
- result
- practical significance

Avoid pseudo-replication from treating highly correlated clips from the same patient as independent subjects.

---

# 14. PERFORM ERROR ANALYSIS

A Q1-level paper should explain not only where the model succeeds but where it fails.

Where possible investigate:

- domain-specific errors
- sensor-specific errors
- class-specific errors
- low-SNR recordings
- ambiguous clips
- false positives
- false negatives
- unusually difficult subjects
- device/domain mismatch
- class imbalance effects

Identify systematic patterns.

Do not create explanations that cannot be connected to evidence.

Separate:

**Observed finding**

from

**Possible interpretation**

---

# 15. BASELINES AND ABLATIONS

Inspect the repository to determine whether there are:

- MFCC models
- CNNs
- ResNets
- Wav2Vec2
- HuBERT
- WavLM
- Whisper
- AudioMAE
- AST
- handcrafted features
- classical classifiers
- different fusion strategies
- frozen vs fine-tuned encoders
- single-domain vs multi-domain training
- individual encoder branches
- pooling alternatives

Do NOT automatically run expensive new experiments.

First report what exists.

From existing evidence, identify which comparisons constitute:

- baseline
- proposed system
- ablation
- robustness validation
- external validation

If a crucial experiment is missing, list it separately under:

`Experiments required before final journal submission`

---

# 16. RECONSTRUCT THE NOVELTY

Do not write generic novelty such as:

> We propose a deep-learning model for sleep apnea detection.

That is insufficient.

Infer what is genuinely novel.

Potential novelty could arise from:

- the problem formulation
- cross-device evaluation
- cross-sensor respiratory acoustics
- multi-encoder representation learning
- device-aware fusion
- unseen-domain evaluation
- systematic robustness characterization
- representation-level analysis
- robustness-gap measurement
- cross-device benchmark creation

But claim them only if justified.

Write contributions that are:

- precise
- measurable
- non-overlapping
- technically defensible

Avoid saying:

- “first”
- “novel”
- “state-of-the-art”
- “device invariant”
- “clinically deployable”

unless there is strong evidence.

---

# 17. CONSTRUCT A STRONG PAPER STORY

The manuscript should follow the scientific narrative:

### Real-world problem

Models trained on one respiratory audio acquisition system may experience substantial distribution shift when deployed on another sensor/device.

↓

### Missing evidence in literature

Existing works may report strong in-domain performance but insufficient cross-device evaluation.

↓

### Research question

Can learned respiratory acoustic representations retain task-relevant information across heterogeneous acquisition conditions?

↓

### Methodological response

Explain exactly what the proposed architecture/training setup does.

↓

### Evaluation strategy

In-domain + cross-domain + cross-device + relevant ablations.

↓

### Key result

Show whether the proposed system reduces performance degradation under domain shift.

↓

### Interpretation

Explain what representations/components drive generalization.

↓

### Limitation

Explain what is still NOT demonstrated.

---

# 18. WRITE THE MANUSCRIPT AT Q1 JOURNAL STANDARD

Create:

`Q1_Paper_Artifact/manuscript.md`

Use approximately this structure:

# Title

Create 5 candidate titles first internally.

Choose the title that best matches the evidence.

Avoid overclaiming.

# Abstract

Structured logically around:

- problem
- gap
- approach
- datasets
- evaluation
- most important quantitative results
- scientific conclusion

No vague marketing language.

# 1. Introduction

Must establish:

- clinical/technical context
- deployment problem
- device/domain shift
- limitations in existing research
- why the question matters
- hypothesis
- approach
- contributions

End with 3–5 sharp contribution statements.

# 2. Related Work

Organize conceptually rather than paper-by-paper.

Potential subsections:

- acoustic sleep apnea analysis
- respiratory/snoring audio models
- self-supervised audio representation learning
- cross-domain/device robustness
- multi-encoder/fusion approaches

If literature references are already available locally, use them.

Do NOT invent citations.

Mark areas requiring literature verification.

# 3. Materials / Dataset

Include a polished dataset table.

Describe:

- population
- sensors
- devices
- labels
- sample rate
- recording conditions
- segmentation
- splits
- ethics if available

Clearly explain domain differences.

# 4. Proposed Method

Include:

- mathematical notation where useful
- preprocessing
- encoders
- representation extraction
- fusion
- prediction head
- objective function
- optimization

Do not add mathematical complexity for appearance.

Every equation must contribute information.

# 5. Experimental Protocol

Explain:

- split strategy
- subject independence
- domain settings
- baselines
- metrics
- statistical testing
- implementation details

# 6. Results

Do not make this a sequence of tables.

Each subsection should answer one research question.

Example:

### RQ1 — How severe is cross-device domain shift?

### RQ2 — How well do conventional representations transfer?

### RQ3 — Does the proposed strategy improve cross-device generalization?

### RQ4 — Which architecture components contribute most?

### RQ5 — What are the major failure modes?

Use figures strategically.

# 7. Discussion

Provide real interpretation.

Discuss:

- why results behave this way
- asymmetric transfer if observed
- impact of anatomical sensor placement
- frequency-response differences
- environmental noise
- representation learning
- practical meaning
- differences between accuracy and robustness
- relationship to real-world deployment

Avoid merely repeating results.

# 8. Limitations

Be scientifically transparent.

Potential issues:

- dataset size
- dataset heterogeneity
- label mismatch
- demographic differences
- lack of paired recordings
- incomplete clinical validation
- sensor differences
- sample-rate mismatch
- external dataset limitations

# 9. Conclusion

State exactly what has been demonstrated.

No overclaiming.

---

# 19. RESULTS SHOULD DRIVE THE CLAIMS — NOT THE OTHER WAY AROUND

This instruction is critical.

Do NOT create a problem statement and then manipulate presentation to support it.

Instead:

`Experiments → empirical findings → defensible hypothesis/problem framing → claim`

Reverse-engineer the strongest paper that the data genuinely permits.

If the original hypothesis is not supported, change the hypothesis.

---

# 20. CREATE AN EXECUTIVE SCIENTIFIC SUMMARY

Create:

`Q1_Paper_Artifact/SCIENTIFIC_STORY.md`

Include:

## One-sentence problem

## One-sentence gap

## One-sentence solution

## One-sentence key result

## One-sentence contribution

## Why a Q1 reviewer should care

## What makes this different from simply training another classifier

## Strongest evidence

## Weakest part of the paper

## Most dangerous reviewer criticism

## How the manuscript addresses it

---

# 21. CREATE A REVIEWER ATTACK DOCUMENT

Create:

`Q1_Paper_Artifact/REVIEWER_AUDIT.md`

Review the resulting work as if you are:

- IEEE Transactions reviewer
- IEEE JBHI reviewer
- Pattern Recognition reviewer
- Biomedical Signal Processing reviewer
- ICASSP reviewer

Score:

- Novelty /10
- Technical quality /10
- Experimental rigor /10
- Dataset validity /10
- Statistical rigor /10
- Reproducibility /10
- Clinical relevance /10
- Visualization quality /10
- Writing /10
- Overall publication readiness /10

Then identify:

### Critical rejection risks

### Major concerns

### Minor concerns

### Claims that are too strong

### Experiments missing

### Figures missing

### Statistical analysis missing

### Literature comparisons missing

### Reproducibility issues

### Dataset leakage risks

### What would convert this from borderline to strong accept

Be highly critical.

---

# 22. CREATE A CLAIM AUDIT

Create:

`Q1_Paper_Artifact/CLAIM_EVIDENCE_AUDIT.md`

Table:

| Manuscript Claim | Supporting Experiment | Evidence | Figure/Table | Confidence | Safe wording |

This is important for avoiding reviewer attacks.

---

# 23. CREATE A FIGURE INVENTORY

Create:

`Q1_Paper_Artifact/FIGURE_INVENTORY.md`

For every generated figure state:

- filename
- manuscript location
- research question addressed
- data source
- computation
- interpretation
- caption
- whether it uses measured values or conceptual illustration

Never mix conceptual diagrams with empirical evidence without labeling them.

---

# 24. CREATE TABLES

Generate publication-ready tables for:

1. Dataset characteristics
2. Experimental protocols
3. Main results
4. Cross-domain performance
5. Baselines
6. Ablations
7. Per-class performance where appropriate
8. Statistical comparisons if available

Do not make tables unnecessarily large.

Bold only genuinely important results.

---

# 25. ADD A REPRODUCIBILITY SECTION

Create:

`Q1_Paper_Artifact/REPRODUCIBILITY.md`

Record:

- environment
- major package versions if available
- random seeds
- split generation
- data preprocessing
- training parameters
- model selection rule
- hardware
- evaluation code
- checkpoint handling

Distinguish reproducible facts from missing information.

---

# 26. CREATE A GAP ANALYSIS

Create:

`Q1_Paper_Artifact/MISSING_EXPERIMENTS.md`

Prioritize missing experiments:

## Tier 1 — Required before journal submission

Experiments whose absence could invalidate the major claim.

## Tier 2 — Strongly recommended

Experiments likely to improve acceptance probability.

## Tier 3 — Optional enhancement

Experiments that improve depth but are not essential.

For each specify:

- research question
- exact experiment
- required data
- expected output
- appropriate figure/table
- why a reviewer would ask for it

Do NOT run Tier 1/2 expensive experiments automatically unless they are trivial analyses using existing predictions/results.

---

# 27. IMPORTANT VISUAL SCIENTIFIC ANALYSES

Where supported, especially look for opportunities to produce:

### Cross-Domain Transfer Matrix

Train domain × test domain.

### Generalization Gap Plot

Compare degradation from in-domain to unseen-domain testing.

### Domain Similarity Map

Demonstrate dataset shift.

### Representation Space Figure

Color once by class and once by acquisition domain.

A good representation should ideally encode the target task while reducing domain dependence.

### Performance vs Domain Distance

If sufficient domains exist, examine whether greater acoustic/embedding distance correlates with larger transfer degradation.

Treat this as exploratory unless statistically justified.

### Per-Domain Confusion Matrices

Determine whether domain shift changes the type of errors.

---

# 28. CHECK WHETHER THE ARCHITECTURE ACTUALLY CONTRIBUTES

Do not attribute performance to architectural components without evidence.

For each component ask:

> What experimental evidence shows this component is necessary?

Examples:

- multiple encoders
- fusion
- attention
- domain-specific projection
- classifier
- fine-tuning

If an ablation does not exist, say:

> The current experiments do not isolate the contribution of this component.

This is preferable to making unsupported claims.

---

# 29. WRITING STYLE

Write like a serious technical paper.

Avoid excessive use of:

- groundbreaking
- revolutionary
- highly effective
- remarkable
- unprecedented
- significant improvement

unless statistically or quantitatively demonstrated.

Prefer language such as:

> The proposed approach achieved...

> Results indicate...

> We observed...

> The findings suggest...

> Under the evaluated cross-domain settings...

> Within the limitations of the available datasets...

Quantitative statements should accompany important conclusions.

---

# 30. DATA INTEGRITY RULE

ABSOLUTE RULE:

**Never fabricate missing numbers.**

If only plot images exist, recover exact values only when reliably available from underlying source files.

If unavailable:

`Exact numerical value unavailable from stored experiment output.`

Do not approximate a number from a plot and present it as measured.

---

# 31. DON'T RETRAIN UNNECESSARILY

The experiments have already been run.

Prioritize:

1. inspection
2. reconstruction
3. validation
4. analysis
5. statistics
6. visualization
7. manuscript generation

Only run lightweight post-hoc analyses on existing outputs when useful.

Do NOT launch long GPU training jobs simply to make the paper look stronger.

Instead list missing training experiments in `MISSING_EXPERIMENTS.md`.

---

# 32. FINAL OUTPUT DIRECTORY

Create:

`/home/pkdas/IEEE_healthcomm_workshop/Q1_Paper_Artifact/`

Recommended structure:

```text
Q1_Paper_Artifact/
│
├── manuscript.md
├── SCIENTIFIC_STORY.md
├── REVIEWER_AUDIT.md
├── CLAIM_EVIDENCE_AUDIT.md
├── MISSING_EXPERIMENTS.md
├── REPRODUCIBILITY.md
├── FIGURE_INVENTORY.md
├── MASTER_RESULTS.csv
│
├── figures/
│   ├── fig01_problem_setting.*
│   ├── fig02_architecture.*
│   ├── fig03_dataset_characteristics.*
│   ├── fig04_domain_shift.*
│   ├── fig05_transfer_matrix.*
│   ├── fig06_generalization_gap.*
│   ├── fig07_model_comparison.*
│   ├── fig08_embedding_analysis.*
│   ├── fig09_confusion_matrix.*
│   ├── fig10_error_analysis.*
│   └── ...
│
├── tables/
│   ├── dataset_table.*
│   ├── main_results.*
│   ├── cross_domain_results.*
│   ├── ablation.*
│   └── statistical_analysis.*
│
└── analysis/
    ├── dataset_audit.md
    ├── architecture_reverse_engineering.md
    ├── experiment_inventory.md
    ├── statistical_analysis.md
    └── domain_shift_analysis.md
```

Generate only files justified by available evidence.

---

# 33. FINAL QUALITY-CONTROL PASS

After generating the artifact, stop writing and review the entire manuscript as a hostile reviewer.

For every paragraph ask:

> Is this claim supported?

For every table ask:

> What scientific question does this answer?

For every figure ask:

> Would removing this figure weaken the paper?

For every architecture component ask:

> Do we have experimental evidence that it helps?

For every performance comparison ask:

> Are the evaluation populations identical and directly comparable?

For every robustness statement ask:

> Did the experiment actually evaluate robustness or merely cross-dataset performance?

For every statistical statement ask:

> Is the statistical unit truly independent?

Then revise the manuscript.

---

# 34. MOST IMPORTANT OBJECTIVE

I do not want a polished version of a weak experiment.

I want you to discover the **strongest legitimate scientific paper hidden inside the experiments already performed**.

Reverse-engineer:

**what was done → what was discovered → why it matters → what problem it genuinely answers → what claims are defensible.**

The final manuscript should make an expert reviewer understand:

1. **What exact problem are we solving?**
2. **Why is this problem scientifically important?**
3. **Why do existing approaches not adequately answer it?**
4. **What exactly did we contribute?**
5. **Why were these particular datasets/devices needed?**
6. **What does each experiment prove?**
7. **Does the evidence really establish cross-device/domain robustness?**
8. **What insight does the work provide beyond another classification result?**
9. **How certain are the conclusions?**
10. **What remains unproven?**

Use the data to determine the scientific story.

Do not force the data to fit a predetermined story.

The target is a **Q1-journal-quality, reviewer-resistant scientific artifact**, not merely an attractive report.