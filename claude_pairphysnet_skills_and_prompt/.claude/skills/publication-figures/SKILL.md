---
name: publication-figures
description: Generate final journal/ICASSP-quality scientific figures from completed cross-device sleep-apnea results, including paired waveforms/spectrograms, SpO2 and desaturation physiology, device-shift plots, architecture, success/failure case atlas, screening plots, and statistical figures. Use only after results are complete or for explicitly marked preliminary figures.
disable-model-invocation: false
---

# Publication Figures

## Core rule
Every data figure must come from real experiment/result files.
Never fabricate values, traces, confidence intervals, labels, attention maps, or "representative" cases.

## Output formats
For every main figure save:
- vector PDF
- SVG where practical
- PNG at 600 dpi

Use consistent:
- font family/size;
- line width;
- panel labels `(a), (b), ...`;
- terminology;
- R/S device names;
- event subtype names;
- axis units.

Use a color-blind-safe palette and ensure grayscale readability.
Do not encode meaning by color alone.

## Figure set

### Figure 1 — Problem setting
Show:
same subject / same night / same room
-> simultaneous Recorder + Smartphone audio
-> acquisition-device domain shift
-> potential cross-device prediction disagreement

Do not claim "microphone hardware only"; use "acquisition-device domain."

### Figure 2 — PairPhysNet architecture
Draw a clear scientific schematic:
Recorder waveform -> shared encoder
Smartphone waveform -> shared encoder
-> physiology-content branch `c`
-> device-style branch `d`
-> paired contrastive objective
-> device adversarial head on `c`
-> device classifier on `d`
-> disentanglement
-> SpO2 auxiliary head
-> event head
-> night aggregator
-> AHI/severity/ODI/burden heads

Use equations/labels sparingly.

### Figure 3 — Paired device-domain characterization
For exact same-event R/S pairs:
- PSD with subject-level CI
- paired spectral difference
- optional coherence
- paired scalar acoustic differences
- pre- and post-normalization amplitude analysis

Do not rely only on independent-window Mann-Whitney tests.

### Figure 4 — Physiologically aligned paired example
One selected paired event:
Top:
Recorder waveform
Recorder log-Mel spectrogram

Middle:
Smartphone waveform
Smartphone log-Mel spectrogram

Bottom shared physiology:
PSG event annotation and subtype
SpO2 trace
baseline
nadir
desaturation amplitude
event-associated desaturation area

Side box:
R prediction
S prediction
baseline model
PairPhysNet model if available
whole-night ODI/burden only if clearly labeled as NIGHT-level values

Do NOT label one local shaded desaturation area as whole-night hypoxic burden.

### Figure 5 — Device shortcut
Combine:
- exploratory PCA/UMAP
- quantitative device-probe BA/AUC with CI
- event-probe BA/AUC
Show baseline vs PairPhysNet physiology-content and device-style branches.

### Figure 6 — Direction-specific transfer forest plot
Rows: model variants
Panels:
R->R vs R->S
S->S vs S->R
Show point estimate + 95% subject-bootstrap CI.
Do not average directions too early.

### Figure 7 — Proposed-method ablation
CE only
CE + Pair
CE + DANN
CE + Pair + DANN
Full PairPhysNet
Show cross-device BA/AUC with corrected CIs.

### Figure 8 — Paired embedding invariance
Distribution of same-event R/S cosine distance:
frozen baseline
CE only
paired contrastive
full PairPhysNet

Also show random cross-event R/S distances as reference.

### Figure 9 — Prediction consistency
Scatter:
`p_R` vs `p_S`
diagonal `y=x`

Report:
mean/median `|p_R-p_S|`
paired consistency before vs after method.

### Figure 10 — Event subtype heatmap
Rows:
OA, CA, MA, Hypopnea
Columns:
R->R, S->S, R->S, S->R
Optional second panel for proposed model.

### Figure 11 — Equal-data pooled control
R-only N
S-only N
R+S balanced N total
R+S full 2N
Test separately on R and S.
Show CIs.

### Figure 12 — Correct/misclassified paired-case atlas
Four rows/cases:
1. R correct / S correct
2. R correct / S wrong
3. R wrong / S correct
4. R wrong / S wrong

Each case contains:
R waveform/spec
S waveform/spec
SpO2/PSG
true label/subtype
p_R/p_S
baseline/proposed predictions

Cases MUST be chosen using the predefined machine-readable selection rule, not manually cherry-picked.

### Figure 13 — Event-aligned SpO2 physiology
By subtype:
mean SpO2 delta vs time around event end
subject-level CI
mark median nadir time
This justifies any event-to-desaturation association window.

### Figure 14 — Night-level screening ROC/PR
Only if screening task is valid.
Show R->R, S->S, R->S, S->R and proposed model.

### Figure 15 — AHI prediction
If regression is valid:
- PSG AHI vs predicted AHI
- separate Bland-Altman figure or panel
- device-paired predicted AHI consistency

### Figure 16 — Supported vs unsupported model capability
Create a restrained summary graphic:
Supported by evidence
Limited/unstable
Not demonstrated

Do not use marketing language.

## Case-selection rules
Persist:
`figures/case_atlas/case_selection.json`

Include:
- eligibility set
- sorting metric
- selected paired_event_id
- why selected
- whether it is median/quantile representative

## Statistical display rules
- Use subject-level CI for headline performance.
- Mark multiple-comparison correction method.
- Do not use stars without defining test/correction.
- Prefer effect size + CI over p-value-only graphics.

## Figure QA
Before finalizing:
- no clipped labels;
- no tiny fonts;
- no rasterized text in vector PDF;
- legends do not hide data;
- units present;
- panel references match manuscript;
- all numbers reproducible from saved CSV/JSON.

Create:
`paired_physio_device/figures/FIGURE_INDEX.md`
mapping each final figure to:
source files, script, config, experiment IDs, and manuscript claim.
