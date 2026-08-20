---
name: journal-paper
description: Generate the final journal-level manuscript, LaTeX source, tables, references, supplementary material, and compiled PDF for the paired cross-device sleep-apnea study after the results audit and final figures are complete. Use only after the experiment matrix is finalized.
disable-model-invocation: true
---

# Journal Paper Generation

## Invocation gate
Before writing:
1. Read `paired_physio_device/artifacts/CLAIM_AUDIT.md`
2. Read `paired_physio_device/artifacts/REVIEWER_ATTACKS.md`
3. Confirm results-audit verdict is PASS FOR PAPER DRAFT or PASS WITH CAVEATS.
4. Confirm `MASTER_RESULTS.csv`, final tables, and `FIGURE_INDEX.md` exist.
5. If not, STOP and explain what is missing.

## Target standard
Write at the level of a strong biomedical signal-processing / audio / sleep-medicine Q1 journal submission, with ICASSP-level methodological precision.

Do not write promotional language.
Do not call negative/near-chance results "robust."
Do not claim novelty that the evidence does not establish.

## Central paper structure
1. Title
2. Abstract
3. Keywords
4. Introduction
5. Related Work
6. Dataset and Clinical Targets
7. Problem Formulation
8. Proposed PairPhysNet Method
9. Experimental Protocol
10. Results
11. Error / Case Analysis
12. Clinical Screening Analysis
13. Discussion
14. Limitations
15. Conclusion
16. Data/Code Availability
17. Ethics/Funding/Acknowledgment placeholders if details are unavailable
18. References
19. Supplementary material

## Required conceptual separation
### Event-level
Call:
"PSG-annotated sleep respiratory event classification"

Do not call this alone "OSA screening."

### Night-level
Only call "sleep-apnea screening" when inference does not use PSG event timestamps and predicts a patient/night endpoint.

## Required contribution framing
Prefer contributions that are actually supported, such as:
- paired acquisition-device characterization;
- quantitative device shortcut evidence;
- paired-device representation learning;
- physiology-guided auxiliary learning;
- cross-device event-level improvement;
- annotation-free night-level screening if successful;
- paired device-consistency analysis.

Do not make Sleep-QuadNet/full_fusion the proposed method.

## ODI / SpO2 / burden writing rules
- ODI and burden are night/subject-level unless a local metric is explicitly defined.
- Event-local shaded SpO2 quantity should be called "event-associated desaturation area" unless it is exactly the validated whole-night burden definition.
- Distinguish physiological auxiliary supervision from deployment inputs.
- If SpO2 is used only during training, state that inference is audio-only.
- Do not say correlation with event count "clinically validates" hypoxic burden.

## Results writing rules
Every headline claim must cite exact:
- table/figure;
- metric;
- CI/effect size;
- statistical test where relevant.

Report negative results naturally.

Do not average R->S and S->R into one number when direction matters.

## Discussion requirements
Explicitly answer:
- what was learned about device shift?
- does PairPhysNet reduce device information?
- which event types remain difficult?
- what happens to prediction consistency across R/S?
- does physiology supervision help?
- does the method truly improve night-level screening?
- what does pooled-device training prove after equal-data control?
- what does this study NOT establish?

## Limitations must include, as applicable
- one cohort;
- one paired device setting;
- no unseen external device;
- no tracheal validation;
- event-centered annotation privilege for Task 1;
- cohort composition limits for AHI thresholds;
- weak subtype counts;
- SpO2 measurement/definition limitations;
- no PSG-equivalent diagnosis claim.

## References
Never invent references, authors, DOIs, URLs, or venue details.
Use:
- verified existing `.bib` entries;
- verified source PDFs/docs;
- web search only if available and necessary.
If a reference cannot be verified, insert an explicit TODO rather than fabricate it.

## Figures
Use only files listed in `FIGURE_INDEX.md`.
Do not regenerate figures opportunistically during manuscript writing unless the figure skill is invoked again and the figure index is updated.

## Tables
Generate:
- cohort/device table;
- task/label definition table;
- model/ablation table;
- direction-specific event results;
- device-probe table;
- physiology auxiliary results;
- equal-data pooled control;
- subtype table;
- night-level screening table;
- compute/deployment table if measured.

## PDF build
Create:
`paired_physio_device/manuscript/`

with:
- `main.tex`
- `references.bib`
- `sections/*.tex`
- `figures/` links/copies
- `tables/`
- `supplementary.tex`
- `BUILD.md`

Compile to:
`paired_physio_device/manuscript/main.pdf`

Use a reproducible LaTeX build command (latexmk preferred if available).
Check:
- no missing references;
- no undefined citations;
- no overfull boxes that materially affect readability;
- figure resolution;
- table width;
- page numbers;
- bibliography rendering.

## Final reviewer simulation
After compiling, write:
`paired_physio_device/manuscript/FINAL_REVIEWER_SIMULATION.md`

Act as:
- Reviewer 1: ICASSP signal-processing expert
- Reviewer 2: sleep-medicine/clinical reviewer
- Reviewer 3: ML/domain-generalization reviewer

Each reviewer must give:
strengths, weaknesses, required revisions, score, confidence.

Then revise the manuscript only where revisions are supported by existing evidence.
Do not invent new results during revision.
