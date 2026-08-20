---
name: results-audit
description: Audit completed sleep-apnea cross-device experiments before figures or paper writing. Use after computation is complete to catch leakage, unsupported claims, missing controls, statistical errors, impossible numbers, incomplete folds, and weak clinical framing.
disable-model-invocation: false
---

# Results Audit

## Goal
Act like a hostile ICASSP / top-Q1 reviewer before the manuscript is written.

## Do not write the paper yet
First validate whether the results are trustworthy and what they actually support.

## Audit checklist

### 1. Data integrity
Verify:
- subject disjointness for every fold;
- paired R/S samples never cross folds;
- train/val/test counts;
- no duplicated event IDs across partitions;
- alignment validity;
- raw sampling-rate assumptions;
- no test-derived normalization/PCA/tuning.

### 2. Experiment completeness
Cross-check plan vs `MASTER_RESULTS.csv`.
Flag:
- missing folds;
- missing seeds;
- failed GPU jobs;
- duplicate experiment IDs;
- incomplete baselines;
- missing equal-data controls;
- missing R+S->R / R+S->S cells;
- missing device probes.

### 3. Numerical sanity
Flag:
- metrics outside valid ranges;
- BA exactly 0.5 across suspiciously many runs;
- AUC/BA inconsistencies;
- near-perfect results incompatible with other evidence;
- identical result files/checksums across supposedly different experiments;
- unexpectedly zero variance;
- single-class predictions.

### 4. Statistical integrity
Verify:
- subject-level bootstrap, not window pseudo-replication;
- paired comparisons use aligned subjects;
- confidence intervals;
- multiple-comparison correction for secondary families;
- primary hypotheses separated from exploratory tests;
- no "significant in >=1 test" figure logic.

### 5. Pooled-device confound
Require:
- equal-total-N R+S control;
- R+S->R;
- R+S->S;
- explanation separating data volume from device diversity.

### 6. Device-invariance evidence
Require quantitative:
- device probe before/after;
- paired embedding distance before/after;
- paired probability disagreement before/after.

PCA/UMAP alone is insufficient.

### 7. Physiological validity
Audit:
- ODI definition;
- hypoxic/desaturation burden definition;
- whether event-local vs night-level targets are correctly separated;
- SpO2 lag/nadir analysis;
- no subject-level ODI/HB repeated as if it were a per-window signal in the proposed method.

### 8. Screening validity
For any claim containing "screening":
- verify inference does not require PSG event timestamps;
- verify full-night/fixed-window task;
- verify AHI distribution supports chosen threshold;
- verify AUROC/PR-AUC/calibration;
- verify patient-level evaluation.

### 9. Error analysis
Require:
- OA/CA/MA/Hypopnea sensitivity;
- paired R/S disagreement;
- both-device failures;
- strong vs weak desaturation failures where justified;
- non-cherry-picked case-selection rule.

### 10. Claim matrix
Create:
`paired_physio_device/artifacts/CLAIM_AUDIT.md`

Columns:
`candidate_claim | supporting_experiments | effect_size | CI/p | caveat | supported_yes_no | safe_wording`

### 11. Reviewer attack list
Create:
`paired_physio_device/artifacts/REVIEWER_ATTACKS.md`

Rank issues:
P0 fatal
P1 major
P2 moderate
P3 presentation

### 12. Final audit verdict
Choose one:
- PASS FOR PAPER DRAFT
- PASS WITH CAVEATS
- FAIL — MORE EXPERIMENTS REQUIRED

Do not invoke `/journal-paper` on FAIL.
