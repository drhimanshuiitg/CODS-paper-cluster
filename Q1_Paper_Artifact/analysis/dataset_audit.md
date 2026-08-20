# Dataset Audit

## Provenance and structure

Source: Tao et al. 2025 [17], public dual-device sleep-audio dataset. 50 subjects total; 41 retained here with reliable dual-device coverage (5 smartphone-only, 2 recorder-only, 2 excluded for alignment quality). Both devices — a bedside clinical recorder (Newamy V03) and a consumer smartphone (OPPO Reno8) — recorded concurrently, same room, same subject, same night, at native 8,000 Hz. This is the structural precondition that makes a genuine device-shift (rather than population/session/environment-shift) claim possible, and it is confirmed directly rather than assumed: `metadata/dataset_manifest_aligned.csv` records per-subject device coverage and the alignment-quality exclusions.

## Is this cross-sensor, cross-population, cross-environment, or cross-device?

**Cross-device (transducer/hardware) shift, specifically — not cross-population, not cross-environment, not cross-session.** Same subjects, same recording session, same physical room, same underlying physiological events (both devices are time-aligned to the same PSG-scored event timeline via a fitted clock-drift correction). The only varying factor across the R/S protocols is the recording hardware itself (frequency response, self-noise floor, gain, placement convention). This distinction is load-bearing for every downstream claim in the manuscript (Section 3.1 states it explicitly) — a cross-population or cross-environment shift would license different, weaker causal claims about *what* is driving the accuracy gap.

## Label construction

Binary window-level label (apnea/hypopnea event present vs. duration-matched absent), derived from PSG clinician annotation. Manifest construction balances negatives against positives per subject, not against the raw highly-imbalanced continuous-recording ratio — this is a deliberate, documented design choice, not an artifact discovered post-hoc.

## Leakage check

A runtime assertion on every single fold-run checks pairwise disjointness of train/validation/test subject-ID sets (not merely at fold-file construction time). This was adopted directly in response to a confirmed historical leakage bug in an earlier, now-deprecated script (`device_robust_sleep_apnea_experiments_v4.py`), whose output is not used anywhere in this artifact. No leakage has been found in the current pipeline's outputs across the ~2,720 fold-runs aggregated into `MASTER_RESULTS.csv`.

## Class balance and event-annotation quality

13,455 PSG-annotated events audited against objective SpO2 desaturation timing (45s lag tolerance): 82.05% overall corroboration rate (OSA 96.96%, hypopnea 74.46% — a difference expected under AASM scoring convention, where hypopnea can be scored on nasal-flow or chest-effort criteria without a desaturation, not necessarily annotator error). Per-subject corroboration rate ranges widely: 8.4%–99.3% across 41 subjects with SpO2+annotation coverage — this per-subject variance is itself a finding (some subjects' annotations are far less desaturation-corroborated than others), independent of the corroboration-filter training ablation's own (negative) result (Section 6.5 of `manuscript.md`).

## What is NOT evaluated / NOT known

- Subject demographic composition (age, sex, BMI, apnea severity distribution) is not reconstructed in this artifact — `metadata/dataset_manifest_aligned.csv` was not exhaustively cross-referenced for these fields in this pass. **NOT EVALUATED.**
- Recording environment acoustic properties (room size, ambient noise floor) beyond what is inferable from the domain-shift statistics themselves are not independently documented. **NOT EVALUATED.**
- No independent second cohort or device pair exists in this project to validate generalization of any dataset-level finding.
