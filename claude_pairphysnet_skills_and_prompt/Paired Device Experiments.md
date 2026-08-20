---
name: paired-device-experiments
description: Run or extend the paired Recorder/Smartphone sleep-respiratory experiments, including PairPhysNet, device probes, SpO2 auxiliary supervision, pooled-data controls, event subtype analysis, and annotation-free night-level screening. Use when implementing, resuming, or extending the experiment pipeline.
---

# Paired Device Experiments

## Purpose
Use this skill for the paired Recorder/Smartphone sleep-respiratory study. The goal is to distinguish respiratory physiology from acquisition-device style and test whether this improves bidirectional cross-device generalization and night-level screening.

## Rules
- Preserve all existing baseline results.
- Work under `paired_physio_device/`.
- Subject-disjoint folds are mandatory.
- Same-event R/S pairs must remain in the same fold.
- Never use PSG-Audio; it is not currently available.
- Never fabricate missing results.
- Never use test labels/statistics for tuning.
- No silent CPU fallback. Use `/gpu-only-4way`.
- Keep event-level "respiratory event classification" distinct from night-level "OSA screening."

## Required stages
1. Audit dataset and raw file metadata.
2. Validate R/S alignment.
3. Reproduce trusted baseline results only as needed.
4. Compute paired signal/device-shift measures.
5. Train subject-disjoint device probes.
6. Implement PairPhysNet:
   - shared SSL encoder;
   - physiology-content projection;
   - device-style projection;
   - paired contrastive loss;
   - device adversarial head;
   - disentanglement loss;
   - local SpO2 auxiliary targets.
7. Run A1–A5 ablations.
8. Run equal-data pooled controls.
9. Run subtype analysis.
10. Run error phenotyping and qualitative paired-case selection.
11. Run annotation-free night-level screening.
12. Run subject-level statistics and multiple-comparison correction.
13. Update `MASTER_RESULTS.csv`.

## PairPhysNet outputs
Event head:
- event vs non-event
- subtype analysis if class counts support it

Physiology auxiliary heads:
- desaturation probability
- desaturation amplitude
- event-to-nadir delay
- event-associated desaturation area

Night-level heads:
- AHI >= 15 if cohort supports it
- AHI >= 30 if cohort supports it
- AHI regression
- ODI regression
- hypoxic/desaturation burden regression

## Required negative-result behavior
If PairPhysNet does not beat CE-only or frozen baselines:
- preserve the result;
- determine whether device-probe accuracy nevertheless changed;
- analyze paired embedding distance and prediction disagreement;
- do not rename the result as a success.

## Completion gate
Before paper generation, invoke:
`/results-audit`

Do not invoke `/journal-paper` until the results audit passes.
