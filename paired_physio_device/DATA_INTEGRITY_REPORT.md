# Data Integrity Report — Stage 0

Findings that change what can be scientifically claimed or executed, discovered by inspecting raw files and executable code directly (per Section A's explicit instruction to not trust prose over ground truth). Each finding states the discrepancy, the evidence, and the required correction to the task definitions in `EXPERIMENT_PLAN.md`.

---

## Finding 1 — No apnea subtype (OA/CA/MA) annotation exists in the raw dataset

**Master prompt expectation (Section B, Task 1 and Section I):** secondary subtype analysis / recall broken out by Obstructive apnea (OA), Central apnea (CA), Mixed apnea (MA), and Hypopnea.

**Evidence:** direct read of all 50 `{sid}_annotation.json` files. Every event's `event_type` field is one of exactly two string values project-wide: `osa` (4,539 events) or `hypo` (8,916 events). No field distinguishing obstructive/central/mixed apnea exists anywhere in the source JSON, and no other raw file (SpO2, HR, sleep-stage CSVs) carries an apnea-subtype label either.

**Consequence:** the OA/CA/MA/Hypopnea four-way (or five-way with Normal) subtype breakdown specified in Sections B and I **cannot be evaluated with this dataset** — doing so would require fabricating labels that do not exist in the source. This is not a processing bug; it is a property of the annotation the dataset actually provides.

**Correction applied in `EXPERIMENT_PLAN.md`:** Task 1's subtype analysis is redefined as the only subtype breakdown the data actually supports — **{osa, hypo}** — and Section I's subtype × protocol heatmap is built for `{osa, hypo}` only, not the five-category scheme in the master prompt's literal text. This is stated plainly rather than silently narrowed.

---

## Finding 2 — The primary night-level screening threshold the master prompt suggests (AHI≥15) has an unusably small negative class in this cohort

**Master prompt expectation (Section B, Task 2):** "primary: moderate-to-severe OSA screening if cohort distribution supports AHI≥15... Before defining thresholds, audit subject-level AHI distribution and report counts. Do not create a binary task with unusably small class counts."

**Evidence:** a real per-subject AHI was computed directly from raw data — annotated `osa`+`hypo` event count divided by true sleep hours (non-W, non-U 30-second epochs from `{sid}_sleep_stage.csv`) — for all 50 subjects, then restricted to the 41 subjects with usable dual-device audio:

| AHI severity bin | n subjects (of 41 dual-device-usable) |
|---|---|
| Normal (<5) | 0 |
| Mild (5–15) | 7 |
| Moderate (15–30) | 9 |
| Severe (≥30) | 25 |

| Threshold | Positive | Negative | Negative fraction |
|---|---|---|---|
| **AHI ≥ 15** | 34 | **7** | **17%** |
| **AHI ≥ 30** | 25 | 16 | 39% |

**Consequence:** this cohort contains **zero** subjects with AHI<5 and only **7** subjects total below AHI 15, across all 41 usable subjects. Under mandatory subject-disjoint 5-fold CV, an AHI≥15 classification task would put roughly 1 negative-class subject per fold on average, with some folds very likely containing 0 negative subjects at all — exactly the "unusably small class count" scenario Section B explicitly instructs against. This is a genuine property of the recruited cohort (apnea-clinic referral population, not population-screening), not a processing artifact.

**Correction applied in `EXPERIMENT_PLAN.md`:** **AHI≥30 (severe) is promoted to the primary night-level screening target** (25/16 split — usable, though still moderately imbalanced, under 5-fold subject-disjoint CV). AHI≥15 is retained only as an explicitly-labeled exploratory/secondary target with the small-negative-class caveat stated in every table/figure that reports it, per Section O's "not supported unless separately demonstrated" framing. AHI regression (Section K) is retained as an additional, threshold-independent target that sidesteps this specific class-count problem.

---

## Finding 3 — `alignment_usable=False` for a subject that is nonetheless included in the trusted manifest

**Evidence:** `metadata/device_alignment_dense_summary.csv` row for subject 01: `alignment_usable=False` (`reliable_fraction=0.51`, `median_reliable_correlation=0.75`). Yet subject 01 has 39,596-row-manifest-confirmed dual-device windows in `metadata/dataset_manifest_aligned.csv` (used throughout the trusted baseline results).

**Resolution, not a bug:** `scripts/build_aligned_manifest.py`'s actual inclusion rule (confirmed by reading the code, not assuming) is **window-level, not subject-level**: "both reference endpoints must lie in one segment; segments split at anchor gaps >1800 s or adjacent lag changes >5 s; intervals between segments are excluded" (`build_aligned_manifest.py:140`). A subject can have a globally unreliable `alignment_usable` flag (computed over the *whole night*) while still contributing individually-validated windows from the reliable segments of that same night. This is a legitimate, more granular methodology than the whole-night summary flag alone suggests.

**Consequence:** the previously-used framing "41/50 subjects, 9 excluded" (from this project's prior Q1 manuscript work) is directionally correct as a subject count but is a simplification — the real inclusion criterion operates at the window/segment level within a subject, not as a single per-subject accept/reject decision. This nuance is now documented here so it is not silently lost.

**Correction applied:** No change to which windows are used (the existing, already-validated window-level rule is trusted and reused). `EXPERIMENT_PLAN.md` and any future manuscript text describing the cohort should say "41 subjects contribute at least one alignment-validated window" rather than implying a simple whole-subject accept/reject gate.

---

## Finding 4 — `pair_alignment_error_ms` in the newly built `pair_inventory.csv` is a coarse proxy, not the authoritative alignment-quality metric

See `AUDIT_REPORT.md` §4. Documented here as a data-integrity item because the column name (mandated verbatim by Section A) could otherwise be mistaken for a validated per-pair residual. The authoritative metrics remain `metadata/device_alignment_models.csv` (`max_absolute_fit_residual_sec`) and `metadata/device_alignment_dense_summary.csv`. No fabrication occurred — the column is computed exactly as documented, just from a coarser signal than its name might suggest, and that limitation is stated explicitly rather than left implicit.

---

## Finding 5 — HR (heart rate) channel is present but sparsely populated

**Evidence:** `{sid}_HR.csv` for subject 01 (first 4 rows inspected) contains literal `-` (missing) for the `Heart Rate (bpm)` field. Not yet audited across all 50 subjects for overall missingness rate — flagged as an open item, not a claim of total unavailability.

**Consequence:** HR is not currently used anywhere in the master prompt's specified architecture (Sections C/D name SpO2, not HR, as the auxiliary-supervision signal), so this does not block any planned Stage. Recorded here for completeness and in case a future extension wants to use HR. **NOT FULLY AUDITED — full missingness-rate-per-subject characterization is not yet done.**

---

## Summary of corrections carried into `EXPERIMENT_PLAN.md`

1. Task 1 subtype analysis: `{osa, hypo}` only, not OA/CA/MA/Hypopnea.
2. Task 2 primary night-level threshold: **AHI≥30**, not AHI≥15 (AHI≥15 demoted to secondary/exploratory with an explicit small-negative-class caveat).
3. Cohort description: "41 subjects contribute ≥1 alignment-validated dual-device window" rather than an unqualified "41/50 subjects passed alignment."
4. `pair_alignment_error_ms` documented as a coarse per-window clock-correction-magnitude proxy; `device_alignment_models.csv`/`device_alignment_dense_summary.csv` remain the authoritative alignment-quality source for any inferential claim.
