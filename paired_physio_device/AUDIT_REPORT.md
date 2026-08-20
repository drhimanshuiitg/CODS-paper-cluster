# Audit Report — Stage 0

Per `CLAUDE_CODE_MASTER_PROMPT.md` Section A. This audit inspects raw files and executable code directly; where prose/prior documentation conflicted with what the files or code actually show, the file/code wins and the conflict is called out explicitly below.

## 1. Raw data directories

`/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/Data/{01..50}/` — 50 subject folders. Each folder contains, when complete:

| File | Confirmed content |
|---|---|
| `{sid}_annotation.json` | dict with keys `record_start`, `awake_intervals`, `events`. Each event: `{event_type, evnet_start [sic — real key name in the source file], event_duration, sleep_stage}`. |
| `{sid}_HR.csv` | `relative position, absolute position, Heart Rate (bpm)` — 1 Hz; many rows are literal `-` (missing) for at least the subject inspected. |
| `{sid}_phone.wav` | Smartphone audio, one continuous file per subject. |
| `{sid}_recorder_1.wav`, `{sid}_recorder_2.wav` | Recorder audio, **split into two sequential files per subject** (confirmed a genuine two-part continuous recording, not two physical recorders — see §3). |
| `{sid}_sleep_stage.csv` | `position (epoch), absolute position, Default Staging Set (stage)` — 30-second epochs (confirmed: consecutive rows for subject 01 are 30s apart), values `{W, N1, N2, N3, R, U}`. |
| `{sid}_SpO2.csv` | `relative position, absolute position, OSat (%)` — 1 Hz. |

50/50 subjects have `annotation.json` + `sleep_stage.csv` present and readable (verified by direct read, not by manifest inference). Dual-device (both Recorder and Smartphone) usable audio, per the existing aligned manifest, covers **41/50** subjects (§3).

## 2. Sampling rates — confirmed directly from file headers, not from prior documentation

`file` command and Python `wave` module, run directly on subject 01's raw WAVs:

```
01_recorder_1.wav: RIFF WAVE, PCM 16-bit, mono, 8000 Hz, duration 11781.568 s
01_recorder_2.wav: RIFF WAVE, PCM 16-bit, mono, 8000 Hz, duration 15844.160 s
01_phone.wav:      RIFF WAVE, PCM 16-bit, mono, 8000 Hz, duration 22257.920 s
```

**Confirms** the previously documented native sample rate (8,000 Hz, mono, 16-bit PCM) for both devices. Full per-file inventory (145 files, all 50 subjects, all devices): `paired_physio_device/audit/raw_audio_inventory.csv`.

## 3. The two-part Recorder file — resolved

For subject 01: `recorder_1` (11,781.568 s) + `recorder_2` (15,844.160 s) = 27,625.728 s combined Recorder duration, vs. 22,257.920 s for the single continuous Smartphone file. This is a **genuine, per-subject total-duration discrepancy between devices** (Recorder recorded ~5,368 s / 1.5 h longer than Smartphone for this subject) — not an artifact of the file split. `metadata/subject_inventory.csv` already records this per subject (`recorder_duration_sec`, `common_duration_sec`, `recorder_segments`) and the existing `scripts/build_aligned_manifest.py` already stitches the two-part Recorder file logically via `audio_segment_durations_json`. This machinery is trusted and reused, not rebuilt.

## 4. Subject/device pairing structure — a genuine strength found in the existing pipeline

`metadata/dataset_manifest_aligned.csv` (39,596 rows, 41 subjects) already carries a `paired_positive_id` column linking each R-device window to its exact same-event S-device counterpart — **for both positive (event) and negative (non-event) windows**, confirmed by direct inspection (19,726 negative rows resolve to 9,863 unique paired IDs, i.e. exactly R+S pairs; 0/19,726 negative rows have `paired_positive_id == own window id`, ruling out independent per-device negative sampling). This is exactly the same-event R/S positive-pair structure Section C6 requires for the paired contrastive objective — **it does not need to be built from scratch**; `paired_physio_device/audit/pair_inventory.csv` (10,325 rows, built fresh from this manifest for this audit) makes it directly consumable.

`pair_alignment_error_ms` in that CSV is computed as `|device_time_offset_start_sec(R) − device_time_offset_start_sec(S)|`, i.e. the magnitude of the per-window recorder clock-drift correction actually applied (S windows carry offset 0 by construction, since S is the alignment reference device). **This is a coarse traceability proxy, not the dataset's authoritative alignment-quality metric** — median 2.2 s, mean 6.1 s, p95 26.6 s, max 67.1 s, and the spread reflects genuine within-night clock drift magnitude (larger later in the night), not necessarily alignment failure. The authoritative, validated alignment-quality metrics already exist and are unchanged by this audit: `metadata/device_alignment_models.csv` (`max_absolute_fit_residual_sec`, `median_anchor_correlation`, `low_confidence` flag) and `metadata/device_alignment_dense_summary.csv` (`reliable_fraction`, `alignment_usable`). See `DATA_INTEGRITY_REPORT.md` §2 for a specific discrepancy found between the per-subject `alignment_usable` flag and actual manifest inclusion.

## 5. PSG annotations — event types confirmed, subtypes confirmed ABSENT

Direct read of all 50 `{sid}_annotation.json` files: every event's `event_type` field takes exactly one of two values, **`osa` (n=4,539) or `hypo` (n=8,916)**, project-wide. **No Obstructive/Central/Mixed apnea subtype field, and no separate "normal" event entries, exist anywhere in the raw annotation source.** ("Normal"/non-event windows in the manifest are constructed by the pipeline as duration-matched negative sampling around annotated events, not as a distinct annotated PSG category.)

**This directly affects Section B/I of the master prompt** — see `DATA_INTEGRITY_REPORT.md` §1 for the full data-integrity consequence and the corrected task definition.

## 6. SpO2 availability

`{sid}_SpO2.csv` present and populated (1 Hz `OSat (%)`) for all 50 subjects inspected via `metadata/odi_hypoxic_burden.csv`, which already has a computed `odi`/`hypoxic_burden`/`sleep_hours` row for every subject in that file (spot-checked subjects 01, 02). SpO2 is real, dense, and usable for event-local auxiliary targets (Section D2).

## 7. Existing code inspected and trusted (not rebuilt)

| Component | File | Status |
|---|---|---|
| Device alignment / clock drift | `scripts/estimate_device_alignment_dense.py`, `scripts/build_aligned_manifest.py` | Present, outputs verified against raw file durations (§3) |
| Manifest / window extraction | `scripts/build_aligned_manifest.py` → `metadata/dataset_manifest_aligned.csv` | Present, paired structure confirmed (§4) |
| Fold assignment | `metadata/subject_folds_5cv_aligned.csv` | Present, subject-level, 5-fold |
| Leakage safeguard | `src/sleep_quadnet/evaluation.py:96` (`raise AssertionError("Subject leakage in split construction")`), `src/sleep_quadnet/metadata.py:406` | Present, confirmed by direct grep of the executable code (not prose) |
| Feature extraction | `src/sleep_quadnet/features.py`, cached at `/scratch/pkdas/IEEE_healthcomm_workshop/cached_features/{classical,hubert,wavlm,wavlm_large,wav2vec2,data2vec_audio,data2vec_spectrogram,hear,odi_hb}/` | Present and cached — confirmed by directory listing |
| Classifiers | `src/sleep_quadnet/evaluation.py` (cuML SVM/RF, XGBoost-CUDA, Torch MLP) | Present |
| Sleep-QuadNet fusion / full_fusion | `src/sleep_quadnet/features.py`, `src/sleep_quadnet/advanced.py` | Present — retained as **baseline only** per Section C |
| PCA / CORAL domain adaptation | `scripts/run_coral.py`, `scripts/run_pca_fix_significance.py`, `scripts/_pca_fix_smoketest.py` | Present |
| Significance testing | `scripts/run_statistics.py`, `scripts/run_ablation_significance.py`, `scripts/run_corroboration_significance.py`, `scripts/run_hubert_odi_hb_significance.py` | Present — paired subject-level bootstrap convention, reused rather than reimplemented |
| Previous manuscript/figures | `paper/conference_101719.tex`; `Q1_Paper_Artifact/` (this session, prior task) | Present, treated as the **preserved baseline study** per the master prompt's Role instruction — not modified by this redesign |

## 8. What this audit did NOT do

- Did not re-derive or modify the device-alignment model itself (Section A: "Do not rewrite source data" — and the existing alignment code is validated, not broken).
- Did not touch, overwrite, or delete any existing `results/`, `metadata/`, or `Q1_Paper_Artifact/` output. All new work is confined to `paired_physio_device/`.
- Did not run any GPU training. The only GPU job run this stage was the preflight check (`GPU_EXECUTION_PLAN.md`).
