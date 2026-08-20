# Device-alignment audit addendum

Date: 2026-08-17 (Asia/Calcutta)

The first read-only RMS-envelope cross-correlation check found that Recorder and Smartphone files are not sample-aligned by their raw file-relative timestamps for every subject. Across the 43 subjects with both devices, a single 300-second midpoint diagnostic produced a median best lag of -0.8 seconds, but only 29 subjects were within 5 seconds and the maximum absolute lag was 90.6 seconds. The initial unaligned classical extraction was stopped before completion; its partial cache was preserved under `cached_features_quarantine_unaligned/` and was never used for an experiment result.

A dense alignment audit was then run without annotations or class labels:

- native 8-kHz audio was converted to 0.1-second log-RMS envelopes by streaming the original files;
- 900-second cross-correlation windows were evaluated every 600 seconds;
- Recorder-minus-Smartphone lag was searched over +/-300 seconds;
- anchors with envelope correlation below 0.45 or a boundary peak were rejected;
- continuous alignment segments were split whenever consecutive reliable anchors were more than 1,800 seconds apart or their lag changed abruptly by more than 5 seconds (consistent with a recording-segment discontinuity);
- device-specific Recorder timestamps were obtained by piecewise-linear interpolation only within a continuous alignment segment;
- a window was retained only when both reference endpoints lay within the same supported segment; uncertain intervals between segments were excluded.

Subjects 34 and 35 had no reliable paired acoustic alignment and were excluded from controlled device comparisons. The corrected benchmark therefore contains 41 subjects, 19,798 logical windows, and 39,596 device rows. It excludes 1,450 logical windows belonging to subjects 34/35 and 926 additional windows outside reliable synchronized regions. The resulting final manifest has 9,935 positive and 9,863 negative logical windows.

Independent validation of the corrected materialization found:

- all logical windows have exactly one Recorder and one Smartphone row;
- all 25 fold/protocol combinations are subject-disjoint and contain both classes;
- no negative window overlaps any respiratory event or awake interval;
- the same fixed aligned fold assignment is used by all representations and classifiers;
- alignment uses no event labels and performs no learned task-model preprocessing.

Authoritative corrected inputs:

- `metadata/dataset_manifest_aligned.csv`
- `metadata/subject_folds_5cv_aligned.csv`
- `metadata/fold_protocols_5cv_aligned.csv`
- `metadata/device_alignment_dense_anchors.csv`
- `metadata/aligned_metadata_validation.json`

The original unaligned metadata files are retained for auditability and are not used by the corrected experiment jobs.
