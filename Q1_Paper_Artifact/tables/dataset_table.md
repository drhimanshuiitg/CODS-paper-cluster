# Table 1 (manuscript Section 3). Dataset Characteristics

| Property | Value | Source |
|---|---|---|
| Source dataset | Tao et al. 2025 [17], public | `metadata/dataset_manifest_aligned.csv` |
| Total subjects in source dataset | 50 | dataset documentation |
| Subjects with reliable dual-device coverage (used) | 41 | `metadata/dataset_manifest_aligned.csv` |
| Subjects excluded | 9 (5 smartphone-only, 2 recorder-only, 2 alignment-quality) | pipeline construction logs |
| Device 1 | Bedside clinical digital recorder (Newamy V03) | dataset documentation |
| Device 2 | Consumer smartphone (OPPO Reno8) | dataset documentation |
| Native sample rate (both devices) | 8,000 Hz | dataset documentation |
| Analysis sample rate | 16,000 Hz (upsampled for SSL-encoder compatibility) | `src/sleep_quadnet/` preprocessing |
| Recording relationship | Concurrent, same room, same subject, same night | dataset documentation |
| Reference annotation | Type-I PSG, clinician-scored apnea/hypopnea events | dataset documentation |
| Total PSG-annotated events audited | 13,455 (8,916 hypopnea, 4,539 OSA) | `results/audit/spo2_corroboration_per_event.csv` |
| Cross-validation structure | 5 subject-disjoint folds; per fold, train/val/test pairwise-disjoint by subject ID (runtime-asserted every run) | `src/sleep_quadnet/` |
| Sliding-window severity subset | 6,013 five-minute epochs, 50 subjects (ground truth); 9,950-row audio manifest, 41 subjects with usable audio | `metadata/sliding_window_ahi_targets.csv`, `metadata/sliding_window_audio_manifest.csv` |

**Note on device-shift type.** Both devices record the same acoustic environment concurrently — this is transducer/hardware-level domain shift (frequency response, self-noise, gain, placement), not a difference in subject population, anatomical site, recording session, or clinical setting. Table 2 (`Q1_Paper_Artifact/tables/domain_shift_audio_stats.csv`) quantifies the resulting shift directly.
