# Evidence Index

No quantitative figures (plots) exist yet for this results snapshot — `results/audit/figures/spo2_corroboration_examples.png` is a qualitative 3-panel spectrogram+SpO2 illustration, not a data-point figure suitable for extraction into `evidence/figures/`, so the figures section is empty rather than populated with a non-quantitative placeholder.

## Tables

| File | Source | Claims | Description |
|------|--------|--------|-------------|
| [tables/table1_main_benchmark_summary.md](tables/table1_main_benchmark_summary.md) | `results/P0_device_gap` | C01, C02, C03, C08, C09 | Per-representation matched/cross-device mean BA/F1/AUC/MCC, 14 representations |
| [tables/table2_pca_fix_significance.md](tables/table2_pca_fix_significance.md) | `results/P1_statistics_pca_fix` | C04, C05 | Full 60-row paired bootstrap, PCA post-fix vs. pre-fix |
| [tables/table3_sliding_window_severity_full.md](tables/table3_sliding_window_severity_full.md) | `results/P3_sliding_window_severity` | C11 | 80/80 completed combinations — run finished 2026-08-19, all 4 protocols |
| [tables/table4_ablation_significance.md](tables/table4_ablation_significance.md) | `results/P0_ablation_statistics` | C10 | Full 89-row paired bootstrap, leave-one-out vs. full_fusion |
| [tables/table5_coral_results.md](tables/table5_coral_results.md) | `results/P1_domain_adaptation` | C06 | Full 30-row CORAL-aligned results |
| [tables/table6_corroboration_significance.md](tables/table6_corroboration_significance.md) | `results/P2_statistics` | C07 | Full 32-row paired bootstrap, filtered vs. unfiltered training |
| [tables/table7_efficiency_benchmark_partial.md](tables/table7_efficiency_benchmark_partial.md) | `results/P0_efficiency` | (supports deployment discussion around C01/C02) | 6/14 representations, partial coverage (G2) |
| [tables/table8_odi_hypoxic_burden_per_subject.md](tables/table8_odi_hypoxic_burden_per_subject.md) | `metadata/odi_hypoxic_burden.csv` | C09, C11 | 50-subject per-subject ODI/hypoxic-burden values |
| [tables/table9_spo2_corroboration_audit_summary.md](tables/table9_spo2_corroboration_audit_summary.md) | `results/audit/spo2_corroboration_per_event.csv` | C07 | 13,455-event corroboration-rate audit, by event type |
| [tables/table10_hubert_vs_hubert_odi_hb_significance.md](tables/table10_hubert_vs_hubert_odi_hb_significance.md) | `results/P0_statistics_hubert_vs_hubert_odi_hb` | C09 | Direct paired bootstrap, hubert vs hubert_odi_hb (added post-review, F03) |
| [tables/table11_refreshed_significance_full_candidate_set.md](tables/table11_refreshed_significance_full_candidate_set.md) | `results/P0_statistics_v2` | C01, C02, C08 | Refreshed significance test, full 14-representation candidate set (added post-review, F05/G1) |
| [tables/table12_refreshed_confidence_intervals.md](tables/table12_refreshed_confidence_intervals.md) | `results/P0_statistics_v2` | C01, C02 | Refreshed per-representation CIs under the primary classifier only (added post-review, F05/G1) |

## Figures

(none — see note above)
