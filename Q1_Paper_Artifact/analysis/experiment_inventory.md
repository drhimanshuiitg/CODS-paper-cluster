# Experiment Inventory

Every experiment family actually run and aggregated into `MASTER_RESULTS.csv`, with completion status. Cross-referenced against the ARA's `logic/experiments.md` (E01–E10).

| Family | Results root | Combinations | Status | Manuscript section |
|---|---|---|---|---|
| Main cross-device benchmark | `results/P0_device_gap` | 1,120 fold-runs (14 reps × 4 classifiers × 4 protocols × 5 folds, minus degenerate/incomplete) | Complete | §6.1–6.2, Table 3, Fig. 6/7 |
| Leave-one-encoder-out ablation | `results/P0_ablation` | 96 combinations attempted, 89 evaluable (7 skipped, incomplete fold coverage) | Complete (89/96) | §6.4, Fig. 8 |
| PCA, pre-fix | `results/P1_dimension_control` | 60 audited combinations | Complete, superseded by post-fix | §6.3 (historical) |
| PCA, post-fix | `results/P1_dimension_control_v3` | 60 combinations | Complete | §6.3, statistical_analysis.md §C |
| CORAL | `results/P1_domain_adaptation` | 160 combinations (4 reps × 4 classifiers × 5 folds × 2 protocols) | Complete | §6.3, Table 4 |
| SpO2-corroboration filter ablation | `results/P2_label_quality_ablation` | 32 combinations | Complete | §6.5, Table 6 |
| Sliding-window severity | `results/P3_sliding_window_severity` | 80/80 combinations | Complete, no significance test yet | §6.6 |
| RS_RS pooled-device protocol | `results/P0_device_gap` (RS_RS rows) | 100 combinations (5 reps × 4 classifiers × 5 folds) | Complete | §6.3, Table 5 |
| Efficiency/latency benchmark | `results/P0_efficiency` | wavlm_large + hear extension | Complete | referenced in §6.5 (HeAR cost context) |
| HuBERT vs. hubert_odi_hb significance | `results/` (dedicated script output) | 8 combinations | Complete | §6.5, Table 6 |

## Experiments NOT run (see `MISSING_EXPERIMENTS.md` for full tiered detail)

- Significance test for the sliding-window severity matched-vs-cross gap (Tier 1).
- Multiple-comparison-corrected re-statement of the 89-combination ablation and representation-leaderboard tests (Tier 1).
- Per-domain/per-class confusion-matrix figure (data exists, figure does not) (Tier 2).
- Per-subject domain-distance vs. accuracy exploratory analysis (Tier 2, requires new computation).
- Representation-level device-invariance training (contrastive/domain-adversarial) (Tier 3, requires new model training — explicitly out of scope per the master prompt's "do not launch long GPU jobs" instruction).
- External dataset / independent device-pair validation (Tier 3, no known suitable public dataset).

## Correspondence to ARA `logic/experiments.md`

This inventory matches ARA experiments E01 (main benchmark), E02 (ablation), E03/E04 (PCA pre/post-fix), E05 (CORAL), E06 (corroboration filter), E07 (HeAR), E08 (ODI/HB), E09 (sliding-window severity), E10 (efficiency) one-to-one; the RS_RS pooled protocol is folded into E01's results root rather than being a separate ARA experiment entry, since it uses the identical main-benchmark pipeline with an additional protocol value.
