# Statistical Analysis Table (manuscript Section 6.2/6.3)

## A. Refreshed 95% CIs, primary classifier (svm_rbf), 5 principal representations

**Source**: `results/P0_statistics_v2/confidence_intervals.csv` (post review-finding F05 refresh — validation-selected primary classifier and candidate set).

| Representation | Protocol | Point estimate | 95% CI |
|---|---|---|---|
| classical | R_R | 0.5474 | [0.5250, 0.5703] |
| classical | S_S | 0.5553 | [0.5311, 0.5784] |
| classical | R_S | 0.5005 | [0.4996, 0.5020] |
| classical | S_R | 0.4920 | [0.4840, 0.4986] |
| wavlm_large | R_R | 0.6113 | [0.5847, 0.6366] |
| wavlm_large | S_S | 0.6123 | [0.5867, 0.6387] |
| wavlm_large | R_S | 0.5336 | [0.5217, 0.5467] |
| wavlm_large | S_R | 0.5320 | [0.5147, 0.5500] |
| data2vec_fusion | R_R | 0.5749 | [0.5472, 0.6016] |
| data2vec_fusion | S_S | 0.6000 | [0.5744, 0.6272] |
| data2vec_fusion | R_S | 0.5325 | [0.5160, 0.5514] |
| data2vec_fusion | S_R | 0.5249 | [0.5089, 0.5405] |
| full_fusion | R_R | 0.5783 | [0.5538, 0.6027] |
| full_fusion | S_S | 0.5952 | [0.5680, 0.6207] |
| full_fusion | R_S | 0.5195 | [0.5029, 0.5397] |
| full_fusion | S_R | 0.5356 | [0.5188, 0.5540] |
| full_fusion_plus_hear | R_R | 0.5775 | [0.5521, 0.6030] |
| full_fusion_plus_hear | S_S | 0.6000 | [0.5727, 0.6278] |
| full_fusion_plus_hear | R_S | 0.5096 | [0.5016, 0.5205] |
| full_fusion_plus_hear | S_R | 0.5347 | [0.5174, 0.5525] |

**Key paired-bootstrap tests derived from this table (2,000 resamples, subject-level):**
- full_fusion vs wavlm_large, R→S: p = 0.217 (not significant)
- full_fusion vs wavlm_large, S→R: p = 0.685 (not significant)
- full_fusion R→R vs R→S (matched vs cross): p < 0.001 (significant)
- full_fusion S→S vs S→R (matched vs cross): p < 0.001 (significant)

## B. Leave-one-encoder-out ablation, significance summary

**Source**: `results/P0_ablation_statistics/ablation_vs_full_fusion.csv`, 89/96 evaluable combinations (7 skipped for incomplete fold coverage).

| Outcome | Count | Fraction |
|---|---|---|
| CI excludes zero (significant), ablated variant higher | 4 | 4.5% |
| CI excludes zero (significant), ablated variant lower | 3 | 3.4% |
| CI includes zero (not significant) | 82 | 92.1% |
| **Total evaluable** | **89** | **100%** |

Mean point-difference (ablated − full_fusion) across all 89: **−0.0003**.

**Caveat (see `REVIEWER_AUDIT.md`): no multiple-comparison correction applied to these 89 tests.** At an uncorrected 5% false-positive rate, roughly 4-5 of 89 tests would be expected to show a spuriously significant result even if there were truly no effect anywhere — the observed 7/89 is not clearly distinguishable from that null expectation. This is disclosed as a required pre-submission revision in `MISSING_EXPERIMENTS.md` (T1.2).

## C. PCA pre-fix vs. post-fix, paired bootstrap summary

**Source**: `results/P1_dimension_control` (pre-fix) vs `results/P1_dimension_control_v3` (post-fix), 60 audited combinations.

| | Pre-fix | Post-fix |
|---|---|---|
| Combinations at exactly 0.500 BA (degenerate collapse) | 43/60 | not observed |
| Mean BA change vs. uncorrected baseline | not applicable (collapsed) | +0.030 |
| Significant positive (CI excludes zero, positive) | — | 30/60 |
| Significant negative (CI excludes zero, negative) | — | **0/60** |

## D. CORAL, full scope (post PCA-fix-scope applied)

**Source**: `results/P1_domain_adaptation`, 160 completed combinations (4 representations × 4 classifiers × 5 folds × 2 cross-device protocols).

| Representation | Uncorrected cross BA (Table 1) | CORAL-aligned cross BA | Δ |
|---|---|---|---|
| hubert | 0.5466 | 0.5293 | −0.0173 |
| wavlm_large | 0.5406 | 0.5255 | −0.0151 |
| full_fusion | 0.5311 | 0.5196 | −0.0115 |
| data2vec_fusion | 0.5347 | 0.5167 | −0.0180 |

All four representations show negative Δ — CORAL is not adopted as a mitigation.

## E. SpO2-corroboration filter ablation, significance summary (corrected values)

**Source**: `results/P2_statistics/corroboration_filter_vs_baseline.csv`, 32 combinations (full_fusion + hubert × {svm_rbf, mlp, random_forest, xgboost} × {R_R, S_S, R_S, S_R} × 5 folds, aggregated).

| Outcome | Count |
|---|---|
| Negative point-difference | 30/32 |
| Marginally positive point-difference | 2/32 (full_fusion/xgboost/R_S: +0.001247; hubert/svm_rbf/R_S: +0.011251) |
| Significant negative (CI excludes zero) | 12/32 |
| Significant positive (CI excludes zero) | 0/32 |

*(This is the corrected value — an earlier internal draft stated 32/32 and 13/32; see `CLAIM_EVIDENCE_AUDIT.md` for the correction record.)*
