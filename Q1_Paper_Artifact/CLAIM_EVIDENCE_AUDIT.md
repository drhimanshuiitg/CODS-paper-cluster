# Claim–Evidence Audit

Every claim appearing in `manuscript.md`, mapped to its supporting experiment, evidence, figure/table, confidence, and the safe wording actually used. This is the private working table referenced by the master prompt (Section 6/33); it is more granular than the manuscript's own prose.

| # | Manuscript claim | Supporting experiment | Evidence | Figure/Table | Confidence | Safe wording used |
|---|---|---|---|---|---|---|
| 1 | Devices differ with near-maximal effect size on acoustic statistics | `analyze_domain_shift.py`, n=600/device | Mann-Whitney U p<1e-30, \|Cliff's δ\|=0.54–1.00 (6 metrics) | Table 2, Fig. 3 | **Strongly supported** | "near-maximal effect size... 5/6 above 0.95" (exact values quoted) |
| 2 | Every representation shows a real matched→cross accuracy drop | `results/P0_device_gap`, all 14 reps | Point estimates, Fig. 6 | Fig. 6 | **Strongly supported** (point estimates); significance confirmed only for full_fusion | "mostly significant... confirmed significant for full_fusion specifically" — significance qualifier kept explicit |
| 3 | Device is more separable than clinical label in HuBERT embedding space | `analyze_embedding_space.py`, n=2,400 | Silhouette 0.145 (device) vs 0.007 (class) | Fig. 5 | **Strongly supported** (as descriptive statistic); **exploratory** as a causal mechanism claim | "offering a mechanistic... explanation" (hedged as explanation, not proof) |
| 4 | HuBERT ranks highest cross-device by raw point estimate | `results/P0_device_gap` aggregate | 0.5466 mean BA | Table 3 | **Strongly supported** (it is a computed mean, unambiguous) | stated plainly as a point-estimate fact |
| 5 | WavLM-large is the validation-selected best single encoder | `results/P0_statistics_v2` (F05 fix) | Validation BA 0.625 vs HuBERT 0.594 | manuscript §6.2 | **Strongly supported** | stated plainly, methodology named explicitly |
| 6 | full_fusion not significantly different from WavLM-large cross-device | Paired bootstrap, `results/P0_statistics_v2` | p=0.217 (R→S), p=0.685 (S→R) | manuscript §6.2 | **Strongly supported** | "statistically indistinguishable" |
| 7 | PCA's degenerate collapse was a fixable scope bug, not a fundamental limitation | Pre/post-fix PCA runs, paired bootstrap (60 combos) | 43/60 collapsed pre-fix; 0/60 significant negative post-fix, 30/60 significant positive | manuscript §6.3 | **Strongly supported** | stated with exact fractions, not "PCA works" |
| 8 | CORAL fails regardless of scope | `results/P1_domain_adaptation`, 160 combos | Negative Δ for all 4 tested representations, specificity-dominant collapse | Table 4 | **Strongly supported** | stated plainly as a negative result, no hedge needed |
| 9 | Pooled-device training closes most of the cross-device gap | RS_RS protocol, 100 combos, 5 representations | Table 5, gap-closed 101–123% | Table 5, Fig. 4 | **Strongly supported** | "closed the great majority... for every representation tested" |
| 10 | No individual encoder is confirmed necessary in the fusion | Leave-one-out ablation, 89 combos | 7/89 individually significant | Fig. 8 | **Strongly supported** as a negative/null finding | "we do not have evidence that any specific encoder... is individually necessary" |
| 11 | SpO2-corroboration filtering is a negative result | `results/P2_statistics/corroboration_filter_vs_baseline.csv`, 32 combos | 30/32 negative point estimates (2/32 marginal positive), 12/32 significant negative, 0/32 significant positive | Table 6 | **Strongly supported** | exact re-verified fractions used (corrected from an earlier internal error — see below) |
| 12 | HeAR is the weakest representation and does not help fused | `results/P0_device_gap` + `results/P0_efficiency` | 0.504 BA alone; fusion-level p=0.256/0.730 | Table 6 | **Strongly supported** | reported plainly as negative |
| 13 | ODI/HB is chance-level alone, indistinguishable fused | `results/P0_device_gap`, paired bootstrap (8 combos) | 0.496 BA alone; 1/8 marginally significant fused | Table 6 | **Strongly supported**; predicted in advance, not just observed | "confirmed, not merely observed, prediction" |
| 14 | Sliding-window severity: positive matched signal, below-chance cross-device | `results/P3_sliding_window_severity`, 80/80 combos | Matched BA 0.531 (SD 0.049); cross BA 0.486 (SD 0.074) | manuscript §6.6 | **Exploratory** — explicitly flagged, no significance test run yet | "This result is directionally consistent with... though it should not be over-read given its preliminary... status" |
| 15 | Domain-adaptation failure favors specificity over sensitivity | Confusion counts across CORAL/PCA-collapse runs | tn/fp/fn/tp fields in `MASTER_RESULTS.csv` | manuscript §6.5 | **Moderately supported** — descriptive pattern across existing runs, no dedicated stratified error-analysis figure built | stated as an aggregate pattern, explicitly notes deeper analysis is `MISSING_EXPERIMENTS.md` Tier 2 |

## Note on Claim #11 (data-integrity correction)

During preparation of this artifact, an internal draft (propagated into the accompanying ARA's `claims.md`/`problem.md`/`exploration_tree.yaml`) stated "32/32 negative, 13/32 significant negative" for the corroboration-filter result. Re-verification directly against `results/P2_statistics/corroboration_filter_vs_baseline.csv` before finalizing this manuscript found the correct values are **30/32 negative (2/32 marginally positive), 12/32 significant negative**. All ARA files and this manuscript now use the corrected, source-verified values. This correction is recorded here per the master prompt's data-integrity requirement, as an example of the verification discipline applied throughout.

## Claims considered and explicitly NOT made

- "Device-invariant" — not claimed anywhere; no invariance mechanism exists in the evaluated system.
- "State-of-the-art" — not claimed; no comparison to other published cross-device acoustic-OSA results exists (none are known to be directly comparable — different dataset, different device pairs).
- "Clinically deployable" — not claimed; Limitations explicitly disclaim this.
- "First to show X" — used once, narrowly, in Introduction contribution #3 ("to our knowledge the first visualization, for this task, ...") — scoped to "for this task" specifically, not a general priority claim.
