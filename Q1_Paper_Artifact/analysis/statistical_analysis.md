# Statistical Analysis — Methodology and Full Write-Up

## Method

All classifier-performance comparisons use a **paired, subject-level bootstrap** (2,000 resamples): per-subject metric scores on an identical, aligned test set are differenced between two arms, and the difference is resampled with replacement *over subjects*, never over windows. This deliberately avoids pseudo-replication — treating multiple, highly-correlated same-subject clips as independent observations would inflate apparent significance. A 95% confidence interval excluding zero is the significance criterion used throughout the classifier-performance results (Sections 6.2–6.5).

The **one exception** is the domain-shift acoustic-statistics test (Section 6.1, Table 2): a window-level, two-sided Mann-Whitney U test on raw signal statistics, paired with Cliff's delta (rank-biserial correlation) as a non-parametric effect size. This is explicitly window-level because it characterizes the aggregate raw-signal distribution across devices, not a subject-level classifier outcome — its p-values should not be read with the same subject-independence guarantee as the classifier-performance tests. This asymmetry is disclosed in `manuscript.md` Section 5.3 and repeated here.

## What was tested, and what survived

1. **Matched-vs-cross generalization gap, full_fusion**: significant in both directions (R→R vs R→S: p<0.001; S→S vs S→R: p<0.001). Robust, single comparison, no multiplicity concern.
2. **full_fusion vs. WavLM-large, cross-device**: not significant in either direction (p=0.217, p=0.685). A single comparison at the project's validation-selected primary classifier — the finding that motivates Section 6.2's "no representation confirmed better" conclusion.
3. **PCA fix validation**: 60 tested combinations, 30/60 significant positive, 0/60 significant negative post-fix — a large, one-directional effect, low false-discovery risk even without formal correction given the complete absence of negative significant results.
4. **CORAL**: negative point-estimate in all 4 representations tested at full scope (160 combinations); reported as a consistent directional finding rather than leaning on a single significance count.
5. **Leave-one-out ablation**: 7/89 significant (4 positive, 3 negative) — see the explicit multiple-comparison caveat in `tables/statistical_analysis.md` Section B. At an uncorrected 5% rate, ~4-5/89 false positives are expected under a true null, making 7/89 not clearly distinguishable from chance. This is the single most important statistical caveat in the entire artifact and is repeated in three places (`REVIEWER_AUDIT.md`, `MISSING_EXPERIMENTS.md` T1.2, and here) deliberately, so it cannot be missed in a resumed session.
6. **SpO2-corroboration filter ablation**: 30/32 negative point estimates, 12/32 significant negative, 0/32 significant positive — a large, consistent, one-directional effect (32 tests, but a preponderance of large negative effects, not borderline null results near p=0.05).

## Effect-size reporting

Cliff's delta used for the domain-shift statistics (Table 2) — chosen because it is non-parametric, has an intuitive [-1,1] range, and does not assume normality (spectral/energy statistics are typically right-skewed). Balanced-accuracy point differences (in percentage points) used for all classifier-performance comparisons, alongside the bootstrap CI — no standardized effect size (e.g., Cohen's d) was computed for these, since balanced-accuracy differences are already on an interpretable, bounded scale.

## Explicitly NOT performed

- **DeLong's test** (for ROC-AUC comparison) — not run; the paired bootstrap on balanced accuracy was used as the primary significance instrument throughout instead, and ROC-AUC is reported descriptively (Table 3) without its own dedicated significance test. **NOT EVALUATED.**
- **McNemar's test** (for paired classification-error comparison) — not run; would be a reasonable complementary test for the ablation comparisons specifically, listed as a candidate addition if this artifact is extended.
- **Wilcoxon signed-rank test** — not run in place of the bootstrap; the bootstrap was judged sufficient and was the pre-existing project convention (established prior to this artifact, reused rather than replaced).
- **Permutation testing** — not run as a separate check; the bootstrap serves the same purpose here (empirical null via resampling) and re-implementing an additional permutation test was judged to add confirmatory value below the master prompt's "do not run new expensive experiments" threshold, given the bootstrap is a standard, valid alternative already in place.
