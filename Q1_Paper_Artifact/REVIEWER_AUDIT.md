# Hostile Reviewer Audit

Self-administered per the master prompt's instruction to act as "a hostile Q1 journal reviewer." Scores are out of 10. This document is intentionally critical — it is not a marketing summary of the paper.

## Scores

| Dimension | Score /10 | Justification |
|---|---|---|
| Novelty | 5 | No new architecture or training method. Novel *contribution* is measurement (direct domain-shift quantification, embedding-space mechanism) and a methodological caution (representation-selection sensitivity), which is real but modest by the standard of what "novelty" usually signals in this venue class. |
| Technical quality | 7 | Subject-disjoint CV with runtime leakage assertions, paired-bootstrap significance throughout, an honestly-diagnosed and fixed implementation bug (PCA) — solid engineering. Docked for the still partly ambiguous "best representation" methodology dependence (Section 6.2) and the un-isolated fusion-mechanism ablation (no learned fusion exists to ablate). |
| Experimental rigor | 7 | Large combinatorial coverage (14 reps × 4 classifiers × 5 protocols), real significance testing, real negative results reported. Docked for: window-level (not subject-level) significance in the domain-shift statistics (explicitly caveated, but still a methodological asymmetry within the same paper); no dedicated held-out test-time-only final evaluation beyond the CV folds; sliding-window severity result has no significance test yet. |
| Dataset validity | 6 | Genuinely concurrent dual-device recording is the paper's structural strength — this is rare and directly enables the causal device-shift claim. Docked heavily for N=41-50, single cohort, single device pair, single annotation protocol — generalization beyond this exact setup is untested. |
| Statistical rigor | 7 | Paired bootstrap at the subject level for all classifier-performance claims, explicit avoidance of pseudo-replication, Mann-Whitney/Cliff's delta correctly used for the distributional claim. Docked for: no multiple-comparison correction stated across the 89-combination ablation test or the 14-representation leaderboard (7/89 and several "marginally significant" results should be read with this in mind — not currently disclosed as a limitation in the manuscript body, only here). |
| Reproducibility | 6 | Seeds, splits, preprocessing, GPU-only policy, and environment are documented (`REPRODUCIBILITY.md`). Docked because package version pins and exact hardware specs were not independently re-verified against a live environment dump in this pass — see `REPRODUCIBILITY.md`'s own "reproducible facts vs. missing info" split. |
| Clinical relevance | 4 | The paper explicitly disclaims clinical deployability (correct, and appropriate), which is honest but also means the clinical-relevance bar for this dimension is deliberately not attempted. The sliding-window severity result gestures toward clinical utility but is preliminary. |
| Visualization quality | 7 | 8 figures, real data, captions state measured-vs-conceptual status explicitly, N/A cells shown honestly rather than fabricated (Figure 4). Docked for: Figure 1/2 being schematic rather than data-driven (appropriate for their purpose, but reviewers sometimes penalize non-data figures in a Results-adjacent section), and for the missing multiple-comparison correction not being visually flagged anywhere. |
| Writing/framing honesty | 8 | Explicit avoidance of "device-invariant," "state-of-the-art," "clinically deployable"; negative results reported in full; a self-caught numerical error corrected before submission (documented in `CLAIM_EVIDENCE_AUDIT.md`). This is a genuine strength of this artifact relative to typical submissions. |
| **Overall** | **6/10** | A methodologically honest, well-instrumented negative/mechanistic-finding paper. Publishable at a Q1 venue that values rigor and negative results over architectural novelty; a **borderline** case at a venue expecting a positive proposed method as the headline contribution. See "risk of rejection" below. |

## Critical rejection risks

1. **"No proposed method" objection.** The single most likely rejection reason. This paper's contribution is measurement + a methodological finding + a training-recipe recommendation (pooling), not a new model. Some Q1 venues in this specific ML-for-health space explicitly want a positive technical contribution. This is a venue-fit risk, not a fixable flaw in the paper itself — see `SCIENTIFIC_STORY.md`.
2. **Sample size.** N=41-50 subjects is small by general ML standards, though not atypical for concurrent dual-device sleep-study data given how expensive it is to collect. A reviewer unfamiliar with the field's typical dataset sizes may not calibrate for this.
3. **Ambiguous "best representation" finding.** Section 6.2's methodology-dependent answer is honest but could be read by an unsympathetic reviewer as "the authors could not decide on their own headline result."

## Major concerns

- **Multiple-comparison correction absent.** 89 ablation comparisons and a 14-way representation leaderboard are each tested with an uncorrected 95%-CI significance criterion. At this scale, some of the "7/89 significant" and single-digit "marginally significant" findings elsewhere (e.g., ODI/HB fused 1/8) are plausible false positives under multiple testing. This should be disclosed as a limitation in the manuscript body (currently only surfaced here) — flagged as a required pre-submission revision.
- **Sliding-window severity result lacks a significance test.** Reported as "preliminary" in the manuscript, correctly, but a reviewer will ask for the test before accepting even the qualified claim.
- **No external dataset or independent device-pair validation.** All findings, including the pooled-training mitigation, are single-dataset. Whether pooling helps because these two devices happen to be reconcilable, or would help more generally, is untested.

## Minor concerns

- Figure 2 omits PCA/CORAL and the additional-signal branches "for legibility" — acceptable, but the caption should more prominently point to where the omitted detail lives (currently does, but briefly).
- The five candidate titles considered (top of `manuscript.md`) are useful process documentation but should not ship in the actual submission file — confirm this section is stripped before submission.
- `hubert_odi_hb` is included in the representation table (Table 1) and cross-device leaderboard (Table 3, Figure 7) despite ODI/HB alone being confirmed chance-level (Section 6.5) — not wrong to include, but the manuscript should make the apparent tension (why does fusing a chance-level feature not hurt, and does it help) explicit rather than leaving a reader to notice it. Currently addressed only in Section 6.5's own text; consider a forward/backward cross-reference from Table 3.

## Overclaimed statements found and status

A line-by-line scan of `manuscript.md` for the master prompt's banned-superlative list ("groundbreaking," "revolutionary," "highly effective," "remarkable," "unprecedented," "significant improvement" used loosely, "first," "novel," "state-of-the-art," "device invariant," "clinically deployable") found:
- **Zero unscoped uses** of "state-of-the-art," "device-invariant," "clinically deployable," "groundbreaking," "revolutionary," "remarkable," "unprecedented."
- **One scoped use of "first"** (Introduction, contribution #3) — explicitly qualified "to our knowledge... for this task," judged acceptable.
- **"Significant"** used only in its statistical sense throughout (paired with p-values or CIs), never as a loose intensifier — checked explicitly.
- **No unscoped uses of "novel"** — Section 4 explicitly disclaims a "novel architecture" framing.

**Conclusion: no overclaiming found requiring revision.** This check should be re-run after any future manuscript edit.

## Missing experiments (see `MISSING_EXPERIMENTS.md` for full tiered detail)

Top 3 a reviewer would most likely demand before acceptance: (1) significance test for the sliding-window severity result, (2) multiple-comparison-corrected re-statement of the ablation and leaderboard significance claims, (3) at least a discussion-level acknowledgment (already partially present) of what a representation-level device-invariance training intervention would look like, given the paper's own negative results motivate it as the logical next step.

## Missing figures/stats/lit-comparisons

- No per-subgroup (e.g., per apnea-severity-class, per-BMI) error breakdown figure.
- No literature comparison table placing this paper's cross-device numbers against any other published cross-device acoustic-OSA result (none identified as directly comparable — stated as a limitation, not fabricated).

## Reproducibility issues

See `REPRODUCIBILITY.md` for the full reproducible-vs-missing breakdown; the main gap is an independently-verified environment/package-version snapshot at time of the exact runs cited (currently sourced from the repository's environment documentation, not a fresh `pip freeze` captured at submission time).

## Leakage risks

Actively mitigated and monitored (runtime disjointness assertion on every fold-run, Section 5.1); the one documented historical leakage incident (an earlier, deprecated script) is disclosed in Limitations (Section 8) rather than hidden — this is a strength, not an open risk, but is listed here because the master prompt requires leakage risk to be explicitly addressed.

## What would flip this from borderline to strong-accept

1. A representation-level device-invariance training experiment (even a modest one — e.g., a contrastive objective on the paired concurrent recordings this dataset uniquely provides) with a positive or honestly-negative result, giving the paper a technical contribution alongside its measurement contribution.
2. Multiple-comparison-corrected significance reporting throughout, disclosed explicitly.
3. A second, independent device-pair or cohort validating the pooled-training mitigation generalizes beyond this specific dataset.
