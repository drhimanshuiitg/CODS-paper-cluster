# Scientific Story (internal — for authors, not for the manuscript body)

**One-sentence problem.** Acoustic OSA screening research reports matched-device accuracy almost exclusively, leaving open whether learned representations actually transfer across the different recording hardware that real-world deployment requires.

**One-sentence gap.** No prior work on this task has directly measured device-level acoustic domain shift at the signal level or asked, in the representation's own embedding space, what it is actually separating.

**One-sentence solution.** Using a rare concurrently-recorded dual-device dataset, we measure the shift directly (signal + embedding level), benchmark 14 representations under rigorous subject-disjoint, significance-tested cross-device evaluation, and test both post-hoc correction and pooled-device training as mitigations.

**One-sentence key result.** Device identity is nearly 20x more separable than the clinical label in a strong frozen encoder's own embedding space, no representation is confirmed significantly more cross-device-robust than any other, and simple pooled-device training — not architecture — closes most of the measured gap.

**One-sentence contribution.** A methodologically rigorous, honestly negative-result-inclusive cross-device benchmark that redirects the field's attention from representation/fusion sophistication toward training-data composition and representation-level device-invariance as the actual open problems.

## Why should a Q1 reviewer care?

Because it corrects a measurement problem the field has largely not acknowledged: papers in this space report point-estimate leaderboards on matched-device data (or occasionally cross-device data) without subject-level significance testing, and without asking whether "best representation" claims survive a more rigorous selection protocol. This paper shows, concretely, that they may not (Section 6.2) — a general methodological caution, not just a result specific to this dataset. It also supplies a mechanistic (not just phenomenological) account of *why* cross-device transfer fails, via the embedding silhouette analysis, which is new evidence, not a re-statement of the accuracy gap already visible in prior cross-device papers.

## Strongest evidence in the paper

1. The signal-level domain-shift statistics (Table 2, Figure 3): six independent acoustic measures, near-maximal effect sizes, p<1e-30 on all six. This is unambiguous and not open to a "small sample, noisy" objection (n=600/device, six converging metrics).
2. The pooled-device training result (Table 5): a clean, low-dimensional, easily-replicated finding (5 representations, all showing the same pattern) with an obvious, immediately actionable practical implication.
3. The corrected PCA-vs-CORAL comparison (Section 6.3): a real bug found, fixed, and validated (0/60 significant negative post-fix) — this is the kind of transparent methodological correction that builds reviewer trust rather than eroding it.

## Weakest part of the paper

The "best representation" claim in Section 6.2 is genuinely ambiguous — two legitimate methodologies disagree (HuBERT by raw point estimate, WavLM-large by validation-selected protocol), and full_fusion is not confirmed significantly different from either. A hostile reviewer could argue the paper does not deliver a clean recommendation for practitioners on which single representation to deploy. We do not think this can or should be resolved by picking one convention and hiding the other — the honest answer is that this specific "best representation" question does not have a statistically confident answer given the tested sample sizes, and the paper should say so plainly (it does, in Section 6.2).

## Most dangerous reviewer criticism, and how it is addressed

**"This is a negative-results paper with no proposed method — why does it belong in this venue?"**
Addressed directly in Section 4's framing ("this paper's primary contribution is not a novel model architecture") and in the Introduction's contribution list, which names the mechanistic embedding-space finding, the pooled-training mitigation, and the methodological caution about representation-selection sensitivity as the actual contributions, rather than presenting a fusion architecture as a headline method it is not. The paper's counter-argument, made explicitly in Discussion (Section 7): a well-instrumented negative/mechanistic result that redirects a field's effort (away from encoder sophistication, toward training-data composition and device-invariance training) is itself a legitimate and useful contribution, particularly given the demonstrated fragility of "best representation" claims in this literature (Section 6.2). This framing is a risk — some venues will still want a positive proposed method — and is flagged as a submission-venue-fit consideration outside this document's scope.

**Secondary risk: "the dataset is small (n=41-50) and single-cohort."**
Addressed in Limitations (Section 8) without minimization; not defensible to argue away, only honestly scoped.
