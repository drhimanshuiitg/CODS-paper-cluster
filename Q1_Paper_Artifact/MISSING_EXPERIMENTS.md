# Missing Experiments

Per the master prompt: do not run new expensive experiments to strengthen the paper artificially. This document lists what is genuinely missing, tiered by how strongly a Q1 reviewer would demand it before acceptance. None of these were run for this artifact.

## Tier 1 — Required before final journal submission

### T1.1 Significance test for the sliding-window severity result
- **RQ:** RQ5 / Section 6.6.
- **Exact experiment:** paired subject-level bootstrap, matched (R_R/S_S) vs. cross-device (R_S/S_R), on the existing 80/80 completed sliding-window-severity fold-runs.
- **Required data:** already exists — `results/P3_sliding_window_severity/`.
- **Expected output:** a CI on the matched-minus-cross gap (point estimate already known: 0.531 vs. 0.486).
- **Figure/table:** a new row in Table 5-style format, or an addition to Table 6.
- **Why a reviewer would ask:** the manuscript already flags this result as "no significance test has yet been run" — an alert reviewer will require it before accepting even the qualified claim. This is the cheapest Tier-1 item (data exists, only the test script is missing) and should be the first thing done in any follow-up session.

### T1.2 Multiple-comparison correction disclosure
- **RQ:** cross-cutting (RQ2, RQ4).
- **Exact experiment:** re-derive significance counts for the 89-combination ablation test (Section 6.4) and the representation-leaderboard comparisons (Section 6.2) under a Benjamini-Hochberg (or Bonferroni, more conservative) correction; report both raw and corrected counts.
- **Required data:** already exists — `results/P0_ablation_statistics/`, `results/P0_statistics_v2/`.
- **Expected output:** a corrected significant-count (likely lower than 7/89 for the ablation).
- **Figure/table:** a footnote or supplementary table in Section 6.4/6.2.
- **Why a reviewer would ask:** standard statistical-rigor expectation at this comparison scale; currently only disclosed in `REVIEWER_AUDIT.md`, not the manuscript body.

## Tier 2 — Strongly recommended

### T2.1 Per-domain / per-class confusion-matrix figure
- **RQ:** RQ5.
- **Exact experiment:** aggregate `tn/fp/fn/tp` (already in `MASTER_RESULTS.csv`) into a multi-panel confusion-matrix figure, stratified by protocol.
- **Required data:** already exists.
- **Expected output:** a 2×2 or 4-panel confusion-matrix grid.
- **Figure/table:** new figure (Figure 9 candidate).
- **Why a reviewer would ask:** Section 6.5's specificity-over-sensitivity claim is currently supported only by prose description of aggregate counts; a reviewer will want to see it.

### T2.2 Performance-vs-domain-distance exploratory analysis
- **RQ:** RQ1/RQ2 (exploratory).
- **Exact experiment:** compute a per-subject domain-distance metric (e.g., per-subject mean acoustic-statistic deviation from the opposite device's population) and scatter against per-subject cross-device accuracy.
- **Required data:** requires a new per-subject join between `tables/domain_shift_audio_stats.csv`-style statistics (currently only aggregate, not per-subject) and per-subject classifier outputs — **not currently computable** without a new, moderate-cost analysis script.
- **Expected output:** a scatter plot with a correlation coefficient (explicitly exploratory, not causal).
- **Figure/table:** new figure, explicitly labeled exploratory per the master prompt's menu item.
- **Why a reviewer would ask:** would strengthen the causal story beyond the current device-level (not subject-level) domain-shift evidence.

### T2.3 Independent re-verified environment/package snapshot
- **RQ:** reproducibility, not a scientific RQ.
- **Exact experiment:** capture a fresh `pip freeze`/`conda list` and hardware `nvidia-smi` snapshot at the exact commit used for the cited results, rather than relying on `REPRODUCIBILITY.md`'s narrative documentation.
- **Required data:** requires access to the exact execution environment at the relevant historical commit — may not be fully recoverable if the environment has since changed.
- **Expected output:** an `environment.lock` or equivalent artifact.
- **Figure/table:** N/A — supplementary reproducibility material.
- **Why a reviewer would ask:** standard reproducibility checklist item at most Q1 ML venues.

## Tier 3 — Optional / future work

### T3.1 Representation-level device-invariance training
- **RQ:** motivated by RQ2/RQ3's negative results as the logical next intervention.
- **Exact experiment:** contrastive fine-tuning (or lightweight adapter training) on this dataset's unique paired concurrent recordings, using same-event cross-device pairs as positives and same-subject-different-event pairs as hard negatives; evaluate under the identical subject-disjoint cross-device protocol already established.
- **Required data:** exists (paired recordings), but requires new model training — explicitly out of scope for this artifact per the master prompt's "do not launch long GPU jobs" instruction.
- **Expected output:** cross-device balanced accuracy compared against the current best (WavLM-large / full_fusion) baseline.
- **Figure/table:** new results table, would become the paper's headline positive result if successful, or a further honest negative result if not.
- **Why a reviewer would ask:** this is the most natural "next paper" a reviewer would want to see attempted, given the diagnosis this paper provides; not required for *this* paper's acceptance but would be the strongest possible follow-up.

### T3.2 External dataset / independent device-pair validation
- **RQ:** generalization of the pooled-training mitigation and the domain-shift findings beyond this specific dataset.
- **Exact experiment:** repeat the Section 6.1/6.3 core analyses (domain-shift statistics, cross-device benchmark, pooled-training comparison) on an independent dual-device or multi-device sleep-audio dataset.
- **Required data:** no such public dataset is currently known to the authors with the same concurrent-recording structure — this is itself a field-level gap, not just a missing experiment.
- **Expected output:** confirmation or refutation of the generalizability of Table 5's pooled-training result.
- **Figure/table:** N/A (future paper).
- **Why a reviewer would ask:** the strongest possible objection to any single-dataset finding; explicitly out of scope here and named as such in Limitations.

### T3.3 Domain-adversarial training / test-time adaptation
- **RQ:** RQ3, alternative mitigation strategy.
- **Exact experiment:** as named in Related Work (Moummad et al. framing) — not implemented here.
- **Required data:** exists; requires new model training.
- **Expected output:** comparison against the pooled-training and post-hoc-correction results in Table 5/Table 4.
- **Figure/table:** would extend Table 4/5.
- **Why a reviewer would ask:** a natural comparison point given CORAL's failure and PCA's fix; not attempted here to avoid unbounded scope creep in this artifact.
