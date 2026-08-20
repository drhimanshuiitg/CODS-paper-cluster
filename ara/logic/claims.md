# Claims

## C01: No representation is significantly better than any other among the top single encoders and full_fusion (revised 2026-08-19, closing G1)
- **Statement**: Two different, both-legitimate measurement conventions disagree on which single encoder ranks highest, and neither confirms a statistically significant winner. (a) Raw point estimates averaged across all 4 classifiers (`evidence/tables/table1_main_benchmark_summary.md`) rank HuBERT highest (0.5466 mean cross-device BA) among non-trivial representations. (b) The project's own more rigorous, pre-established methodology -- select the primary classifier by validation performance (svm_rbf), then select the best single encoder by validation performance under that classifier -- ranks `wavlm_large` highest (validation BA 0.6252 vs. HuBERT's 0.5940), not HuBERT. Under method (b), a paired bootstrap shows full_fusion vs. wavlm_large is **not** statistically significant in either cross-device direction (R_S: p=0.217, CI includes zero; S_R: p=0.685, CI includes zero).
- **Status**: supported (revised and narrowed from an earlier, less careful version of this claim -- see Interpretation)
- **Falsification criteria**: a re-run of `scripts/run_statistics.py` (validation-selected classifier and encoder, full current candidate pool) showing a statistically significant difference between full_fusion and the validation-selected best single encoder.
- **Proof**: [E01]
- **Evidence basis**: `evidence/tables/table1_main_benchmark_summary.md` (method a); `evidence/tables/table11_refreshed_significance_full_candidate_set.md`, `evidence/tables/table12_refreshed_confidence_intervals.md` (method b, `results/P0_statistics_v2`, re-run in response to Level 2 review finding F05).
- **Interpretation**: This claim previously asserted "HuBERT is the best representation" using only method (a) -- a Level 2 review (F05/G1) flagged that this had never been checked against the project's own more rigorous method (b) once wavlm_large was added to the candidate pool. Re-running it changed the answer: method (b) selects a *different* encoder (wavlm_large) as best, and even then finds no significant gap between it and full_fusion. The two methods' disagreement is itself informative -- it means the specific identity of "the best representation" is sensitive to classifier-selection methodology, while the higher-level finding that no representation decisively beats the others (C02's core point) holds under both methods.
- **Dependencies**: none
- **Tags**: main-benchmark, cross-device, encoder-comparison, revised-post-review

## C02: Naive multi-encoder concatenation does not significantly improve cross-device generalization over the best single encoder (narrowed 2026-08-19)
- **Statement**: By raw point estimate (mean over all 4 classifiers), every multi-encoder fusion tested (full_fusion, full_fusion_v2, full_fusion_plus_hear, data2vec_fusion) has lower mean cross-device balanced accuracy than plain HuBERT alone. Under the project's more rigorous validation-selected methodology (primary classifier svm_rbf, best single encoder wavlm_large -- see C01), full_fusion is statistically indistinguishable from the best single encoder in both cross-device directions (neither significantly better nor significantly worse). full_fusion_plus_hear vs. full_fusion is likewise not significant in either direction (R_S: p=0.256; S_R: p=0.730).
- **Status**: supported, narrowed — "fusion does not help" is confirmed; "fusion actively underperforms" is a point-estimate observation, not a significance-tested finding
- **Falsification criteria**: a fusion representation with mean cross-device balanced accuracy >= the best single encoder's (point-estimate form), or a paired bootstrap showing a fusion representation *significantly* exceeding the validation-selected best single encoder (significance-tested form).
- **Proof**: [E01]
- **Evidence basis**: `evidence/tables/table1_main_benchmark_summary.md` (point-estimate form); `evidence/tables/table11_refreshed_significance_full_candidate_set.md` (significance-tested form, added post-review per F05).
- **Interpretation**: Thematically consistent with C10 (ablation redundancy finding) — the fusion's extra encoders appear to add noise/redundancy rather than complementary signal, at least under naive concatenation (no learned fusion weighting, no contrastive objective). Note this is a consistency relationship, not a logical prerequisite: C02 is independently established by E01 alone, without needing C10 to hold. The practically important reading is the significance-tested one: there is no confirmed cost to using full_fusion instead of a single encoder, but also no confirmed benefit -- the honest finding is "statistically a wash, so prefer the cheaper single encoder on efficiency grounds (E10)," not "fusion is proven worse."
- **Dependencies**: none (see Interpretation — thematically related to C10, but not a logical prerequisite; the reviewer-flagged circular C02<->C10 Dependencies pair was removed 2026-08-19)
- **Tags**: main-benchmark, fusion, negative-result, revised-post-review

## C03: The matched-to-cross-device performance gap is real and present across every representation
- **Statement**: Mean matched-device balanced accuracy exceeds mean cross-device balanced accuracy for every representation tested except `odi_hb` (gap = 0.0, itself a chance-level, device-agnostic-by-construction feature).
- **Status**: supported
- **Falsification criteria**: any real (non-odi_hb) representation with cross-device balanced accuracy >= matched-device balanced accuracy.
- **Proof**: [E01]
- **Evidence basis**: `evidence/tables/table1_main_benchmark_summary.md`, gap column.
- **Interpretation**: This project's significance testing (`results/P0_statistics`, and its post-review refresh `results/P0_statistics_v2` — see C01) found this gap statistically significant at p<0.001 for `full_fusion` specifically (R_R vs. R_S: p<0.001, +5.9pt; S_S vs. S_R: p<0.001, +6.0pt), unchanged by the refresh since the device-gap comparison specs were not themselves extended to the newer representations (only the encoder-vs-encoder comparisons were, per F05). The gap's significance for wavlm_large, hear, and full_fusion_plus_hear specifically has still not been directly tested — only the point-estimate pattern (O3) is confirmed for those three.
- **Dependencies**: none
- **Tags**: main-benchmark, device-gap, core-finding

## C04: PCA dimensionality reduction, fit only on source-device data, collapses to a degenerate single-class predictor in cross-device transfer
- **Statement**: Prior to the 2026-08-19 fix, PCA-reduced classifiers (`results/P1_dimension_control`) produced exactly balanced accuracy 0.500 in 43 of 60 audited representation/classifier/fold/dimension combinations, with maximum predicted test probability never exceeding 0.50 in one transfer direction.
- **Status**: supported
- **Falsification criteria**: re-inspection of `results/P1_dimension_control`'s completion.json files showing balanced accuracy materially different from 0.500 in those combinations. Note (reviewer-flagged, 2026-08-19): this is a historical/archival falsification criterion, not a predictive one — it can only meaningfully be checked once, against static past results that will not change, unlike C01/C05/C10/C11's forward-looking criteria which describe what a new or repeated experiment would need to show.
- **Proof**: [E03]
- **Evidence basis**: `results/P1_statistics_pca_fix/pca_fix_vs_baseline.csv` `collapsed_baseline` field (43/60); direct `completion.json` inspection.
- **Interpretation**: none beyond the direct observation.
- **Dependencies**: none
- **Tags**: pca, domain-adaptation, bug, negative-result-superseded, historical-claim

## C05: The PCA collapse is caused by a source-device-only refit scope and is resolved by scoping it like CORAL
- **Statement**: Extending the PCA refit's fit data from source-device train+val subjects only to also include unlabeled target-device validation subjects (the same scope CORAL's covariance alignment already used) resolves the collapse and yields a statistically significant cross-device balanced-accuracy improvement of +0.0299 mean (30/60 combinations individually significant positive, 0/60 significant negative).
- **Status**: supported
- **Falsification criteria**: a paired bootstrap re-run showing a non-significant or negative mean effect, or any individually-significant-negative combination.
- **Proof**: [E04]
- **Evidence basis**: `evidence/tables/table2_pca_fix_significance.md` (full 60-row paired bootstrap output).
- **Interpretation**: This reframes PCA's role in the broader domain-adaptation narrative — not "PCA fails on this task" but "PCA fails only when scoped inconsistently with an already-working technique in the same codebase."
- **Dependencies**: C04
- **Tags**: pca, domain-adaptation, bug-fix, positive-result

## C06: CORAL feature-space alignment consistently reduces cross-device accuracy relative to uncorrected features
- **Statement**: Across all 6 representation/direction combinations tested (`{data2vec_fusion, full_fusion, hubert}` x `{R_S, S_R}`, svm_rbf only), CORAL-aligned balanced accuracy is lower than the uncorrected baseline in every case (e.g. HuBERT S->R: 53.6% CORAL-aligned vs. 56.5% uncorrected, per `paper/conference_101719.tex`'s already-verified figure), with the aligned classifier showing specificity dominance (0.78-0.99) over sensitivity (0.01-0.29) in every configuration.
- **Status**: supported
- **Falsification criteria**: any of the 6 tested combinations showing CORAL-aligned balanced accuracy >= uncorrected baseline.
- **Proof**: [E05]
- **Evidence basis**: `results/P1_domain_adaptation` (30 completed combinations); cross-checked against `paper/conference_101719.tex`'s CORAL paragraph, which was independently written from the same result files earlier this session.
- **Interpretation**: Unlike PCA (C04/C05), no scope-artifact explanation was found for CORAL's failure — it is treated as the more fundamental negative result of the two domain-adaptation techniques tested, though this has not been independently stress-tested (e.g. no attempt yet to check whether CORAL's regularization parameter or covariance-estimation sample size is itself a fixable scope issue analogous to the PCA case).
- **Dependencies**: none
- **Tags**: coral, domain-adaptation, negative-result

## C07: Filtering training-positive windows lacking SpO2-desaturation corroboration does not improve, and measurably hurts, cross-device performance
- **Statement**: Paired subject-level bootstrap over 32 representation/classifier/protocol combinations comparing corroboration-filtered training vs. unfiltered baseline (identical, unfiltered test set both arms) shows balanced-accuracy point difference negative in 30/32 cases (2/32 marginally positive), individually significant negative in 12/32, and significant positive in 0/32.
- **Status**: supported (as a negative result — refutes the original hypothesis that label-quality filtering would help)
- **Falsification criteria**: a re-run showing any individually-significant-positive combination, or a non-negative mean point difference.
- **Proof**: [E06]
- **Evidence basis**: `results/P2_statistics/corroboration_filter_vs_baseline.csv`.
- **Interpretation**: The most likely explanation on record is that AASM-scored hypopnea events lacking a corroborating SpO2 desaturation (chest-effort/flow-scored, not desaturation-scored) are still real positives rather than annotation noise, so removing them only shrinks and re-imbalances the training set without a compensating label-quality gain. This interpretation is plausible but not independently tested (e.g. no manual clinical re-adjudication of the "uncorroborated" events was performed).
- **Dependencies**: none
- **Tags**: label-quality, spo2, negative-result

## C08: HeAR (health-acoustic foundation model) underperforms general-purpose speech SSL encoders on this task
- **Statement**: HeAR alone achieves cross-device balanced accuracy 0.5040, the lowest of any representation tested that carries real per-window audio signal (`odi_hb` is lower still at 0.4958, but is a deliberately degenerate, per-subject-constant baseline with no per-window signal by construction, per C09 -- not a genuinely competing representation); by raw point estimate, fusing HeAR into full_fusion reduces cross-device balanced accuracy from 0.5311 (full_fusion alone) to 0.5288 (full_fusion_plus_hear). A direct paired bootstrap of full_fusion_plus_hear vs. full_fusion under the primary classifier (svm_rbf) does **not** reach significance in either cross-device direction (R_S: p=0.256; S_R: p=0.730) -- the fusion-level "HeAR hurts" reading is a point-estimate observation, not a significance-tested one. HeAR alone being the weakest single representation is the more solidly evidenced part of this claim.
- **Status**: supported (as a negative result) for "HeAR alone is the weakest real representation"; point-estimate-only, not significance-tested, for "adding HeAR to full_fusion hurts"
- **Falsification criteria**: a re-run showing HeAR or a HeAR-inclusive fusion with cross-device balanced accuracy exceeding the corresponding non-HeAR comparator; for the fusion-level sub-claim specifically, a paired bootstrap showing a significant effect in either direction would change "not significant" to a confirmed result.
- **Proof**: [E07]
- **Evidence basis**: `evidence/tables/table1_main_benchmark_summary.md`; `evidence/tables/table11_refreshed_significance_full_candidate_set.md` (`full_fusion_plus_hear_vs_full_fusion_{R_S,S_R}` rows, added post-review per F05).
- **Interpretation**: HeAR's pretraining corpus (broad health acoustics: coughs, breathing, ~174k hours) may not specifically discriminate the snoring/breath-sound patterns this task's binary label distinguishes, unlike general-purpose speech SSL encoders whose pretraining (though not health-specific) may better capture fine-grained acoustic-event boundaries relevant here. This is a plausible post-hoc explanation, not independently verified (e.g. no probing/representation-similarity analysis was run to test it directly). The fusion-level result being non-significant is itself informative: it means HeAR's inclusion is closer to "harmless dead weight" (adds cost, no confirmed accuracy change) than "actively damaging" once tested rigorously, which is a more precise and slightly less negative finding than the pre-review point-estimate-only framing suggested.
- **Dependencies**: none
- **Tags**: hear, foundation-model, negative-result

## C09: ODI/Hypoxic-Burden, as a per-subject-constant feature, adds no per-window discriminative signal
- **Statement**: `odi_hb` alone achieves balanced accuracy 0.4958 in both matched- and cross-device regimes (chance-level, and identical between regimes since the feature does not vary by device or window); `hubert_odi_hb` (HuBERT + ODI/HB concatenated) is statistically indistinguishable from HuBERT alone in 7 of 8 tested classifier/protocol combinations (paired subject-level bootstrap, cross-device), with one marginal exception (random_forest/S_R, +0.0086 BA, CI [0.0030, 0.0147]).
- **Status**: supported
- **Falsification criteria**: `odi_hb` alone materially exceeding chance, or `hubert_odi_hb` significantly exceeding HuBERT alone in a majority of tested combinations in a paired test.
- **Proof**: [E08]
- **Evidence basis**: `evidence/tables/table1_main_benchmark_summary.md`; `results/P0_statistics_hubert_vs_hubert_odi_hb/hubert_odi_hb_vs_hubert.csv` (reviewer-requested direct paired-bootstrap test, added 2026-08-19, in response to Level 2 review finding F03 — previously this specific pair's "statistically indistinguishable" language had no direct significance test cited, only point-estimate proximity).
- **Interpretation**: This was theoretically predicted before being empirically tested this session (a per-subject-constant value cannot add per-window discriminative signal to a per-window classification task) and is treated as a confirmed, not merely observed, prediction — now with a direct significance test backing the "indistinguishable" language for 7/8 combinations, and an honestly-reported single small exception rather than a rounded-away one. ODI/Hypoxic-Burden's validated correlation with PSG-annotated event counts (Pearson r=0.83 ODI, r=0.61 hypoxic burden — `scripts/compute_odi_hypoxic_burden.py` docstring) establishes the feature is a real severity signal at the subject level; the negative result here is specifically about its use as a raw per-window input feature, not a statement about its clinical validity.
- **Dependencies**: none
- **Tags**: odi-hb, feature-engineering, negative-result

## C10: Individual encoders within the 5-encoder fusion carry largely redundant, not complementary, information
- **Statement**: A leave-one-encoder-out ablation (6 variants x 4 classifiers x 4 protocols x 5 folds) compared against full_fusion via paired bootstrap over 89 evaluable combinations shows mean balanced-accuracy point difference -0.0003, with only 7/89 combinations individually significant (4 positive toward the ablated variant, 3 negative).
- **Status**: supported
- **Falsification criteria**: a majority of combinations showing a consistent, significant directional effect (either dropping any encoder consistently helps, or consistently hurts).
- **Proof**: [E02]
- **Evidence basis**: `results/P0_ablation_statistics/ablation_vs_full_fusion.csv`.
- **Interpretation**: This statistically-tested finding corrects an earlier, less rigorous point-estimate-only read of the same ablation (which had suggested "dropping almost any encoder helps") — the significance test shows that read was mostly noise, not a real effect. It is presented here as the corrected version, and is thematically consistent with C02 (see C02's Interpretation), though neither is a logical prerequisite for the other.
- **Dependencies**: none (see Interpretation)
- **Tags**: ablation, fusion, encoder-redundancy

## C11: Whole-night sliding-window severity is not confirmed learnable cross-device with a single-encoder, no-device-invariance-mechanism classifier (resolved 2026-08-19 — run completed)
- **Statement**: A HuBERT-feature-based binary classifier distinguishing SpO2-derived "severe" whole-night 5-minute epochs from non-severe epochs achieves matched-device balanced accuracy 0.5307 (std 0.0488, n=40) — a small but real positive signal — and cross-device balanced accuracy 0.4859 (std 0.0742, n=40), **below chance**, subject-disjoint, all 4 protocols, full 80/80 combinations now complete.
- **Status**: supported (as a negative/null result for cross-device transfer specifically; the matched-device signal, while weak, is directionally consistent with the target being real)
- **Falsification criteria**: a paired significance test showing cross-device balanced accuracy significantly above 0.5, or significantly above the matched-device figure (neither of which the point estimates suggest, but no formal significance test has been run for this specific comparison yet — see Interpretation).
- **Proof**: [E09]
- **Evidence basis**: `evidence/tables/table3_sliding_window_severity_full.md` — full 80/80 combinations, all 4 protocols.
- **Interpretation**: This closes G4's open question with a result consistent with C01-C03's device-gap pattern extending to this task: cross-device transfer fails (mean BA below chance) using a single encoder with no explicit device-invariance mechanism, while matched-device shows a real (if modest) positive signal, suggesting the target itself carries genuine learnable structure — the failure is plausibly a transfer problem, not a target-validity problem. No formal paired-bootstrap significance test has yet been run for matched-vs-cross or cross-vs-chance specifically for this experiment (unlike C01-C10, which all have one) -- the point estimates are suggestive but not yet significance-confirmed, a real remaining gap for this specific claim.
- **Dependencies**: C09 (uses the same underlying ODI/HB-derived severity ground truth, but as a per-epoch label rather than a per-subject feature)
- **Tags**: sliding-window, severity, negative-result, device-gap-consistent
