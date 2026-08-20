# Table 11: Refreshed paired-bootstrap significance, full candidate set (balanced accuracy)

**Source**: results/P0_statistics_v2/bootstrap_differences.csv

**Caption**: Re-run of the project's primary significance methodology (results/P0_statistics), extended to include wavlm_large, hear, and full_fusion_plus_hear as candidates -- added in response to ARA Level 2 review findings F05/G1. best_single_encoder is now selected as wavlm_large (validation BA 0.6252 under svm_rbf, vs hubert 0.5940), not hubert.

**Extraction type**: raw_table

| comparison | left | right | point_difference | ci95_low | ci95_high | p_value_two_sided | interpretation |
|---|---|---|---|---|---|---|---|
| full_fusion_vs_data2vec_fusion_R_S | full_fusion/R_S | data2vec_fusion/R_S | -0.0130 | -0.0295 | 0.0039 | 0.1470 | CI includes zero |
| full_fusion_vs_wavlm_large_R_S | full_fusion/R_S | wavlm_large/R_S | -0.0141 | -0.0344 | 0.0084 | 0.2170 | CI includes zero |
| wavlm_large_vs_classical_R_S | wavlm_large/R_S | classical/R_S | 0.0331 | 0.0214 | 0.0457 | 0.0000 | CI excludes zero |
| full_fusion_plus_hear_vs_full_fusion_R_S | full_fusion_plus_hear/R_S | full_fusion/R_S | -0.0099 | -0.0297 | 0.0064 | 0.2560 | CI includes zero |
| full_fusion_vs_data2vec_fusion_S_R | full_fusion/S_R | data2vec_fusion/S_R | 0.0107 | -0.0042 | 0.0270 | 0.1800 | CI includes zero |
| full_fusion_vs_wavlm_large_S_R | full_fusion/S_R | wavlm_large/S_R | 0.0036 | -0.0135 | 0.0216 | 0.6850 | CI includes zero |
| wavlm_large_vs_classical_S_R | wavlm_large/S_R | classical/S_R | 0.0400 | 0.0216 | 0.0596 | 0.0000 | CI excludes zero |
| full_fusion_plus_hear_vs_full_fusion_S_R | full_fusion_plus_hear/S_R | full_fusion/S_R | -0.0009 | -0.0057 | 0.0036 | 0.7300 | CI includes zero |
| full_fusion_R_R_vs_R_S | full_fusion/R_R | full_fusion/R_S | 0.0588 | 0.0326 | 0.0844 | 0.0000 | CI excludes zero |
| full_fusion_S_S_vs_S_R | full_fusion/S_S | full_fusion/S_R | 0.0596 | 0.0277 | 0.0913 | 0.0000 | CI excludes zero |
