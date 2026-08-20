# Table 10: HuBERT vs. HuBERT+ODI/HB, paired bootstrap significance (balanced accuracy)

**Source**: results/P0_statistics_hubert_vs_hubert_odi_hb/hubert_odi_hb_vs_hubert.csv

**Caption**: Direct paired subject-level bootstrap (2000 iterations) between hubert and hubert_odi_hb, cross-device protocols (R_S, S_R), all 4 classifiers -- added in response to ARA Level 2 review finding F03 (C09's 'statistically indistinguishable' language previously had no direct significance test cited for this specific pair).

**Extraction type**: raw_table

| classifier | protocol | hubert_odi_hb_point | hubert_point | point_difference | ci95_low | ci95_high | p_value_two_sided | interpretation |
|---|---|---|---|---|---|---|---|---|
| svm_rbf | R_S | 0.5243 | 0.5298 | -0.0055 | -0.0259 | 0.0124 | 0.5770 | CI includes zero |
| svm_rbf | S_R | 0.5533 | 0.5503 | 0.0030 | -0.0070 | 0.0164 | 0.6750 | CI includes zero |
| mlp | R_S | 0.5350 | 0.5350 | 0.0000 | -0.0153 | 0.0149 | 0.9900 | CI includes zero |
| mlp | S_R | 0.5354 | 0.5339 | 0.0015 | -0.0217 | 0.0255 | 0.8820 | CI includes zero |
| random_forest | R_S | 0.5467 | 0.5444 | 0.0023 | -0.0120 | 0.0166 | 0.7410 | CI includes zero |
| random_forest | S_R | 0.5329 | 0.5244 | 0.0086 | 0.0030 | 0.0147 | 0.0030 | CI excludes zero |
| xgboost | R_S | 0.5369 | 0.5465 | -0.0096 | -0.0224 | 0.0025 | 0.1200 | CI includes zero |
| xgboost | S_R | 0.5370 | 0.5418 | -0.0048 | -0.0137 | 0.0043 | 0.3110 | CI includes zero |
