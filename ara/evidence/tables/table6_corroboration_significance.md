# Table 6: SpO2-corroboration-filtered training vs. unfiltered baseline, paired bootstrap (balanced accuracy)

**Source**: results/P2_statistics/corroboration_filter_vs_baseline.csv

**Caption**: Paired subject-level bootstrap (2000 iterations), 32 representation/classifier/protocol combinations, full_fusion+hubert (primary) plus wavlm+data2vec_fusion (extended coverage).

**Extraction type**: raw_table

| representation | classifier | protocol | filtered_point | baseline_point | point_difference | ci95_low | ci95_high | p_value_two_sided | interpretation |
|---|---|---|---|---|---|---|---|---|---|
| full_fusion | mlp | R_R | 0.5575 | 0.5676 | -0.0102 | -0.0246 | 0.0047 | 0.1740 | CI includes zero |
| full_fusion | mlp | R_S | 0.5356 | 0.5365 | -0.0009 | -0.0188 | 0.0186 | 0.8820 | CI includes zero |
| full_fusion | mlp | S_R | 0.5178 | 0.5259 | -0.0081 | -0.0235 | 0.0069 | 0.2930 | CI includes zero |
| full_fusion | mlp | S_S | 0.5698 | 0.5748 | -0.0050 | -0.0235 | 0.0126 | 0.5730 | CI includes zero |
| full_fusion | random_forest | R_R | 0.5599 | 0.5679 | -0.0081 | -0.0190 | 0.0029 | 0.1480 | CI includes zero |
| full_fusion | random_forest | R_S | 0.5132 | 0.5196 | -0.0064 | -0.0183 | 0.0084 | 0.3550 | CI includes zero |
| full_fusion | random_forest | S_R | 0.5098 | 0.5218 | -0.0120 | -0.0254 | 0.0005 | 0.0620 | CI includes zero |
| full_fusion | random_forest | S_S | 0.5689 | 0.5916 | -0.0227 | -0.0330 | -0.0134 | 0.0000 | CI excludes zero |
| full_fusion | svm_rbf | R_R | 0.5720 | 0.5783 | -0.0063 | -0.0173 | 0.0037 | 0.2430 | CI includes zero |
| full_fusion | svm_rbf | R_S | 0.5137 | 0.5195 | -0.0058 | -0.0282 | 0.0098 | 0.6080 | CI includes zero |
| full_fusion | svm_rbf | S_R | 0.5242 | 0.5356 | -0.0114 | -0.0251 | -0.0001 | 0.0460 | CI excludes zero |
| full_fusion | svm_rbf | S_S | 0.5788 | 0.5952 | -0.0164 | -0.0280 | -0.0047 | 0.0090 | CI excludes zero |
| full_fusion | xgboost | R_R | 0.5624 | 0.5724 | -0.0100 | -0.0213 | 0.0003 | 0.0580 | CI includes zero |
| full_fusion | xgboost | R_S | 0.5218 | 0.5206 | 0.0012 | -0.0087 | 0.0110 | 0.8020 | CI includes zero |
| full_fusion | xgboost | S_R | 0.5049 | 0.5211 | -0.0163 | -0.0279 | -0.0054 | 0.0030 | CI excludes zero |
| full_fusion | xgboost | S_S | 0.5666 | 0.5959 | -0.0293 | -0.0455 | -0.0132 | 0.0000 | CI excludes zero |
| hubert | mlp | R_R | 0.5555 | 0.5617 | -0.0062 | -0.0196 | 0.0069 | 0.3780 | CI includes zero |
| hubert | mlp | R_S | 0.5197 | 0.5350 | -0.0152 | -0.0377 | 0.0080 | 0.1890 | CI includes zero |
| hubert | mlp | S_R | 0.5138 | 0.5339 | -0.0201 | -0.0409 | -0.0022 | 0.0230 | CI excludes zero |
| hubert | mlp | S_S | 0.5777 | 0.5780 | -0.0003 | -0.0164 | 0.0157 | 0.9430 | CI includes zero |
| hubert | random_forest | R_R | 0.5473 | 0.5607 | -0.0134 | -0.0239 | -0.0045 | 0.0070 | CI excludes zero |
| hubert | random_forest | R_S | 0.5414 | 0.5444 | -0.0030 | -0.0171 | 0.0138 | 0.6180 | CI includes zero |
| hubert | random_forest | S_R | 0.5140 | 0.5244 | -0.0104 | -0.0213 | 0.0010 | 0.0680 | CI includes zero |
| hubert | random_forest | S_S | 0.5746 | 0.5847 | -0.0101 | -0.0197 | -0.0006 | 0.0330 | CI excludes zero |
| hubert | svm_rbf | R_R | 0.5673 | 0.5767 | -0.0095 | -0.0238 | 0.0023 | 0.1360 | CI includes zero |
| hubert | svm_rbf | R_S | 0.5410 | 0.5298 | 0.0113 | -0.0007 | 0.0253 | 0.0620 | CI includes zero |
| hubert | svm_rbf | S_R | 0.5215 | 0.5503 | -0.0289 | -0.0553 | -0.0074 | 0.0050 | CI excludes zero |
| hubert | svm_rbf | S_S | 0.5923 | 0.5953 | -0.0030 | -0.0147 | 0.0090 | 0.6460 | CI includes zero |
| hubert | xgboost | R_R | 0.5510 | 0.5647 | -0.0137 | -0.0257 | -0.0027 | 0.0170 | CI excludes zero |
| hubert | xgboost | R_S | 0.5312 | 0.5465 | -0.0154 | -0.0331 | 0.0020 | 0.0800 | CI includes zero |
| hubert | xgboost | S_R | 0.5169 | 0.5418 | -0.0249 | -0.0389 | -0.0113 | 0.0000 | CI excludes zero |
| hubert | xgboost | S_S | 0.5704 | 0.5953 | -0.0249 | -0.0381 | -0.0120 | 0.0000 | CI excludes zero |
