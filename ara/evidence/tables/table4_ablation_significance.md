# Table 4: Leave-one-encoder-out ablation vs. full_fusion, paired bootstrap (balanced accuracy)

**Source**: results/P0_ablation_statistics/ablation_vs_full_fusion.csv

**Caption**: Paired subject-level bootstrap (2000 iterations), 89/96 evaluable combinations (7 skipped for incomplete fold coverage at time of test -- see skipped_combos.csv in the same directory).

**Extraction type**: raw_table

| representation | classifier | protocol | ablation_point | full_fusion_point | point_difference | ci95_low | ci95_high | p_value_two_sided | interpretation |
|---|---|---|---|---|---|---|---|---|---|
| full_minus_data2vec | mlp | R_R | 0.5617 | 0.5676 | -0.0060 | -0.0220 | 0.0095 | 0.4390 | CI includes zero |
| full_minus_data2vec | mlp | R_S | 0.5382 | 0.5365 | 0.0017 | -0.0169 | 0.0199 | 0.8660 | CI includes zero |
| full_minus_data2vec | mlp | S_R | 0.5231 | 0.5259 | -0.0028 | -0.0206 | 0.0158 | 0.8020 | CI includes zero |
| full_minus_data2vec | mlp | S_S | 0.5659 | 0.5748 | -0.0089 | -0.0311 | 0.0127 | 0.4420 | CI includes zero |
| full_minus_data2vec | random_forest | R_R | 0.5619 | 0.5679 | -0.0060 | -0.0157 | 0.0032 | 0.1920 | CI includes zero |
| full_minus_data2vec | random_forest | R_S | 0.5194 | 0.5196 | -0.0002 | -0.0135 | 0.0150 | 0.9680 | CI includes zero |
| full_minus_data2vec | random_forest | S_R | 0.5211 | 0.5218 | -0.0006 | -0.0151 | 0.0118 | 0.9890 | CI includes zero |
| full_minus_data2vec | random_forest | S_S | 0.5824 | 0.5916 | -0.0093 | -0.0199 | 0.0015 | 0.0930 | CI includes zero |
| full_minus_data2vec | svm_rbf | R_S | 0.5248 | 0.5195 | 0.0053 | -0.0086 | 0.0177 | 0.3970 | CI includes zero |
| full_minus_data2vec | svm_rbf | S_R | 0.5394 | 0.5356 | 0.0038 | -0.0099 | 0.0179 | 0.6180 | CI includes zero |
| full_minus_data2vec | xgboost | R_R | 0.5711 | 0.5724 | -0.0013 | -0.0113 | 0.0079 | 0.7940 | CI includes zero |
| full_minus_data2vec | xgboost | R_S | 0.5135 | 0.5206 | -0.0071 | -0.0221 | 0.0065 | 0.3400 | CI includes zero |
| full_minus_data2vec | xgboost | S_R | 0.5295 | 0.5211 | 0.0084 | -0.0026 | 0.0180 | 0.1260 | CI includes zero |
| full_minus_data2vec | xgboost | S_S | 0.5851 | 0.5959 | -0.0108 | -0.0261 | 0.0044 | 0.1870 | CI includes zero |
| full_minus_data2vec_audio | mlp | R_R | 0.5776 | 0.5676 | 0.0100 | -0.0043 | 0.0244 | 0.1770 | CI includes zero |
| full_minus_data2vec_audio | mlp | R_S | 0.5304 | 0.5365 | -0.0061 | -0.0256 | 0.0170 | 0.6100 | CI includes zero |
| full_minus_data2vec_audio | mlp | S_R | 0.5451 | 0.5259 | 0.0193 | 0.0049 | 0.0342 | 0.0070 | CI excludes zero |
| full_minus_data2vec_audio | mlp | S_S | 0.5882 | 0.5748 | 0.0134 | -0.0072 | 0.0325 | 0.2160 | CI includes zero |
| full_minus_data2vec_audio | random_forest | R_R | 0.5742 | 0.5679 | 0.0063 | -0.0006 | 0.0129 | 0.0680 | CI includes zero |
| full_minus_data2vec_audio | random_forest | R_S | 0.5230 | 0.5196 | 0.0034 | -0.0078 | 0.0144 | 0.5740 | CI includes zero |
| full_minus_data2vec_audio | random_forest | S_R | 0.5285 | 0.5218 | 0.0067 | -0.0062 | 0.0179 | 0.3120 | CI includes zero |
| full_minus_data2vec_audio | random_forest | S_S | 0.5861 | 0.5916 | -0.0055 | -0.0137 | 0.0017 | 0.1490 | CI includes zero |
| full_minus_data2vec_audio | svm_rbf | R_S | 0.5170 | 0.5195 | -0.0025 | -0.0197 | 0.0103 | 0.8290 | CI includes zero |
| full_minus_data2vec_audio | svm_rbf | S_R | 0.5464 | 0.5356 | 0.0108 | 0.0062 | 0.0157 | 0.0000 | CI excludes zero |
| full_minus_data2vec_audio | xgboost | R_R | 0.5777 | 0.5724 | 0.0053 | -0.0027 | 0.0127 | 0.1840 | CI includes zero |
| full_minus_data2vec_audio | xgboost | R_S | 0.5174 | 0.5206 | -0.0032 | -0.0158 | 0.0063 | 0.6540 | CI includes zero |
| full_minus_data2vec_audio | xgboost | S_R | 0.5350 | 0.5211 | 0.0138 | 0.0019 | 0.0274 | 0.0200 | CI excludes zero |
| full_minus_data2vec_audio | xgboost | S_S | 0.5940 | 0.5959 | -0.0019 | -0.0097 | 0.0058 | 0.6370 | CI includes zero |
| full_minus_data2vec_spectrogram | mlp | R_R | 0.5600 | 0.5676 | -0.0076 | -0.0197 | 0.0051 | 0.2310 | CI includes zero |
| full_minus_data2vec_spectrogram | mlp | R_S | 0.5400 | 0.5365 | 0.0035 | -0.0193 | 0.0291 | 0.8490 | CI includes zero |
| full_minus_data2vec_spectrogram | mlp | S_R | 0.5198 | 0.5259 | -0.0061 | -0.0205 | 0.0077 | 0.4110 | CI includes zero |
| full_minus_data2vec_spectrogram | mlp | S_S | 0.5720 | 0.5748 | -0.0028 | -0.0178 | 0.0129 | 0.7330 | CI includes zero |
| full_minus_data2vec_spectrogram | random_forest | R_R | 0.5710 | 0.5679 | 0.0031 | -0.0043 | 0.0108 | 0.4300 | CI includes zero |
| full_minus_data2vec_spectrogram | random_forest | R_S | 0.5303 | 0.5196 | 0.0107 | -0.0037 | 0.0262 | 0.1600 | CI includes zero |
| full_minus_data2vec_spectrogram | random_forest | S_R | 0.5156 | 0.5218 | -0.0062 | -0.0216 | 0.0070 | 0.3820 | CI includes zero |
| full_minus_data2vec_spectrogram | random_forest | S_S | 0.5811 | 0.5916 | -0.0106 | -0.0207 | -0.0006 | 0.0380 | CI excludes zero |
| full_minus_data2vec_spectrogram | svm_rbf | R_S | 0.5158 | 0.5195 | -0.0037 | -0.0136 | 0.0048 | 0.4200 | CI includes zero |
| full_minus_data2vec_spectrogram | svm_rbf | S_R | 0.5334 | 0.5356 | -0.0022 | -0.0141 | 0.0084 | 0.7670 | CI includes zero |
| full_minus_data2vec_spectrogram | xgboost | R_R | 0.5615 | 0.5724 | -0.0109 | -0.0258 | 0.0020 | 0.1070 | CI includes zero |
| full_minus_data2vec_spectrogram | xgboost | R_S | 0.5218 | 0.5206 | 0.0012 | -0.0106 | 0.0133 | 0.8900 | CI includes zero |
| full_minus_data2vec_spectrogram | xgboost | S_R | 0.5262 | 0.5211 | 0.0051 | -0.0049 | 0.0152 | 0.3150 | CI includes zero |
| full_minus_data2vec_spectrogram | xgboost | S_S | 0.5900 | 0.5959 | -0.0059 | -0.0191 | 0.0055 | 0.3310 | CI includes zero |
| full_minus_hubert | mlp | R_R | 0.5656 | 0.5676 | -0.0021 | -0.0118 | 0.0076 | 0.7170 | CI includes zero |
| full_minus_hubert | mlp | R_S | 0.5419 | 0.5365 | 0.0054 | -0.0162 | 0.0292 | 0.6480 | CI includes zero |
| full_minus_hubert | mlp | S_R | 0.5464 | 0.5259 | 0.0206 | 0.0010 | 0.0435 | 0.0290 | CI excludes zero |
| full_minus_hubert | mlp | S_S | 0.5799 | 0.5748 | 0.0051 | -0.0132 | 0.0231 | 0.5790 | CI includes zero |
| full_minus_hubert | random_forest | R_R | 0.5669 | 0.5679 | -0.0010 | -0.0078 | 0.0068 | 0.7680 | CI includes zero |
| full_minus_hubert | random_forest | R_S | 0.5231 | 0.5196 | 0.0035 | -0.0106 | 0.0212 | 0.7060 | CI includes zero |
| full_minus_hubert | random_forest | S_R | 0.5170 | 0.5218 | -0.0048 | -0.0154 | 0.0056 | 0.3690 | CI includes zero |
| full_minus_hubert | random_forest | S_S | 0.5835 | 0.5916 | -0.0082 | -0.0175 | 0.0017 | 0.1120 | CI includes zero |
| full_minus_hubert | svm_rbf | R_R | 0.5737 | 0.5783 | -0.0046 | -0.0120 | 0.0022 | 0.1910 | CI includes zero |
| full_minus_hubert | svm_rbf | R_S | 0.5224 | 0.5195 | 0.0029 | -0.0093 | 0.0145 | 0.6310 | CI includes zero |
| full_minus_hubert | svm_rbf | S_R | 0.5353 | 0.5356 | -0.0003 | -0.0106 | 0.0101 | 0.9600 | CI includes zero |
| full_minus_hubert | svm_rbf | S_S | 0.5929 | 0.5952 | -0.0023 | -0.0085 | 0.0038 | 0.4780 | CI includes zero |
| full_minus_hubert | xgboost | R_R | 0.5744 | 0.5724 | 0.0020 | -0.0046 | 0.0093 | 0.5650 | CI includes zero |
| full_minus_hubert | xgboost | R_S | 0.5168 | 0.5206 | -0.0038 | -0.0147 | 0.0073 | 0.4960 | CI includes zero |
| full_minus_hubert | xgboost | S_R | 0.5279 | 0.5211 | 0.0068 | -0.0062 | 0.0196 | 0.2940 | CI includes zero |
| full_minus_hubert | xgboost | S_S | 0.5868 | 0.5959 | -0.0091 | -0.0189 | 0.0005 | 0.0620 | CI includes zero |
| full_minus_wav2vec2 | mlp | R_R | 0.5633 | 0.5676 | -0.0044 | -0.0182 | 0.0115 | 0.5200 | CI includes zero |
| full_minus_wav2vec2 | mlp | R_S | 0.5357 | 0.5365 | -0.0008 | -0.0300 | 0.0274 | 0.9420 | CI includes zero |
| full_minus_wav2vec2 | mlp | S_R | 0.5348 | 0.5259 | 0.0089 | -0.0086 | 0.0267 | 0.3160 | CI includes zero |
| full_minus_wav2vec2 | mlp | S_S | 0.5756 | 0.5748 | 0.0008 | -0.0161 | 0.0183 | 0.9320 | CI includes zero |
| full_minus_wav2vec2 | random_forest | R_R | 0.5645 | 0.5679 | -0.0035 | -0.0111 | 0.0040 | 0.3760 | CI includes zero |
| full_minus_wav2vec2 | random_forest | R_S | 0.5174 | 0.5196 | -0.0022 | -0.0116 | 0.0078 | 0.6030 | CI includes zero |
| full_minus_wav2vec2 | random_forest | S_R | 0.5178 | 0.5218 | -0.0040 | -0.0166 | 0.0065 | 0.5070 | CI includes zero |
| full_minus_wav2vec2 | random_forest | S_S | 0.5879 | 0.5916 | -0.0037 | -0.0129 | 0.0040 | 0.4160 | CI includes zero |
| full_minus_wav2vec2 | svm_rbf | R_R | 0.5843 | 0.5783 | 0.0060 | -0.0016 | 0.0154 | 0.1680 | CI includes zero |
| full_minus_wav2vec2 | svm_rbf | R_S | 0.5183 | 0.5195 | -0.0012 | -0.0131 | 0.0079 | 0.8530 | CI includes zero |
| full_minus_wav2vec2 | svm_rbf | S_R | 0.5267 | 0.5356 | -0.0089 | -0.0191 | -0.0001 | 0.0500 | CI excludes zero |
| full_minus_wav2vec2 | xgboost | R_R | 0.5730 | 0.5724 | 0.0006 | -0.0090 | 0.0098 | 0.9080 | CI includes zero |
| full_minus_wav2vec2 | xgboost | R_S | 0.5232 | 0.5206 | 0.0026 | -0.0106 | 0.0154 | 0.6340 | CI includes zero |
| full_minus_wav2vec2 | xgboost | S_R | 0.5277 | 0.5211 | 0.0066 | -0.0018 | 0.0154 | 0.1300 | CI includes zero |
| full_minus_wav2vec2 | xgboost | S_S | 0.5907 | 0.5959 | -0.0052 | -0.0152 | 0.0048 | 0.2930 | CI includes zero |
| full_minus_wavlm | mlp | R_R | 0.5659 | 0.5676 | -0.0018 | -0.0139 | 0.0112 | 0.7890 | CI includes zero |
| full_minus_wavlm | mlp | R_S | 0.5140 | 0.5365 | -0.0225 | -0.0362 | -0.0101 | 0.0010 | CI excludes zero |
| full_minus_wavlm | mlp | S_R | 0.5298 | 0.5259 | 0.0039 | -0.0141 | 0.0213 | 0.6540 | CI includes zero |
| full_minus_wavlm | mlp | S_S | 0.5733 | 0.5748 | -0.0015 | -0.0172 | 0.0158 | 0.8490 | CI includes zero |
| full_minus_wavlm | random_forest | R_R | 0.5730 | 0.5679 | 0.0051 | -0.0013 | 0.0122 | 0.1170 | CI includes zero |
| full_minus_wavlm | random_forest | R_S | 0.5186 | 0.5196 | -0.0010 | -0.0129 | 0.0151 | 0.7660 | CI includes zero |
| full_minus_wavlm | random_forest | S_R | 0.5230 | 0.5218 | 0.0012 | -0.0091 | 0.0116 | 0.7620 | CI includes zero |
| full_minus_wavlm | random_forest | S_S | 0.5924 | 0.5916 | 0.0008 | -0.0051 | 0.0076 | 0.7560 | CI includes zero |
| full_minus_wavlm | svm_rbf | R_R | 0.5800 | 0.5783 | 0.0017 | -0.0096 | 0.0177 | 0.8900 | CI includes zero |
| full_minus_wavlm | svm_rbf | R_S | 0.5160 | 0.5195 | -0.0035 | -0.0130 | 0.0052 | 0.4770 | CI includes zero |
| full_minus_wavlm | svm_rbf | S_R | 0.5299 | 0.5356 | -0.0057 | -0.0145 | 0.0031 | 0.1850 | CI includes zero |
| full_minus_wavlm | svm_rbf | S_S | 0.5938 | 0.5952 | -0.0014 | -0.0105 | 0.0083 | 0.7630 | CI includes zero |
| full_minus_wavlm | xgboost | R_R | 0.5666 | 0.5724 | -0.0058 | -0.0131 | 0.0011 | 0.0990 | CI includes zero |
| full_minus_wavlm | xgboost | R_S | 0.5206 | 0.5206 | -0.0000 | -0.0094 | 0.0092 | 0.9590 | CI includes zero |
| full_minus_wavlm | xgboost | S_R | 0.5265 | 0.5211 | 0.0054 | -0.0041 | 0.0143 | 0.2510 | CI includes zero |
| full_minus_wavlm | xgboost | S_S | 0.5884 | 0.5959 | -0.0075 | -0.0164 | 0.0019 | 0.1220 | CI includes zero |
