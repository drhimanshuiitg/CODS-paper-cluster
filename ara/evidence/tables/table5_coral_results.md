# Table 5: CORAL feature-space alignment, full results

**Source**: results/P1_domain_adaptation (30 completed combinations)

**Caption**: All CORAL-aligned classifier results (source-device train+val, target-device-validation-fit covariance alignment, svm_rbf only). Compare against Table 1 rows for the same representation/protocol (uncorrected baseline).

**Extraction type**: raw_table

| representation | protocol | fold | classifier | balanced_accuracy | sensitivity | specificity |
|---|---|---|---|---|---|---|
| data2vec_fusion_coral | R_S | 0 | svm_rbf | 0.5018 | 0.0041 | 0.9995 |
| data2vec_fusion_coral | R_S | 1 | svm_rbf | 0.5017 | 0.0801 | 0.9233 |
| data2vec_fusion_coral | R_S | 2 | svm_rbf | 0.5064 | 0.0497 | 0.9630 |
| data2vec_fusion_coral | R_S | 3 | svm_rbf | 0.5365 | 0.1568 | 0.9162 |
| data2vec_fusion_coral | R_S | 4 | svm_rbf | 0.4997 | 0.0328 | 0.9666 |
| data2vec_fusion_coral | S_R | 0 | svm_rbf | 0.5000 | 0.0415 | 0.9586 |
| data2vec_fusion_coral | S_R | 1 | svm_rbf | 0.5085 | 0.1135 | 0.9036 |
| data2vec_fusion_coral | S_R | 2 | svm_rbf | 0.5078 | 0.0592 | 0.9564 |
| data2vec_fusion_coral | S_R | 3 | svm_rbf | 0.5126 | 0.0764 | 0.9488 |
| data2vec_fusion_coral | S_R | 4 | svm_rbf | 0.4922 | 0.1312 | 0.8532 |
| full_fusion_coral | R_S | 0 | svm_rbf | 0.5005 | 0.0041 | 0.9970 |
| full_fusion_coral | R_S | 1 | svm_rbf | 0.4979 | 0.0139 | 0.9818 |
| full_fusion_coral | R_S | 2 | svm_rbf | 0.5018 | 0.0035 | 1.0000 |
| full_fusion_coral | R_S | 3 | svm_rbf | 0.5089 | 0.0344 | 0.9834 |
| full_fusion_coral | R_S | 4 | svm_rbf | 0.5000 | 0.0000 | 1.0000 |
| full_fusion_coral | S_R | 0 | svm_rbf | 0.5114 | 0.0415 | 0.9813 |
| full_fusion_coral | S_R | 1 | svm_rbf | 0.4988 | 0.0299 | 0.9677 |
| full_fusion_coral | S_R | 2 | svm_rbf | 0.4964 | 0.0075 | 0.9853 |
| full_fusion_coral | S_R | 3 | svm_rbf | 0.4985 | 0.0121 | 0.9849 |
| full_fusion_coral | S_R | 4 | svm_rbf | 0.5097 | 0.0565 | 0.9629 |
| hubert_coral | R_S | 0 | svm_rbf | 0.5258 | 0.1235 | 0.9280 |
| hubert_coral | R_S | 1 | svm_rbf | 0.4986 | 0.2046 | 0.7925 |
| hubert_coral | R_S | 2 | svm_rbf | 0.5137 | 0.1064 | 0.9210 |
| hubert_coral | R_S | 3 | svm_rbf | 0.5442 | 0.2018 | 0.8866 |
| hubert_coral | R_S | 4 | svm_rbf | 0.5011 | 0.1029 | 0.8993 |
| hubert_coral | S_R | 0 | svm_rbf | 0.5478 | 0.3220 | 0.7737 |
| hubert_coral | S_R | 1 | svm_rbf | 0.5313 | 0.2449 | 0.8178 |
| hubert_coral | S_R | 2 | svm_rbf | 0.5271 | 0.1727 | 0.8815 |
| hubert_coral | S_R | 3 | svm_rbf | 0.5492 | 0.3829 | 0.7155 |
| hubert_coral | S_R | 4 | svm_rbf | 0.5220 | 0.3451 | 0.6990 |
