# Table 2: PCA target-aware-refit fix vs. pre-fix baseline, paired bootstrap (balanced accuracy)

**Source**: results/P1_statistics_pca_fix/pca_fix_vs_baseline.csv (full CSV has 4 metrics x 60 combos; this table is the balanced_accuracy rows only)

**Caption**: Paired subject-level bootstrap (2000 iterations) comparing post-fix (P1_dimension_control_v3) vs pre-fix (P1_dimension_control) on all 60 overlapping representation/classifier/protocol/fold/dimension combinations.

**Extraction type**: raw_table

| representation | classifier | protocol | fold | dimension | fixed_point | baseline_point | point_difference | ci95_low | ci95_high | p_value_two_sided | interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_fusion | svm_rbf | R_S | 0 | 384 | 0.5331 | 0.5002 | 0.0330 | -0.0088 | 0.0925 | 0.2400 | CI includes zero |
| full_fusion | svm_rbf | R_S | 0 | 768 | 0.5211 | 0.5000 | 0.0211 | -0.0104 | 0.0704 | 0.4540 | CI includes zero |
| full_fusion | svm_rbf | R_S | 0 | 1536 | 0.5273 | 0.5000 | 0.0273 | 0.0002 | 0.0647 | 0.0450 | CI excludes zero |
| full_fusion | svm_rbf | R_S | 1 | 384 | 0.5337 | 0.5000 | 0.0337 | 0.0043 | 0.0714 | 0.0170 | CI excludes zero |
| full_fusion | svm_rbf | R_S | 1 | 768 | 0.5306 | 0.5000 | 0.0306 | 0.0090 | 0.0529 | 0.0040 | CI excludes zero |
| full_fusion | svm_rbf | R_S | 1 | 1536 | 0.5424 | 0.5000 | 0.0424 | 0.0142 | 0.0725 | 0.0040 | CI excludes zero |
| full_fusion | svm_rbf | R_S | 2 | 384 | 0.5097 | 0.5000 | 0.0097 | -0.0192 | 0.0401 | 0.6070 | CI includes zero |
| full_fusion | svm_rbf | R_S | 2 | 768 | 0.5077 | 0.5000 | 0.0077 | -0.0155 | 0.0340 | 0.6320 | CI includes zero |
| full_fusion | svm_rbf | R_S | 2 | 1536 | 0.5105 | 0.5000 | 0.0105 | -0.0022 | 0.0255 | 0.0980 | CI includes zero |
| full_fusion | svm_rbf | R_S | 3 | 384 | 0.5411 | 0.5028 | 0.0383 | -0.0171 | 0.0988 | 0.2130 | CI includes zero |
| full_fusion | svm_rbf | R_S | 3 | 768 | 0.5417 | 0.5000 | 0.0417 | 0.0027 | 0.0878 | 0.0320 | CI excludes zero |
| full_fusion | svm_rbf | R_S | 3 | 1536 | 0.5107 | 0.5000 | 0.0107 | -0.0068 | 0.0284 | 0.2510 | CI includes zero |
| full_fusion | svm_rbf | R_S | 4 | 384 | 0.5308 | 0.5000 | 0.0308 | -0.0022 | 0.0687 | 0.0710 | CI includes zero |
| full_fusion | svm_rbf | R_S | 4 | 768 | 0.5100 | 0.5000 | 0.0100 | -0.0096 | 0.0319 | 0.3580 | CI includes zero |
| full_fusion | svm_rbf | R_S | 4 | 1536 | 0.5141 | 0.5000 | 0.0141 | 0.0015 | 0.0269 | 0.0260 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 0 | 384 | 0.5418 | 0.5256 | 0.0162 | -0.0365 | 0.0651 | 0.5540 | CI includes zero |
| full_fusion | svm_rbf | S_R | 0 | 768 | 0.5181 | 0.5000 | 0.0181 | -0.0532 | 0.0810 | 0.5790 | CI includes zero |
| full_fusion | svm_rbf | S_R | 0 | 1536 | 0.5252 | 0.5000 | 0.0252 | -0.0395 | 0.0863 | 0.4180 | CI includes zero |
| full_fusion | svm_rbf | S_R | 1 | 384 | 0.5758 | 0.5160 | 0.0598 | 0.0209 | 0.1000 | 0.0040 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 1 | 768 | 0.5641 | 0.5000 | 0.0641 | 0.0178 | 0.1095 | 0.0100 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 1 | 1536 | 0.5507 | 0.5000 | 0.0507 | 0.0004 | 0.1088 | 0.0470 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 2 | 384 | 0.5600 | 0.5037 | 0.0562 | 0.0364 | 0.0790 | 0.0000 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 2 | 768 | 0.5083 | 0.5000 | 0.0083 | -0.0246 | 0.0439 | 0.6490 | CI includes zero |
| full_fusion | svm_rbf | S_R | 2 | 1536 | 0.5380 | 0.5000 | 0.0380 | 0.0098 | 0.0641 | 0.0070 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 3 | 384 | 0.5766 | 0.5335 | 0.0431 | 0.0235 | 0.0609 | 0.0000 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 3 | 768 | 0.5306 | 0.5000 | 0.0306 | 0.0122 | 0.0506 | 0.0010 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 3 | 1536 | 0.5465 | 0.5000 | 0.0465 | 0.0162 | 0.0730 | 0.0010 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 4 | 384 | 0.5558 | 0.4995 | 0.0563 | 0.0281 | 0.0847 | 0.0000 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 4 | 768 | 0.5528 | 0.5000 | 0.0528 | 0.0130 | 0.0987 | 0.0010 | CI excludes zero |
| full_fusion | svm_rbf | S_R | 4 | 1536 | 0.5587 | 0.5000 | 0.0587 | 0.0206 | 0.0944 | 0.0000 | CI excludes zero |
| full_fusion_v2 | svm_rbf | R_S | 0 | 384 | 0.5153 | 0.5003 | 0.0150 | -0.0587 | 0.1010 | 0.7660 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 0 | 768 | 0.5223 | 0.5000 | 0.0223 | -0.0164 | 0.0798 | 0.4260 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 0 | 1536 | 0.5373 | 0.5000 | 0.0373 | 0.0010 | 0.0954 | 0.0430 | CI excludes zero |
| full_fusion_v2 | svm_rbf | R_S | 1 | 384 | 0.5384 | 0.5016 | 0.0368 | 0.0150 | 0.0645 | 0.0000 | CI excludes zero |
| full_fusion_v2 | svm_rbf | R_S | 1 | 768 | 0.5340 | 0.5000 | 0.0340 | 0.0039 | 0.0619 | 0.0300 | CI excludes zero |
| full_fusion_v2 | svm_rbf | R_S | 1 | 1536 | 0.5226 | 0.5000 | 0.0226 | -0.0103 | 0.0536 | 0.1660 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 2 | 384 | 0.4981 | 0.5001 | -0.0020 | -0.0321 | 0.0274 | 0.8680 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 2 | 768 | 0.5024 | 0.5000 | 0.0024 | -0.0173 | 0.0239 | 0.8490 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 2 | 1536 | 0.5022 | 0.5000 | 0.0022 | -0.0132 | 0.0207 | 0.8370 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 3 | 384 | 0.5269 | 0.5018 | 0.0251 | -0.0148 | 0.0696 | 0.2490 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 3 | 768 | 0.5308 | 0.5000 | 0.0308 | -0.0084 | 0.0721 | 0.1200 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 3 | 1536 | 0.5327 | 0.5000 | 0.0327 | 0.0082 | 0.0558 | 0.0050 | CI excludes zero |
| full_fusion_v2 | svm_rbf | R_S | 4 | 384 | 0.5191 | 0.4990 | 0.0201 | -0.0173 | 0.0630 | 0.3380 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 4 | 768 | 0.5236 | 0.5000 | 0.0236 | -0.0099 | 0.0568 | 0.1600 | CI includes zero |
| full_fusion_v2 | svm_rbf | R_S | 4 | 1536 | 0.5259 | 0.5000 | 0.0259 | 0.0035 | 0.0535 | 0.0110 | CI excludes zero |
| full_fusion_v2 | svm_rbf | S_R | 0 | 384 | 0.5384 | 0.5389 | -0.0006 | -0.0605 | 0.0584 | 0.9890 | CI includes zero |
| full_fusion_v2 | svm_rbf | S_R | 0 | 768 | 0.5138 | 0.5000 | 0.0138 | -0.0457 | 0.0709 | 0.6880 | CI includes zero |
| full_fusion_v2 | svm_rbf | S_R | 0 | 1536 | 0.5280 | 0.5000 | 0.0280 | -0.0170 | 0.0780 | 0.2450 | CI includes zero |
| full_fusion_v2 | svm_rbf | S_R | 1 | 384 | 0.5564 | 0.5632 | -0.0068 | -0.0389 | 0.0285 | 0.7160 | CI includes zero |
| full_fusion_v2 | svm_rbf | S_R | 1 | 768 | 0.5408 | 0.5000 | 0.0408 | 0.0078 | 0.0809 | 0.0090 | CI excludes zero |
| full_fusion_v2 | svm_rbf | S_R | 1 | 1536 | 0.5586 | 0.5000 | 0.0586 | 0.0155 | 0.1090 | 0.0030 | CI excludes zero |
| full_fusion_v2 | svm_rbf | S_R | 2 | 384 | 0.5623 | 0.4964 | 0.0659 | 0.0410 | 0.1002 | 0.0000 | CI excludes zero |
| full_fusion_v2 | svm_rbf | S_R | 2 | 768 | 0.5077 | 0.5000 | 0.0077 | -0.0252 | 0.0413 | 0.6570 | CI includes zero |
| full_fusion_v2 | svm_rbf | S_R | 2 | 1536 | 0.5195 | 0.5000 | 0.0195 | -0.0111 | 0.0521 | 0.2210 | CI includes zero |
| full_fusion_v2 | svm_rbf | S_R | 3 | 384 | 0.5439 | 0.5254 | 0.0185 | -0.0087 | 0.0455 | 0.1790 | CI includes zero |
| full_fusion_v2 | svm_rbf | S_R | 3 | 768 | 0.5414 | 0.5000 | 0.0414 | 0.0071 | 0.0762 | 0.0130 | CI excludes zero |
| full_fusion_v2 | svm_rbf | S_R | 3 | 1536 | 0.5397 | 0.5000 | 0.0397 | 0.0128 | 0.0669 | 0.0020 | CI excludes zero |
| full_fusion_v2 | svm_rbf | S_R | 4 | 384 | 0.5596 | 0.4994 | 0.0601 | 0.0289 | 0.0942 | 0.0000 | CI excludes zero |
| full_fusion_v2 | svm_rbf | S_R | 4 | 768 | 0.5606 | 0.5000 | 0.0606 | 0.0208 | 0.1063 | 0.0010 | CI excludes zero |
| full_fusion_v2 | svm_rbf | S_R | 4 | 1536 | 0.5319 | 0.5000 | 0.0319 | 0.0029 | 0.0620 | 0.0230 | CI excludes zero |
