# Table 3: Sliding-window whole-night severity classification, FULL results (80/80 complete)

**Source**: results/P3_sliding_window_severity (80/80 combinations, completed 2026-08-19)

**Caption**: Every completed combination, all 4 protocols. Superseded the earlier 28/80 partial snapshot (matched-device only) this table originally held -- now includes cross-device (R_S, S_R) rows.

**Extraction type**: raw_table

| protocol | classifier | fold | balanced_accuracy | f1 | roc_auc |
|---|---|---|---|---|---|
| R_R | mlp | 0 | 0.5565 | 0.5462 | 0.5518 |
| R_R | mlp | 1 | 0.4956 | 0.5532 | 0.5003 |
| R_R | mlp | 2 | 0.4685 | 0.4628 | 0.4582 |
| R_R | mlp | 3 | 0.6394 | 0.5792 | 0.6854 |
| R_R | mlp | 4 | 0.4721 | 0.4056 | 0.4816 |
| R_R | random_forest | 0 | 0.5446 | 0.4783 | 0.5527 |
| R_R | random_forest | 1 | 0.5005 | 0.5589 | 0.5166 |
| R_R | random_forest | 2 | 0.5121 | 0.4723 | 0.5247 |
| R_R | random_forest | 3 | 0.6008 | 0.5109 | 0.6307 |
| R_R | random_forest | 4 | 0.4866 | 0.4150 | 0.4606 |
| R_R | svm_rbf | 0 | 0.5235 | 0.4743 | 0.5609 |
| R_R | svm_rbf | 1 | 0.4697 | 0.5563 | 0.4495 |
| R_R | svm_rbf | 2 | 0.5005 | 0.4925 | 0.5010 |
| R_R | svm_rbf | 3 | 0.5730 | 0.4726 | 0.5925 |
| R_R | svm_rbf | 4 | 0.4589 | 0.4864 | 0.4759 |
| R_R | xgboost | 0 | 0.5434 | 0.4860 | 0.5594 |
| R_R | xgboost | 1 | 0.4901 | 0.5497 | 0.5054 |
| R_R | xgboost | 2 | 0.4838 | 0.4409 | 0.4965 |
| R_R | xgboost | 3 | 0.6077 | 0.5222 | 0.6381 |
| R_R | xgboost | 4 | 0.4908 | 0.4343 | 0.4712 |
| R_S | mlp | 0 | 0.5022 | 0.4686 | 0.4818 |
| R_S | mlp | 1 | 0.4398 | 0.5478 | 0.3718 |
| R_S | mlp | 2 | 0.4810 | 0.3647 | 0.4814 |
| R_S | mlp | 3 | 0.5732 | 0.5129 | 0.5982 |
| R_S | mlp | 4 | 0.5056 | 0.4495 | 0.5333 |
| R_S | random_forest | 0 | 0.5538 | 0.5455 | 0.5222 |
| R_S | random_forest | 1 | 0.4590 | 0.5696 | 0.3999 |
| R_S | random_forest | 2 | 0.4226 | 0.3793 | 0.3711 |
| R_S | random_forest | 3 | 0.5682 | 0.5196 | 0.5311 |
| R_S | random_forest | 4 | 0.5494 | 0.5687 | 0.5713 |
| R_S | svm_rbf | 0 | 0.5119 | 0.4696 | 0.5226 |
| R_S | svm_rbf | 1 | 0.4738 | 0.5925 | 0.5178 |
| R_S | svm_rbf | 2 | 0.3895 | 0.3690 | 0.3522 |
| R_S | svm_rbf | 3 | 0.5095 | 0.4732 | 0.5399 |
| R_S | svm_rbf | 4 | 0.5196 | 0.5370 | 0.5268 |
| R_S | xgboost | 0 | 0.5344 | 0.5255 | 0.5161 |
| R_S | xgboost | 1 | 0.4373 | 0.5267 | 0.4116 |
| R_S | xgboost | 2 | 0.3728 | 0.3118 | 0.3443 |
| R_S | xgboost | 3 | 0.5050 | 0.4492 | 0.4893 |
| R_S | xgboost | 4 | 0.5412 | 0.5328 | 0.5574 |
| S_R | mlp | 0 | 0.5482 | 0.5088 | 0.5282 |
| S_R | mlp | 1 | 0.2980 | 0.3571 | 0.2655 |
| S_R | mlp | 2 | 0.4891 | 0.3955 | 0.4649 |
| S_R | mlp | 3 | 0.5565 | 0.4800 | 0.5886 |
| S_R | mlp | 4 | 0.5740 | 0.5786 | 0.5561 |
| S_R | random_forest | 0 | 0.5257 | 0.4914 | 0.5091 |
| S_R | random_forest | 1 | 0.3175 | 0.3363 | 0.2963 |
| S_R | random_forest | 2 | 0.4673 | 0.4183 | 0.4561 |
| S_R | random_forest | 3 | 0.5713 | 0.5217 | 0.6069 |
| S_R | random_forest | 4 | 0.5490 | 0.5977 | 0.5468 |
| S_R | svm_rbf | 0 | 0.5278 | 0.5134 | 0.5308 |
| S_R | svm_rbf | 1 | 0.3448 | 0.4655 | 0.2823 |
| S_R | svm_rbf | 2 | 0.4416 | 0.4500 | 0.4256 |
| S_R | svm_rbf | 3 | 0.5180 | 0.4852 | 0.5989 |
| S_R | svm_rbf | 4 | 0.4936 | 0.5700 | 0.5064 |
| S_R | xgboost | 0 | 0.4835 | 0.4587 | 0.4869 |
| S_R | xgboost | 1 | 0.3229 | 0.3396 | 0.3007 |
| S_R | xgboost | 2 | 0.4735 | 0.4358 | 0.4507 |
| S_R | xgboost | 3 | 0.5578 | 0.5127 | 0.5815 |
| S_R | xgboost | 4 | 0.5246 | 0.5831 | 0.5362 |
| S_S | mlp | 0 | 0.5439 | 0.4811 | 0.5687 |
| S_S | mlp | 1 | 0.5581 | 0.6496 | 0.5477 |
| S_S | mlp | 2 | 0.4634 | 0.3882 | 0.4648 |
| S_S | mlp | 3 | 0.5584 | 0.4957 | 0.5796 |
| S_S | mlp | 4 | 0.5637 | 0.4482 | 0.5148 |
| S_S | random_forest | 0 | 0.5487 | 0.4965 | 0.5627 |
| S_S | random_forest | 1 | 0.5098 | 0.5945 | 0.5141 |
| S_S | random_forest | 2 | 0.4903 | 0.4810 | 0.4893 |
| S_S | random_forest | 3 | 0.5998 | 0.5499 | 0.6416 |
| S_S | random_forest | 4 | 0.5081 | 0.4543 | 0.4603 |
| S_S | svm_rbf | 0 | 0.5740 | 0.5204 | 0.5664 |
| S_S | svm_rbf | 1 | 0.5783 | 0.6606 | 0.5776 |
| S_S | svm_rbf | 2 | 0.4593 | 0.4725 | 0.4614 |
| S_S | svm_rbf | 3 | 0.6200 | 0.5605 | 0.6333 |
| S_S | svm_rbf | 4 | 0.5084 | 0.4964 | 0.4881 |
| S_S | xgboost | 0 | 0.5721 | 0.5219 | 0.5822 |
| S_S | xgboost | 1 | 0.5132 | 0.5654 | 0.5068 |
| S_S | xgboost | 2 | 0.5085 | 0.4907 | 0.4813 |
| S_S | xgboost | 3 | 0.6036 | 0.5507 | 0.6339 |
| S_S | xgboost | 4 | 0.5305 | 0.4635 | 0.4640 |
