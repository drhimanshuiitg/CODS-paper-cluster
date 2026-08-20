# Ablation Table — Leave-One-Encoder-Out vs. full_fusion

**Source**: `results/P0_ablation_statistics/ablation_vs_full_fusion.csv`, 89/96 evaluable combinations (paired subject-level bootstrap, 2,000 resamples).

## Summary by ablated component

| Ablated encoder | # combinations tested | # significant | Direction of significant results |
|---|---|---|---|
| data2vec (spectrogram+audio jointly, `full_minus_data2vec`) | 14 | 0 | — |
| data2vec_audio | 14 | 2 | Both positive (removing it *helped* in those 2 configs) |
| data2vec_spectrogram | 14 | 1 | Negative (removing it hurt) |
| hubert | 14 | 1 | Positive (removing it helped) |
| wav2vec2 | 14 | 1 | Negative (removing it hurt) |
| wavlm | 14 | 1 | Negative (removing it hurt) |
| **Total** | **89*** (7 skipped, incomplete fold coverage) | **7** | 4 positive, 3 negative |

**Interpretation.** No encoder shows a consistent, one-directional significant effect across its own tested combinations — e.g., removing `data2vec_audio` is significantly *better* in 2 configurations and not significant in the other 12, which is inconsistent with that encoder carrying essential unique signal. This supports Section 6.4's conclusion that no individual encoder is confirmed necessary in the fusion, though the caveat in `tables/statistical_analysis.md` (Section B, no multiple-comparison correction) applies.

## Full significant-result rows (7/89)

| Ablated variant | Classifier | Protocol | Point difference | 95% CI | p-value |
|---|---|---|---|---|---|
| full_minus_data2vec_audio | mlp | S_R | +0.0193 | [0.0049, 0.0342] | 0.007 |
| full_minus_data2vec_audio | svm_rbf | S_R | +0.0108 | [0.0062, 0.0157] | <0.001 |
| full_minus_data2vec_audio | xgboost | S_R | +0.0138 | [0.0019, 0.0274] | 0.020 |
| full_minus_data2vec_spectrogram | random_forest | S_S | −0.0106 | [−0.0207, −0.0006] | 0.038 |
| full_minus_hubert | mlp | S_R | +0.0206 | [0.0010, 0.0435] | 0.029 |
| full_minus_wav2vec2 | svm_rbf | S_R | −0.0089 | [−0.0191, −0.0001] | 0.050 |
| full_minus_wavlm | mlp | R_S | −0.0225 | [−0.0362, −0.0101] | 0.001 |

Full 89-row table available in the accompanying ARA (`ara/evidence/tables/table4_ablation_significance.md`) and `MASTER_RESULTS.csv` (`experiment_family == "leave_one_encoder_out_ablation"`).
