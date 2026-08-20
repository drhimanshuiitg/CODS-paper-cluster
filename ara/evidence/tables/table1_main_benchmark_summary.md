# Table 1: Main cross-device benchmark, per-representation summary

**Source**: results/P0_device_gap (1,120 completed fold-runs, aggregated)

**Caption**: Mean balanced accuracy / F1 / ROC-AUC / MCC per representation, split by matched-device (R_R,S_S) vs cross-device (R_S,S_R) regime, averaged over all 4 classifiers and applicable folds/protocols.

**Extraction type**: raw_table

| representation | regime | balanced_accuracy | f1 | roc_auc | mcc |
|---|---|---|---|---|---|
| classical | cross | 0.4988 | 0.0514 | 0.5337 | -0.0064 |
| classical | matched | 0.5509 | 0.4783 | 0.5838 | 0.1069 |
| data2vec_audio | cross | 0.5254 | 0.3840 | 0.5533 | 0.0569 |
| data2vec_audio | matched | 0.5663 | 0.5525 | 0.6017 | 0.1341 |
| data2vec_fusion | cross | 0.5347 | 0.3519 | 0.5836 | 0.0839 |
| data2vec_fusion | matched | 0.5827 | 0.5620 | 0.6230 | 0.1669 |
| data2vec_spectrogram | cross | 0.5261 | 0.3077 | 0.5734 | 0.0679 |
| data2vec_spectrogram | matched | 0.5772 | 0.5461 | 0.6184 | 0.1571 |
| full_fusion | cross | 0.5311 | 0.3125 | 0.5877 | 0.0778 |
| full_fusion | matched | 0.5843 | 0.5567 | 0.6318 | 0.1709 |
| full_fusion_plus_hear | cross | 0.5288 | 0.2769 | 0.5910 | 0.0769 |
| full_fusion_plus_hear | matched | 0.5864 | 0.5613 | 0.6341 | 0.1750 |
| full_fusion_v2 | cross | 0.5357 | 0.3355 | 0.5944 | 0.0886 |
| full_fusion_v2 | matched | 0.5797 | 0.5542 | 0.6258 | 0.1613 |
| hear | cross | 0.5040 | 0.1936 | 0.5238 | 0.0117 |
| hear | matched | 0.5788 | 0.5920 | 0.6121 | 0.1599 |
| hubert | cross | 0.5466 | 0.4034 | 0.5931 | 0.1045 |
| hubert | matched | 0.5841 | 0.5610 | 0.6306 | 0.1705 |
| hubert_odi_hb | cross | 0.5473 | 0.4079 | 0.5903 | 0.1069 |
| hubert_odi_hb | matched | 0.5855 | 0.5655 | 0.6313 | 0.1732 |
| odi_hb | cross | 0.4958 | 0.4622 | 0.4949 | -0.0095 |
| odi_hb | matched | 0.4958 | 0.4622 | 0.4949 | -0.0095 |
| wav2vec2 | cross | 0.5225 | 0.3409 | 0.5517 | 0.0527 |
| wav2vec2 | matched | 0.5545 | 0.5310 | 0.5826 | 0.1101 |
| wavlm | cross | 0.5335 | 0.4397 | 0.5705 | 0.0740 |
| wavlm | matched | 0.5671 | 0.5500 | 0.6055 | 0.1355 |
| wavlm_large | cross | 0.5406 | 0.3416 | 0.5920 | 0.1011 |
| wavlm_large | matched | 0.6114 | 0.6200 | 0.6590 | 0.2241 |
