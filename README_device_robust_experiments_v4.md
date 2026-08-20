# Device-robust sleep-apnea benchmark runner v4

This version keeps the experiment lightweight and close to the original comparison.

## Representations

1. `classical` — handcrafted acoustic baseline.
2. `wav2vec2` — Wav2Vec2 embedding.
3. `hubert` — HuBERT embedding.
4. `wavlm` — WavLM embedding.
5. `data2vec` — one Data2Vec representation: raw waveform Data2VecAudio + Mel-spectrogram Data2VecVision concatenated internally.
6. `ssl_fusion` — Wav2Vec2 + HuBERT + WavLM + Data2Vec.

No automatic pair/triple/quad fusion grid is generated.

## Run

```bash
python device_robust_sleep_apnea_experiments_v4.py --config config_device_robust_v4.yaml
```

## Main outputs

- `final_comparison_all_results.csv`
- `final_overall_comparison_sorted.csv`
- `best_model_per_protocol.csv`
- `representation_manifest.csv`
- `latex_tables/journal_results_tables.tex`
- trained models under `models/`
- predictions under `predictions/`
- ROC/PR/confusion/calibration plots under `figures/`

