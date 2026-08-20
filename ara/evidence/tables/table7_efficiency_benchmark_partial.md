# Table 7: Deployment/efficiency benchmark, extended coverage (8/14 representations)

**Source**: results/P0_efficiency/component_runs/*.json

**Caption**: Warm-GPU, batch-size-1, single-clip latency and peak memory per feature-extraction stage. Extended 2026-08-19 (post-review, F04-adjacent) to add wavlm_large and hear -- hear's peak_gpu_memory_bytes is reported as 0 not because it uses no GPU memory, but because it runs in a separate isolated-venv subprocess whose GPU memory this process's torch.cuda accounting cannot see (observed ~630MB via nvidia-smi in job logs instead). Still missing: full_fusion, full_fusion_v2, full_fusion_plus_hear, data2vec_fusion -- these are concatenations of already-benchmarked component features with no separate model to time; their latency is the sum of their components' latencies, not independently measured here.

**Extraction type**: raw_table

| feature | model_id | latency_mean_sec | latency_std_sec | peak_gpu_memory_mb | real_time_factor_mean | clips |
|---|---|---|---|---|---|---|
| classical | handcrafted | 0.072261 | 0.000456 | 0.000000 | 0.003613 | 20 |
| data2vec_audio | facebook/data2vec-audio-base-960h | 0.032317 | 0.000271 | 846.604288 | 0.001616 | 20 |
| data2vec_spectrogram | facebook/data2vec-vision-base | 0.038302 | 0.000552 | 554.060800 | 0.001915 | 20 |
| hear | google/hear | 0.022278 | 0.000429 | 0.000000 | 0.011139 | 20 |
| hubert | facebook/hubert-base-ls960 | 0.025934 | 0.000280 | 722.866176 | 0.001297 | 20 |
| wav2vec2 | facebook/wav2vec2-base | 0.025793 | 0.000170 | 722.866176 | 0.001290 | 20 |
| wavlm | microsoft/wavlm-base | 0.048976 | 0.000316 | 787.701248 | 0.002449 | 20 |
| wavlm_large | microsoft/wavlm-large | 0.097941 | 0.003065 | 2152.555008 | 0.004897 | 20 |
