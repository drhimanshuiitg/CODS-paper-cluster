# Autonomous execution and recovery log

All commands below were run under SLURM in accordance with `GPU_INSTRUCTIONS.md`.
Raw files under `/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/Data` were
opened read-only; no dataset file was modified or copied into the project.

## Feature-extraction recovery

- Job 1421 (`sq_ssl_all`) stopped before extracting an SSL feature because
  `AutoProcessor` attempted to load a tokenizer that the base HuBERT checkpoint
  does not provide. The waveform checkpoints now use `AutoFeatureExtractor`;
  the vision checkpoint continues to use `AutoImageProcessor`.
- Job 1424 resumed the same HuBERT cache and completed rows 0--211, then stopped
  on a final one-sample audio chunk. Long events are still processed without
  truncation, but a terminal chunk shorter than 400 samples is now merged into
  its preceding chunk (or padded when it is the only chunk).
- Job 1425 is the second resumable retry. It restarted at the first incomplete
  bitmap row, passed the former failing row, completed and validated both HuBERT
  and WavLM, and preserved all existing finite feature rows. It was intentionally
  cancelled after WavLM completed because its next step was safely waiting on the
  cache lock held by the parallel Wav2Vec2 worker; the remaining independent
  Data2Vec-audio branch was then launched in the released slot. This cancellation
  is a scheduling optimization, not a failed feature result.

The failed attempts and their original logs are retained in `logs/`; the old
failure record is retained in `cached_features/hubert/peak/failures.jsonl` for
traceability. A cache is accepted for evaluation only when every completion
bitmap entry is true and its metadata status is `complete`.

## Environment recovery

- The initial PyTorch CUDA build did not contain code for the allocated
  Blackwell GPU. Job 1417 upgraded the project environment to a CUDA 12.8 build
  and verified a real CUDA tensor operation on the allocated MIG device. The
  machine-readable check is in
  `environment/blackwell_validation_slurm_1417.json`.
