# Environment

- **Python**: 3.12.13
- **Framework**: PyTorch 2.9.0+cu128 (main pipeline, `evaluation.py`/`advanced.py`/`features.py`); TensorFlow 2.21.0 with `tensorflow[and-cuda]` extras (isolated venv, HeAR only)
- **Hardware**: single MIG 24GB GPU slice per SLURM job (`gpu:mig24gb:1`), 4-8 CPUs per job depending on `--workers` parallelism, IIT Guwahati GPU cluster (`gpu.iitg.ac.in`)
- **Key dependencies**:
  - `transformers` 4.45.2 (HuBERT/WavLM/Wav2Vec2/Data2Vec loading)
  - `scikit-learn` 1.5.2 (PCA, metrics, CPU-side classifier utilities)
  - `xgboost` 2.1.1 (`device="cuda"`)
  - `numpy` 1.26.4
  - `cuml-cu12` 26.8.0 (isolated venv at `/scratch/pkdas/IEEE_healthcomm_workshop/gpu_classifier_test/`, reached via subprocess bridge — kept separate because cuML requires newer numpy/scipy/scikit-learn than the main pipeline is pinned to)
  - `huggingface-hub` 0.26.2 (isolated HeAR venv only — pinned down from 1.28.0 after that version dropped `from_pretrained_keras`, a real dependency bug found and fixed this session)
- **Random seeds**: base seed 42 (`configs/base.yaml`), fold-perturbed per src/configs/training.md's "Random seed" entry; bootstrap significance tests use a separate, also-fixed seed (42) for their own resampling RNG, independent of the model-fitting seed
- **Isolated-venv architecture** (3 separate Python environments, each with a conflicting dependency stack from the main pipeline, all reached via `subprocess.run()` bridges from the main-pipeline process):
  1. Main pipeline venv (`/home/pkdas/IEEE_healthcomm_workshop/.venv`) — torch/transformers/sklearn/xgboost, does all orchestration, data loading, and classifier dispatch
  2. cuML venv (`/scratch/pkdas/IEEE_healthcomm_workshop/gpu_classifier_test/`) — `random_forest`/`svm_rbf` GPU fits/scores only
  3. HeAR venv (`/scratch/pkdas/IEEE_healthcomm_workshop/hear_extractor/`) — HeAR embedding extraction only
- **GPU-only enforcement**: every subprocess bridge and every in-process GPU-accelerated classifier carries an explicit hard-fail check (`torch.cuda.is_available()`, `tf.config.list_physical_devices("GPU")`, or cuML's native `CUDARuntimeError` with no CPU code path) — no silent CPU fallback anywhere in the current pipeline, a project-wide policy adopted and audited this session after a real incident (see logic/solution/heuristics.md H03).
