#!/usr/bin/env python3
"""Extract HeAR (google/hear) embeddings into the same resumable feature-cache
format extract_feature_cache() uses for every other representation, so
load_representation()/run_main_benchmark.py pick "hear" up transparently.

HeAR only runs in an isolated TF-Keras venv (see hear_extractor/), reached
via a subprocess bridge -- mirrors gpu_classifier_test/cuml_worker.py's
pattern, for the same reason (conflicting dependency stack). This script
does the audio I/O (in the live pipeline's own torch venv, via the same
load_manifest_window() every other feature uses) and batches many rows into
one subprocess call so the ~4-minute one-time model load is amortized over
thousands of clips instead of paid once per row.

Fixed-length input: HeAR only accepts exactly 2.0s/16kHz/32000-sample raw
audio -- there is no variable-length or pooling mode in the model itself.
Manifest windows vary in length (event windows, matched negatives), so each
window is reduced to one 2.0s clip: the center 32000 samples if the window
is longer, zero-padded (centered) if shorter. This is a real, documented
simplification (see --help), not a hidden approximation -- flagged in the
cache metadata as clip_policy="center_crop_or_pad".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.features import FEATURE_DIMENSIONS, _open_cache, cache_paths  # noqa: E402
from sleep_quadnet.io import config_hash, file_sha256, load_manifest_window, load_yaml, read_csv_rows  # noqa: E402

CLIP_SAMPLES = 32000  # HeAR's fixed native input: 2.0s @ 16kHz
HEAR_VENV_PYTHON = "/userhome/phd/h.sharma/CODS-paper/hear_extractor/bin/python3"
HEAR_WORKER = "/userhome/phd/h.sharma/CODS-paper/hear_extractor/hear_worker.py"
HF_HOME = "/userhome/phd/h.sharma/CODS-paper/cache/huggingface"


def to_fixed_clip(audio: np.ndarray) -> np.ndarray:
    n = audio.shape[0]
    if n == CLIP_SAMPLES:
        return audio
    if n > CLIP_SAMPLES:
        start = (n - CLIP_SAMPLES) // 2
        return audio[start : start + CLIP_SAMPLES]
    pad_total = CLIP_SAMPLES - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio, (pad_left, pad_right), mode="constant")


def run_hear_batch(clips: np.ndarray, work_dir: Path) -> np.ndarray:
    clips_path = work_dir / "clips.npz"
    out_path = work_dir / "embeddings.npz"
    np.savez(clips_path, X=clips.astype(np.float32))
    # Bug fix (2026-08-19): this previously REPLACED the subprocess environment
    # entirely with a minimal {HF_HOME, PATH} dict, which wiped out
    # CUDA_VISIBLE_DEVICES and the NVIDIA driver library paths the SLURM
    # allocation injects into the parent's environment -- silently forcing
    # TensorFlow onto CPU inside the job even though a MIG GPU slice was
    # allocated and reserved (confirmed live: process at 377% CPU, absent
    # from nvidia-smi --query-compute-apps entirely, after 5+ minutes).
    # Inherit the full parent environment and only add HF_HOME on top.
    env = {**os.environ, "HF_HOME": HF_HOME}
    result = subprocess.run(
        [HEAR_VENV_PYTHON, HEAR_WORKER, "embed", str(clips_path), str(out_path)],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"hear_worker.py failed:\nstdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}")
    log_gpu_detection(result.stderr)
    return np.load(out_path)["embeddings"]


def log_gpu_detection(stderr_text: str) -> None:
    """Surface TF's own GPU-detection lines so a silent CPU fallback (e.g. a
    stripped subprocess environment losing CUDA_VISIBLE_DEVICES) is visible
    in the job log immediately, not only discoverable by manually checking
    nvidia-smi mid-run."""
    for line in stderr_text.splitlines():
        lowered = line.lower()
        if "cuda" in lowered or "gpu" in lowered:
            print(f"[hear gpu-detection] {line}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "cached_features")
    parser.add_argument("--preprocessing", default="peak")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per subprocess call -- amortizes the ~4min model load.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["project_root"] = str(PROJECT_ROOT)
    rows = read_csv_rows(args.manifest)
    signature = config_hash(config, "hear", args.preprocessing, file_sha256(args.manifest), "clip_policy=center_crop_or_pad")
    paths = cache_paths(args.cache_root, "hear", args.preprocessing)
    feature_array, complete = _open_cache(paths, rows, "hear", args.preprocessing, signature)
    pending = np.flatnonzero(~np.asarray(complete))

    with tempfile.TemporaryDirectory(dir=args.cache_root) as tmp:
        work_dir = Path(tmp)
        progress = tqdm(total=len(pending), desc="extract hear", unit="clip", dynamic_ncols=True)
        for batch_start in range(0, len(pending), args.batch_size):
            batch_indices = pending[batch_start : batch_start + args.batch_size]
            clips = np.empty((len(batch_indices), CLIP_SAMPLES), dtype=np.float32)
            for position, index in enumerate(batch_indices):
                audio, sample_rate = load_manifest_window(rows[int(index)], config, args.preprocessing)
                if sample_rate != 16000:
                    raise ValueError(f"HeAR requires 16kHz audio, got {sample_rate}")
                clips[position] = to_fixed_clip(audio)
            embeddings = run_hear_batch(clips, work_dir)
            if embeddings.shape != (len(batch_indices), 512):
                raise ValueError(f"Unexpected HeAR output shape {embeddings.shape}")
            if not np.isfinite(embeddings).all():
                raise ValueError("Non-finite HeAR embeddings in batch")
            feature_array[batch_indices] = embeddings
            complete[batch_indices] = True
            feature_array.flush()
            complete.flush()
            progress.update(len(batch_indices))
            progress.set_postfix(completed=int(np.asarray(complete).sum()), refresh=False)
        progress.close()

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata.update({
        "status": "complete" if bool(np.asarray(complete).all()) else "partial",
        "completed_rows": int(np.asarray(complete).sum()),
        "model_id": "google/hear",
        "clip_policy": "center_crop_or_pad",
        "clip_samples": CLIP_SAMPLES,
    })
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metadata["status"] != "complete":
        raise RuntimeError(f"HeAR feature cache incomplete: {paths['root']}")
    print(json.dumps({"status": "complete", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
