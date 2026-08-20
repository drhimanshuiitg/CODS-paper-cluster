#!/usr/bin/env python3
"""Warm-up controlled component latency and memory benchmark for P0-D."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from tqdm import tqdm
import subprocess
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.features import (
    FEATURE_DIMENSIONS,
    MODEL_SPECS,
    NO_GPU_NEEDED,
    _audio_ssl_vector,
    _load_model,
    classical_features,
    mel_image,
)
from sleep_quadnet.io import load_manifest_window, load_yaml, read_csv_rows


HEAR_VENV_PYTHON = "/userhome/phd/h.sharma/CODS-paper/hear_extractor/bin/python3"
HEAR_WORKER = "/userhome/phd/h.sharma/CODS-paper/hear_extractor/hear_worker.py"
HF_HOME = "/userhome/phd/h.sharma/CODS-paper/cache/huggingface"
HEAR_CLIP_SAMPLES = 32000  # 2.0s @ 16kHz, HeAR's fixed native input


def to_hear_clip(audio: np.ndarray) -> np.ndarray:
    """Same center-crop-or-zero-pad policy as extract_hear_features.py::to_fixed_clip."""
    n = audio.shape[0]
    if n == HEAR_CLIP_SAMPLES:
        return audio
    if n > HEAR_CLIP_SAMPLES:
        start = (n - HEAR_CLIP_SAMPLES) // 2
        return audio[start : start + HEAR_CLIP_SAMPLES]
    pad_total = HEAR_CLIP_SAMPLES - n
    pad_left = pad_total // 2
    return np.pad(audio, (pad_left, pad_total - pad_left), mode="constant")


def benchmark_hear(rows: list[dict], config: dict, warmup: int) -> dict:
    """HeAR has no in-process transformers model to load here (isolated
    TF-Keras venv, subprocess bridge -- see extract_hear_features.py); this
    mirrors that same bridge instead of going through run_once()/_load_model()."""
    clips = np.empty((len(rows), HEAR_CLIP_SAMPLES), dtype=np.float32)
    for i, row in enumerate(rows):
        audio, sample_rate = load_manifest_window(row, config, "peak")
        if sample_rate != 16000:
            raise ValueError(f"HeAR requires 16kHz audio, got {sample_rate}")
        clips[i] = to_hear_clip(audio)

    with tempfile.TemporaryDirectory() as tmp:
        clips_path = Path(tmp) / "clips.npz"
        out_path = Path(tmp) / "timings.json"
        np.savez(clips_path, X=clips)
        env = {**os.environ, "HF_HOME": HF_HOME}  # inherit, never replace -- see GPU_INSTRUCTIONS.md
        result = subprocess.run(
            [HEAR_VENV_PYTHON, HEAR_WORKER, "bench", str(clips_path), str(out_path)],
            env=env, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"hear_worker.py bench failed:\nstdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}")
        payload = json.loads(out_path.read_text())
    return payload


def representative_rows(rows: list[dict], count: int) -> list[dict]:
    strata: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        strata.setdefault((row["device"], row["label"]), []).append(row)
    selected = []
    per_stratum = max(1, count // len(strata))
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda row: (abs(float(row["duration_sec"]) - 20.0), row["sample_id"]))
        selected.extend(candidates[:per_stratum])
    return selected[:count]


def run_once(row: dict, feature: str, config: dict, processor, model, input_kind: str, device: torch.device):
    preprocessing = "peak_filter" if feature == "classical" else "peak"
    audio, sample_rate = load_manifest_window(row, config, preprocessing)
    if feature == "classical":
        return classical_features(audio, sample_rate)
    if input_kind == "audio":
        return _audio_ssl_vector(audio, sample_rate, processor, model, device, float(config["audio"]["ssl_max_chunk_seconds"]))
    image = mel_image(audio, sample_rate, config)
    inputs = processor(images=[image], return_tensors="pt")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model(inputs.pixel_values.to(device))
    return output.last_hidden_state.mean(dim=1)[0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    # odi_hb excluded: it's a static per-subject CSV lookup (see
    # features.py:_load_odi_hb_lookup), not audio-derived -- run_once() below
    # has no branch for it (nor should it grow one just for this benchmark;
    # "per-clip extraction latency" isn't a meaningful question for an O(1)
    # dict lookup). classical is the only other NO_GPU_NEEDED feature and IS
    # supported below, since it does real per-clip DSP work worth timing.
    parser.add_argument("--feature", choices=sorted(FEATURE_DIMENSIONS.keys() - {"odi_hb"}), required=True)
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    config = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    rows = representative_rows(read_csv_rows(PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv"), args.clips)

    if args.feature == "hear":
        # HeAR: isolated TF-Keras venv, subprocess bridge -- no in-process
        # torch model to load or time here (see benchmark_hear() docstring).
        # GPU-presence is enforced inside hear_worker.py itself, not here.
        process = psutil.Process()
        rss_before = process.memory_info().rss
        payload = benchmark_hear(rows, config, args.warmup)
        rss_after = process.memory_info().rss
        timings = payload["timings_sec"]
        audio_durations = [2.0] * len(timings)  # HeAR's fixed native clip length, not the source window's true duration
        record = {
            "feature": "hear", "model_id": "google/hear", "feature_dimension": FEATURE_DIMENSIONS["hear"],
            "clips": len(rows), "warmup_clips": payload["warmup_clips"], "batch_size": 1,
            "latency_mean_sec": statistics.mean(timings), "latency_std_sec": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "clips_per_second": len(timings) / sum(timings),
            "real_time_factor_mean": statistics.mean(t / d for t, d in zip(timings, audio_durations)),
            "peak_gpu_memory_bytes": 0,  # not measured here -- GPU memory lives in the separate hear_extractor venv process, not this one; see results/audit or job logs for observed VRAM (~630MB) instead
            "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "cpu_rss_before_bytes": rss_before, "cpu_rss_after_bytes": rss_after, "cpu_rss_delta_bytes": rss_after - rss_before,
            "gpu": None, "torch": torch.__version__, "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
            "timings_sec": timings, "audio_durations_sec": audio_durations,
            "note": "HeAR clips are fixed 2.0s (center-crop-or-pad, see extract_hear_features.py) regardless of source window duration -- real_time_factor is computed against that fixed length, not the manifest window's true duration, unlike every other feature in this benchmark.",
        }
        root = PROJECT_ROOT / "results" / "P0_efficiency" / "component_runs"
        root.mkdir(parents=True, exist_ok=True)
        target = root / "hear.json"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark: {target}")
        target.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))
        return

    device = torch.device("cuda" if args.feature not in NO_GPU_NEEDED else "cpu")
    if args.feature not in NO_GPU_NEEDED and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if args.feature == "classical":
        model_id, processor, model, kind = "handcrafted", None, None, "audio"
    else:
        model_id, processor, model, kind = _load_model(args.feature, device, local_files_only=True)
    for row in rows[: args.warmup]:
        run_once(row, args.feature, config, processor, model, kind, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    timings = []
    audio_durations = []
    for row in tqdm(rows, desc=f"benchmark {args.feature}", unit="clip"):
        start = time.perf_counter()
        vector = run_once(row, args.feature, config, processor, model, kind, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        audio_durations.append(float(row["duration_sec"]))
        if np.asarray(vector).shape != (FEATURE_DIMENSIONS[args.feature],):
            raise ValueError("Unexpected benchmark feature shape")
    rss_after = process.memory_info().rss
    record = {
        "feature": args.feature, "model_id": model_id, "feature_dimension": FEATURE_DIMENSIONS[args.feature],
        "clips": len(rows), "warmup_clips": min(args.warmup, len(rows)), "batch_size": 1,
        "latency_mean_sec": statistics.mean(timings), "latency_std_sec": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "clips_per_second": len(timings) / sum(timings),
        "real_time_factor_mean": statistics.mean(time_value / duration for time_value, duration in zip(timings, audio_durations)),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "cpu_rss_before_bytes": rss_before, "cpu_rss_after_bytes": rss_after, "cpu_rss_delta_bytes": rss_after - rss_before,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "timings_sec": timings, "audio_durations_sec": audio_durations,
    }
    root = PROJECT_ROOT / "results" / "P0_efficiency" / "component_runs"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{args.feature}.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {target}")
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
