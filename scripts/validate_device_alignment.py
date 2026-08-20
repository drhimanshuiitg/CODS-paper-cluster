#!/usr/bin/env python3
"""Read-only paired-device synchronization check using RMS-envelope cross-correlation."""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import signal
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.io import load_manifest_window, load_yaml

_CONFIG = None


def initialize(config):
    global _CONFIG
    _CONFIG = config


def envelope(audio: np.ndarray, sample_rate: int, frame_seconds: float = 0.1) -> np.ndarray:
    frame = int(round(frame_seconds * sample_rate))
    usable = len(audio) // frame * frame
    values = np.sqrt(np.mean(np.square(audio[:usable].reshape(-1, frame), dtype=np.float64), axis=1))
    values = np.log(values + 1e-8)
    values -= values.mean()
    scale = values.std()
    return values / scale if scale > 1e-12 else values


def validate_subject(info: dict) -> dict:
    if _CONFIG is None:
        raise RuntimeError("Worker not initialized")
    common = float(info["common_duration_sec"])
    length = min(300.0, common / 4)
    start = max(0.0, common / 2 - length / 2)
    end = start + length
    base = {"start_sec": start, "end_sec": end, "sample_id": f"alignment_{info['subject_id']}", "audio_segment_durations_json": ""}
    recorder = {
        **base, "audio_paths_json": info["recorder_paths_json"],
        "audio_segment_durations_json": info["recorder_durations_json"],
    }
    phone = {
        **base, "audio_paths_json": json.dumps([info["phone_path"]]),
        "audio_segment_durations_json": json.dumps([float(info["phone_duration_sec"])]),
    }
    recorder_audio, sample_rate = load_manifest_window(recorder, _CONFIG, "raw")
    phone_audio, phone_rate = load_manifest_window(phone, _CONFIG, "raw")
    if sample_rate != phone_rate:
        raise ValueError("Sample-rate mismatch")
    left = envelope(recorder_audio, sample_rate)
    right = envelope(phone_audio, sample_rate)
    correlation = signal.correlate(left, right, mode="full", method="fft") / len(left)
    lags = signal.correlation_lags(len(left), len(right), mode="full")
    max_lag_frames = int(round(120 / 0.1))
    mask = np.abs(lags) <= max_lag_frames
    best = np.flatnonzero(mask)[int(np.argmax(correlation[mask]))]
    return {
        "subject_id": info["subject_id"], "window_start_sec": start, "window_duration_sec": length,
        "best_lag_sec_recorder_relative_to_smartphone": float(lags[best] * 0.1),
        "best_envelope_correlation": float(correlation[best]),
        "zero_lag_envelope_correlation": float(correlation[np.flatnonzero(lags == 0)[0]]),
    }


def main() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    with (PROJECT_ROOT / "metadata" / "subject_inventory.csv").open(newline="", encoding="utf-8") as handle:
        subjects = [row for row in csv.DictReader(handle) if row["paired_available"] == "True"]
    with ProcessPoolExecutor(max_workers=8, initializer=initialize, initargs=(config,)) as executor:
        rows = list(tqdm(executor.map(validate_subject, subjects), total=len(subjects), desc="device alignment", unit="subject"))
    lags = np.asarray([row["best_lag_sec_recorder_relative_to_smartphone"] for row in rows])
    correlations = np.asarray([row["best_envelope_correlation"] for row in rows])
    summary = {
        "subjects": len(rows), "analysis_window_seconds_per_subject": 300,
        "lag_search_range_seconds": 120, "envelope_frame_seconds": 0.1,
        "median_best_lag_sec": float(np.median(lags)), "max_absolute_best_lag_sec": float(np.max(np.abs(lags))),
        "subjects_with_abs_lag_le_1_sec": int(np.sum(np.abs(lags) <= 1)),
        "subjects_with_abs_lag_le_5_sec": int(np.sum(np.abs(lags) <= 5)),
        "median_best_envelope_correlation": float(np.median(correlations)),
        "note": "Diagnostic only; no alignment transform is learned or applied.",
    }
    root = PROJECT_ROOT / "results" / "audit"
    csv_path = root / "device_alignment_validation.csv"
    json_path = root / "device_alignment_validation.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Refusing to overwrite device-alignment validation")
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
