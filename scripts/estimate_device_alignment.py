#!/usr/bin/env python3
"""Estimate subject-specific Recorder time mapping from paired unlabeled audio envelopes."""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import signal, stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.io import load_manifest_window, load_yaml

_CONFIG = None


def initialize(config):
    global _CONFIG
    _CONFIG = config


def rms_envelope(audio: np.ndarray, sample_rate: int, frame_seconds: float = 0.1) -> np.ndarray:
    frame = int(round(frame_seconds * sample_rate))
    usable = len(audio) // frame * frame
    values = np.sqrt(np.mean(np.square(audio[:usable].reshape(-1, frame), dtype=np.float64), axis=1))
    values = np.log(values + 1e-8)
    values -= values.mean()
    deviation = values.std()
    return values / deviation if deviation > 1e-12 else values


def row_for(info: dict, device: str, start: float, end: float) -> dict:
    if device == "R":
        paths = info["recorder_paths_json"]
        durations = info["recorder_durations_json"]
    else:
        paths = json.dumps([info["phone_path"]])
        durations = json.dumps([float(info["phone_duration_sec"])] )
    return {"sample_id": f"alignment_{info['subject_id']}_{device}_{start}", "start_sec": start, "end_sec": end,
            "audio_paths_json": paths, "audio_segment_durations_json": durations}


def anchor_lag(info: dict, center: float, window_seconds: float, max_lag_seconds: float) -> dict:
    if _CONFIG is None:
        raise RuntimeError("Worker not initialized")
    start = center - window_seconds / 2
    end = center + window_seconds / 2
    recorder, sample_rate = load_manifest_window(row_for(info, "R", start, end), _CONFIG, "raw")
    phone, _ = load_manifest_window(row_for(info, "S", start, end), _CONFIG, "raw")
    recorder_envelope = rms_envelope(recorder, sample_rate)
    phone_envelope = rms_envelope(phone, sample_rate)
    correlation = signal.correlate(recorder_envelope, phone_envelope, mode="full", method="fft") / len(recorder_envelope)
    lags = signal.correlation_lags(len(recorder_envelope), len(phone_envelope), mode="full")
    max_lag_frames = int(round(max_lag_seconds / 0.1))
    mask = np.abs(lags) <= max_lag_frames
    eligible = np.flatnonzero(mask)
    best = eligible[int(np.argmax(correlation[mask]))]
    return {"subject_id": info["subject_id"], "anchor_center_phone_sec": center,
            "recorder_time_minus_phone_time_sec": float(lags[best] * 0.1), "envelope_correlation": float(correlation[best])}


def estimate_subject(info: dict) -> tuple[list[dict], dict]:
    common = float(info["common_duration_sec"])
    window_seconds = min(900.0, common / 6)
    half = window_seconds / 2
    centers = np.linspace(half + 60, common - half - 60, 5)
    anchors = [anchor_lag(info, float(center), window_seconds, 300.0) for center in centers]
    center_values = np.asarray([row["anchor_center_phone_sec"] for row in anchors])
    lags = np.asarray([row["recorder_time_minus_phone_time_sec"] for row in anchors])
    correlations = np.asarray([row["envelope_correlation"] for row in anchors])
    reliable = correlations >= 0.45
    used = reliable if reliable.sum() >= 3 else np.ones(len(anchors), dtype=bool)
    slope, intercept, _, _ = stats.theilslopes(lags[used], center_values[used])
    predictions = intercept + slope * center_values
    residuals = lags - predictions
    model = {
        "subject_id": info["subject_id"], "recorder_time_mapping": "recorder_sec = phone_sec + intercept_sec + slope_sec_per_sec * phone_sec",
        "intercept_sec": float(intercept), "slope_sec_per_sec": float(slope), "clock_drift_seconds_per_hour": float(slope * 3600),
        "anchors": len(anchors), "reliable_anchors": int(reliable.sum()), "median_anchor_correlation": float(np.median(correlations)),
        "max_absolute_fit_residual_sec": float(np.max(np.abs(residuals[used]))),
        "low_confidence": bool(reliable.sum() < 3 or np.median(correlations) < 0.45 or np.max(np.abs(residuals[used])) > 2.0),
        "phone_duration_sec": float(info["phone_duration_sec"]), "recorder_duration_sec": float(info["recorder_duration_sec"]),
    }
    for row, predicted, residual, is_reliable in zip(anchors, predictions, residuals, reliable):
        row["fitted_lag_sec"] = float(predicted)
        row["residual_sec"] = float(residual)
        row["correlation_reliable"] = bool(is_reliable)
    return anchors, model


def main() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    with (PROJECT_ROOT / "metadata" / "subject_inventory.csv").open(newline="", encoding="utf-8") as handle:
        subjects = [row for row in csv.DictReader(handle) if row["paired_available"] == "True"]
    anchors = []
    models = []
    with ProcessPoolExecutor(max_workers=8, initializer=initialize, initargs=(config,)) as executor:
        for subject_anchors, model in tqdm(executor.map(estimate_subject, subjects), total=len(subjects), desc="alignment estimation", unit="subject"):
            anchors.extend(subject_anchors)
            models.append(model)
    root = PROJECT_ROOT / "metadata"
    anchor_path = root / "device_alignment_anchors.csv"
    model_path = root / "device_alignment_models.csv"
    validation_path = PROJECT_ROOT / "results" / "audit" / "device_alignment_model_validation.json"
    if anchor_path.exists() or model_path.exists() or validation_path.exists():
        raise FileExistsError("Refusing to overwrite alignment-model outputs")
    with anchor_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(anchors[0]))
        writer.writeheader(); writer.writerows(anchors)
    with model_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(models[0]))
        writer.writeheader(); writer.writerows(models)
    summary = {
        "subjects": len(models), "anchors_per_subject": 5, "alignment_uses_labels": False,
        "low_confidence_subjects": [row["subject_id"] for row in models if row["low_confidence"]],
        "median_absolute_intercept_sec": float(np.median(np.abs([row["intercept_sec"] for row in models]))),
        "max_absolute_intercept_sec": float(np.max(np.abs([row["intercept_sec"] for row in models]))),
        "median_absolute_clock_drift_sec_per_hour": float(np.median(np.abs([row["clock_drift_seconds_per_hour"] for row in models]))),
        "max_absolute_fit_residual_sec": float(np.max([row["max_absolute_fit_residual_sec"] for row in models])),
        "method": "Theil-Sen line through five 900-second paired RMS-envelope cross-correlation anchors; lag search +/-300 seconds",
    }
    validation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
