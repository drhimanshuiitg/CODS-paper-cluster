#!/usr/bin/env python3
"""Dense, streaming paired-device lag map for corrected window materialization."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_SECONDS = 0.1
WINDOW_SECONDS = 900.0
STEP_SECONDS = 600.0
MAX_LAG_SECONDS = 300.0
RELIABLE_CORRELATION = 0.45


def streaming_rms(paths: list[str], sample_rate: int = 8000) -> np.ndarray:
    frame = int(round(FRAME_SECONDS * sample_rate))
    values = []
    remainder = np.empty(0, dtype=np.float32)
    for path in paths:
        with sf.SoundFile(path, "r") as handle:
            if handle.samplerate != sample_rate or handle.channels != 1:
                raise ValueError(f"Unexpected WAV format: {path}")
            while True:
                block = handle.read(frame * 4096, dtype="float32", always_2d=False)
                if not len(block):
                    break
                if len(remainder):
                    block = np.concatenate([remainder, block])
                count = len(block) // frame
                if count:
                    framed = block[: count * frame].reshape(count, frame)
                    values.append(np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1)))
                remainder = block[count * frame :]
    if len(remainder):
        values.append(np.asarray([np.sqrt(np.mean(np.square(remainder, dtype=np.float64)))]))
    return np.log(np.concatenate(values) + 1e-8)


def local_lag(recorder: np.ndarray, phone: np.ndarray, center_frame: int) -> tuple[float, float, bool]:
    half = int(round(WINDOW_SECONDS / FRAME_SECONDS / 2))
    left = recorder[center_frame - half : center_frame + half].copy()
    right = phone[center_frame - half : center_frame + half].copy()
    left = (left - left.mean()) / max(left.std(), 1e-12)
    right = (right - right.mean()) / max(right.std(), 1e-12)
    correlation = signal.correlate(left, right, mode="full", method="fft") / len(left)
    lags = signal.correlation_lags(len(left), len(right), mode="full")
    max_frames = int(round(MAX_LAG_SECONDS / FRAME_SECONDS))
    eligible = np.flatnonzero(np.abs(lags) <= max_frames)
    best = eligible[int(np.argmax(correlation[eligible]))]
    lag_seconds = float(lags[best] * FRAME_SECONDS)
    at_boundary = abs(lag_seconds) >= MAX_LAG_SECONDS - FRAME_SECONDS
    return lag_seconds, float(correlation[best]), at_boundary


def estimate(info: dict) -> tuple[list[dict], dict]:
    recorder_paths = json.loads(info["recorder_paths_json"])
    recorder = streaming_rms(recorder_paths)
    phone = streaming_rms([info["phone_path"]])
    common_frames = min(len(recorder), len(phone))
    half = int(round(WINDOW_SECONDS / FRAME_SECONDS / 2))
    step = int(round(STEP_SECONDS / FRAME_SECONDS))
    centers = list(range(half, common_frames - half + 1, step))
    if centers[-1] < common_frames - half:
        centers.append(common_frames - half)
    anchors = []
    for center in centers:
        lag, correlation, at_boundary = local_lag(recorder, phone, center)
        anchors.append(
            {"subject_id": info["subject_id"], "anchor_phone_sec": center * FRAME_SECONDS,
             "recorder_time_minus_phone_time_sec": lag, "envelope_correlation": correlation,
             "peak_at_search_boundary": at_boundary,
             "reliable": correlation >= RELIABLE_CORRELATION and not at_boundary}
        )
    reliable = [row for row in anchors if row["reliable"]]
    reliable_times = np.asarray([row["anchor_phone_sec"] for row in reliable])
    gaps = np.diff(reliable_times) if len(reliable_times) > 1 else np.asarray([np.inf])
    fraction = len(reliable) / len(anchors)
    median_correlation = float(np.median([row["envelope_correlation"] for row in reliable])) if reliable else 0.0
    usable = len(reliable) >= 3 and fraction >= 0.7 and median_correlation >= 0.5 and float(np.max(gaps)) <= 1800.0
    model = {
        "subject_id": info["subject_id"], "anchors": len(anchors), "reliable_anchors": len(reliable),
        "reliable_fraction": fraction, "median_reliable_correlation": median_correlation,
        "max_gap_between_reliable_anchors_sec": float(np.max(gaps)), "alignment_usable": usable,
        "phone_envelope_duration_sec": len(phone) * FRAME_SECONDS,
        "recorder_envelope_duration_sec": len(recorder) * FRAME_SECONDS,
        "mapping": "piecewise-linear interpolation of reliable recorder-minus-phone lag anchors",
    }
    return anchors, model


def main() -> None:
    with (PROJECT_ROOT / "metadata" / "subject_inventory.csv").open(newline="", encoding="utf-8") as handle:
        subjects = [row for row in csv.DictReader(handle) if row["paired_available"] == "True"]
    anchors = []
    models = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for subject_anchors, model in tqdm(executor.map(estimate, subjects), total=len(subjects), desc="dense alignment", unit="subject", dynamic_ncols=True):
            anchors.extend(subject_anchors); models.append(model)
    anchor_path = PROJECT_ROOT / "metadata" / "device_alignment_dense_anchors.csv"
    model_path = PROJECT_ROOT / "metadata" / "device_alignment_dense_summary.csv"
    validation_path = PROJECT_ROOT / "results" / "audit" / "device_alignment_dense_validation.json"
    if anchor_path.exists() or model_path.exists() or validation_path.exists():
        raise FileExistsError("Refusing to overwrite dense alignment outputs")
    with anchor_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(anchors[0])); writer.writeheader(); writer.writerows(anchors)
    with model_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(models[0])); writer.writeheader(); writer.writerows(models)
    summary = {
        "subjects": len(models), "usable_subjects": [row["subject_id"] for row in models if row["alignment_usable"]],
        "excluded_alignment_subjects": [row["subject_id"] for row in models if not row["alignment_usable"]],
        "frame_seconds": FRAME_SECONDS, "window_seconds": WINDOW_SECONDS, "anchor_step_seconds": STEP_SECONDS,
        "lag_search_seconds": MAX_LAG_SECONDS, "reliability_correlation_threshold": RELIABLE_CORRELATION,
        "uses_annotations_or_labels": False,
    }
    validation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
