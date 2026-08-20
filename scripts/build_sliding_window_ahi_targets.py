#!/usr/bin/env python3
"""Build clinically-real, whole-night sliding-window severity targets from
raw SpO2, as ground truth for a future non-annotation-privileged prediction
task (predict local desaturation burden directly from continuous audio,
rather than classifying pre-selected annotated-event windows).

This deliberately does NOT reuse the manifest's annotation-privileged
windows (dataset_manifest_aligned.csv): those windows are centered on
PSG-annotated events plus matched negatives, which is fine for the
event-classification task but would leak the very thing a "does this
stretch of the night look like moderate/severe OSA" model should be
learning to detect on its own. Instead this bins the FULL night into fixed,
non-overlapping epochs on a clock grid, independent of where any event
happens to fall, using exactly the same desaturation-detection convention
already validated in compute_odi_hypoxic_burden.py (r=0.83 vs PSG-annotated
event counts).

Per-epoch target: desaturation-event count and hypoxic-burden area
occurring inside that epoch (an event is attributed to the epoch containing
its start), scaled to an hourly rate (epoch_ahi_proxy) so epochs are
comparable regardless of epoch length, plus the standard AASM-style
severity bin computed from that rate. Awake epochs are flagged, not
dropped -- excluded_from_training makes the awake-exclusion policy an
explicit, auditable column instead of a silent drop, so a downstream
consumer can choose to include them as an explicit "awake/no-event-expected"
class if that turns out to be useful.

Output: metadata/sliding_window_ahi_targets.csv, one row per (subject,
epoch): subject_id, epoch_start_sec, epoch_end_sec, epoch_duration_sec,
desat_count, hypoxic_burden_area_pctmin, epoch_ahi_proxy, severity_bin,
awake_fraction, excluded_from_training.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPOCH_SEC = 300.0  # 5-minute epochs: long enough for a stable local event rate, short enough for real localization


def str2seconds(text: str) -> float:
    hours_text, minutes_text, seconds_text = text.split(":")
    hours = int(hours_text)
    minutes = int(minutes_text)
    seconds = float(seconds_text) if "." in seconds_text else int(seconds_text)
    if hours < 12:
        hours += 24
    return hours * 3600 + minutes * 60 + seconds


def load_spo2(path: Path, record_start: float) -> tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if row[2] == "-":
                continue
            times.append(str2seconds(row[1]) - record_start)
            values.append(float(row[2]))
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(values)[order]


def mask_awake(times: np.ndarray, awake_intervals: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, end in awake_intervals:
        mask |= (times >= start) & (times <= end)
    return mask


def detect_desaturations(
    times: np.ndarray, values: np.ndarray, awake_mask: np.ndarray,
    drop_pct: float = 3.0, baseline_window_sec: float = 100.0, min_duration_sec: float = 8.0,
) -> list[tuple[float, float, float]]:
    """Returns (start_sec, end_sec, area_pctsec) per qualifying desaturation
    event -- same convention as compute_odi_hypoxic_burden.py, extended to
    keep event timing (not just area) so events can be binned into epochs."""
    n = len(times)
    if n < 10:
        return []
    baseline = np.full(n, np.nan)
    window: deque[tuple[float, float]] = deque()
    for i in range(n):
        window.append((times[i], values[i]))
        while window[0][0] < times[i] - baseline_window_sec:
            window.popleft()
        baseline[i] = max(value for _, value in window)
    below_threshold = (baseline - values >= drop_pct) & (~awake_mask)
    events = []
    i = 0
    while i < n:
        if below_threshold[i]:
            j = i
            while j < n and below_threshold[j]:
                j += 1
            duration = times[j - 1] - times[i]
            if duration >= min_duration_sec:
                area = float(np.trapz(np.clip(baseline[i:j] - values[i:j], 0, None), times[i:j]))
                events.append((float(times[i]), float(times[j - 1]), area))
            i = j
        else:
            i += 1
    return events


def severity_bin(ahi_proxy: float) -> str:
    # Standard AASM AHI severity thresholds, applied to the epoch's
    # extrapolated hourly rate (not a real whole-night AHI -- "proxy" is
    # kept in the column name deliberately).
    if ahi_proxy < 5:
        return "normal"
    if ahi_proxy < 15:
        return "mild"
    if ahi_proxy < 30:
        return "moderate"
    return "severe"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("/userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "metadata" / "sliding_window_ahi_targets.csv")
    parser.add_argument("--epoch-sec", type=float, default=EPOCH_SEC)
    args = parser.parse_args()

    rows = []
    subjects_written = 0
    for spo2_path in sorted(args.dataset_dir.glob("*/*_SpO2.csv")):
        subject_id = spo2_path.parent.name
        annotation_path = spo2_path.parent / f"{subject_id}_annotation.json"
        if not annotation_path.exists():
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        record_start = float(annotation["record_start"])
        awake_relative = [(start - record_start, end - record_start) for start, end in annotation.get("awake_intervals", [])]

        times, values = load_spo2(spo2_path, record_start)
        if len(times) < 10:
            continue
        awake_mask = mask_awake(times, awake_relative)
        events = detect_desaturations(times, values, awake_mask)

        night_end = float(times[-1])
        n_epochs = int(np.ceil(night_end / args.epoch_sec))
        epoch_desat_count = np.zeros(n_epochs, dtype=int)
        epoch_area = np.zeros(n_epochs, dtype=float)
        for start, _end, area in events:
            idx = min(int(start // args.epoch_sec), n_epochs - 1)
            epoch_desat_count[idx] += 1
            epoch_area[idx] += area

        # Awake fraction per epoch, from the SpO2-sample-level awake mask
        # (same signal used to exclude awake time from event detection),
        # binned onto the same epoch grid via the sample timestamps.
        sample_epoch = np.minimum((times // args.epoch_sec).astype(int), n_epochs - 1)
        awake_fraction = np.zeros(n_epochs, dtype=float)
        counts = np.zeros(n_epochs, dtype=int)
        np.add.at(awake_fraction, sample_epoch, awake_mask.astype(float))
        np.add.at(counts, sample_epoch, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            awake_fraction = np.where(counts > 0, awake_fraction / np.maximum(counts, 1), 1.0)

        for e in range(n_epochs):
            epoch_start = e * args.epoch_sec
            epoch_end = min((e + 1) * args.epoch_sec, night_end)
            duration = epoch_end - epoch_start
            if duration <= 0:
                continue
            hourly_scale = 3600.0 / duration
            ahi_proxy = epoch_desat_count[e] * hourly_scale
            hb_proxy = (epoch_area[e] / 60.0) * hourly_scale  # %*sec -> %*min -> per hour
            rows.append({
                "subject_id": subject_id, "epoch_start_sec": round(epoch_start, 1), "epoch_end_sec": round(epoch_end, 1),
                "epoch_duration_sec": round(duration, 1), "desat_count": int(epoch_desat_count[e]),
                "hypoxic_burden_area_pctmin": round(epoch_area[e] / 60.0, 4), "epoch_ahi_proxy": round(ahi_proxy, 4),
                "epoch_hb_proxy": round(hb_proxy, 4), "severity_bin": severity_bin(ahi_proxy),
                "awake_fraction": round(float(awake_fraction[e]), 4),
                "excluded_from_training": bool(awake_fraction[e] >= 0.5),
            })
        subjects_written += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "subject_id", "epoch_start_sec", "epoch_end_sec", "epoch_duration_sec", "desat_count",
            "hypoxic_burden_area_pctmin", "epoch_ahi_proxy", "epoch_hb_proxy", "severity_bin",
            "awake_fraction", "excluded_from_training",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} epochs across {subjects_written} subjects to {args.output}")


if __name__ == "__main__":
    main()
