#!/usr/bin/env python3
"""Cross-reference PSG-annotated OSA/hypopnea events against objectively-
detected SpO2 desaturation events, to audit annotation trustworthiness
rather than take it on faith.

For every annotated osa/hypo event, checks whether a nearby (+45s lag
tolerance, matching typical circulatory delay between a respiratory event
and its SpO2 nadir) desaturation (>=3% drop from a rolling 100s baseline,
excluding awake intervals) exists. Also reports SpO2 desaturations with no
nearby annotated event at all -- candidates for under-annotation.

Outputs (results/audit/):
  - spo2_corroboration_per_subject.csv   -- per-subject corroboration rates
  - spo2_corroboration_per_event.csv     -- per annotated event, corroborated True/False
  - uncorroborated_desat_candidates.csv  -- SpO2 desaturations with no matching annotation
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_odi_hypoxic_burden import load_spo2, mask_awake  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path("/userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data")
OUTPUT_DIR = PROJECT_ROOT / "results" / "audit"
LAG_TOLERANCE_SEC = 45.0  # circulatory delay between event onset and SpO2 nadir


def detect_desat_windows(times, values, awake_mask, drop_pct=3.0, baseline_window_sec=100.0, min_duration_sec=8.0):
    """Same detection as compute_odi_hypoxic_burden.detect_desaturations, but
    keeps event start/end times instead of collapsing to area-under-curve."""
    from collections import deque

    n = len(times)
    if n < 10:
        return []
    baseline = np.full(n, np.nan)
    window = deque()
    for i in range(n):
        window.append((times[i], values[i]))
        while window[0][0] < times[i] - baseline_window_sec:
            window.popleft()
        baseline[i] = max(v for _, v in window)
    below = (baseline - values >= drop_pct) & (~awake_mask)
    events = []
    i = 0
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            duration = times[j - 1] - times[i]
            if duration >= min_duration_sec:
                events.append({
                    "start": float(times[i]), "end": float(times[j - 1]), "duration": float(duration),
                    "max_drop_pct": float(np.max(baseline[i:j] - values[i:j])),
                })
            i = j
        else:
            i += 1
    return events


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_subject_rows = []
    per_event_rows = []
    uncorroborated_desat_rows = []

    for spo2_path in sorted(DATASET_DIR.glob("*/*_SpO2.csv")):
        subject_id = spo2_path.parent.name
        annotation_path = spo2_path.parent / f"{subject_id}_annotation.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        record_start = float(annotation["record_start"])
        awake_relative = [(s - record_start, e - record_start) for s, e in annotation.get("awake_intervals", [])]

        times, values = load_spo2(spo2_path, record_start)
        if len(times) < 10:
            continue
        awake_mask = mask_awake(times, awake_relative)
        desat_events = detect_desat_windows(times, values, awake_mask)

        annotated_events = []
        for event in annotation.get("events", []):
            if event.get("event_type") not in ("osa", "hypo"):
                continue
            start = float(event["evnet_start"]) - record_start
            duration = float(event["event_duration"])
            annotated_events.append({"type": event["event_type"], "start": start, "end": start + duration})

        matched_annotation_indices: set[int] = set()
        for desat in desat_events:
            overlapped_indices = [
                idx for idx, ann_event in enumerate(annotated_events)
                if desat["start"] <= ann_event["end"] + LAG_TOLERANCE_SEC and desat["end"] >= ann_event["start"]
            ]
            matched_annotation_indices.update(overlapped_indices)
            if not overlapped_indices:
                uncorroborated_desat_rows.append({
                    "subject_id": subject_id, "desat_start_sec": desat["start"], "desat_end_sec": desat["end"],
                    "duration_sec": desat["duration"], "max_drop_pct": desat["max_drop_pct"],
                })

        for idx, ann_event in enumerate(annotated_events):
            per_event_rows.append({
                "subject_id": subject_id, "event_type": ann_event["type"],
                "start_sec": ann_event["start"], "end_sec": ann_event["end"],
                "spo2_corroborated": idx in matched_annotation_indices,
            })

        n_corroborated = len(matched_annotation_indices)
        per_subject_rows.append({
            "subject_id": subject_id, "n_annotated": len(annotated_events), "n_corroborated": n_corroborated,
            "pct_corroborated": round(100 * n_corroborated / len(annotated_events), 1) if annotated_events else float("nan"),
            "n_desat_events_detected": len(desat_events),
        })

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {path}")

    write_csv(OUTPUT_DIR / "spo2_corroboration_per_subject.csv", per_subject_rows)
    write_csv(OUTPUT_DIR / "spo2_corroboration_per_event.csv", per_event_rows)
    write_csv(OUTPUT_DIR / "uncorroborated_desat_candidates.csv", uncorroborated_desat_rows)


if __name__ == "__main__":
    main()
