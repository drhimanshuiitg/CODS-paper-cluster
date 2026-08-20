#!/usr/bin/env python3
"""Section D2 prerequisite: empirically characterize event-end-to-nadir SpO2
timing by event subtype {osa, hypo}, using the manifest's own already-
validated reference-clock event boundaries (reference_start_sec /
reference_end_sec, phone-reference timeline -- same clock as each subject's
{sid}_SpO2.csv 'relative position' column). Reuses the trusted alignment;
does not re-derive it from raw annotation.json record_start (which uses a
different clock reference not directly comparable without the same
correction the existing pipeline already applies).

Explicitly required by the master prompt before any fixed-lag SpO2 auxiliary
target may be defined: 'Do NOT use an arbitrary fixed lag without auditing
the actual dataset distribution.' Login-node safe: CSV parsing only."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
DATASET_ROOT = Path("/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/Data")
OUT_DIR = PROJECT / "paired_physio_device" / "results" / "physiology"
SEARCH_WINDOW_SEC = 150  # generous post-event search window; empirically trimmed below
BASELINE_WINDOW_SEC = 100  # matches this project's existing ODI baseline convention


def load_spo2(subject_id: str) -> np.ndarray | None:
    path = DATASET_ROOT / subject_id / f"{subject_id}_SpO2.csv"
    if not path.exists():
        return None
    vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        col = [k for k in reader.fieldnames if "OSat" in k][0]
        for row in reader:
            v = row[col]
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(np.nan)
    return np.array(vals, dtype=np.float64)


def main():
    manifest_path = PROJECT / "metadata" / "dataset_manifest_aligned.csv"
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    # one row per real event (dedupe R/S duplicates of the same paired_positive_id),
    # positives only, real subtypes only
    events_by_pid = {}
    for row in rows:
        if row["label"] != "1":
            continue
        pid = row["paired_positive_id"]
        if pid not in events_by_pid:
            events_by_pid[pid] = row

    spo2_cache: dict[str, np.ndarray] = {}
    results = defaultdict(list)  # event_type -> list of (delay_to_nadir_sec, amplitude, baseline)
    n_skipped = 0

    for pid, row in events_by_pid.items():
        sid = row["subject_id"]
        event_type = row["event_type"]
        if event_type not in ("osa", "hypo"):
            n_skipped += 1
            continue
        try:
            event_end = float(row["reference_end_sec"])
        except (KeyError, ValueError):
            n_skipped += 1
            continue

        if sid not in spo2_cache:
            spo2_cache[sid] = load_spo2(sid)
        spo2 = spo2_cache[sid]
        if spo2 is None:
            n_skipped += 1
            continue

        end_idx = int(round(event_end))
        base_start = max(0, end_idx - BASELINE_WINDOW_SEC)
        baseline_window = spo2[base_start:end_idx]
        baseline_window = baseline_window[~np.isnan(baseline_window)]
        if len(baseline_window) < 10:
            n_skipped += 1
            continue
        baseline = float(np.max(baseline_window))  # rolling-max baseline, matches ODI convention

        search_start = end_idx
        search_end = min(len(spo2), end_idx + SEARCH_WINDOW_SEC)
        search_window = spo2[search_start:search_end]
        if len(search_window) < 5 or np.all(np.isnan(search_window)):
            n_skipped += 1
            continue

        nadir_rel_idx = int(np.nanargmin(search_window))
        nadir_val = float(search_window[nadir_rel_idx])
        delay_to_nadir_sec = nadir_rel_idx  # seconds after event end, since 1 Hz
        amplitude = baseline - nadir_val

        results[event_type].append({
            "delay_to_nadir_sec": delay_to_nadir_sec,
            "amplitude": amplitude,
            "baseline": baseline,
            "nadir": nadir_val,
            "hit_search_boundary": nadir_rel_idx == len(search_window) - 1,
        })

    summary = {"search_window_sec": SEARCH_WINDOW_SEC, "baseline_window_sec": BASELINE_WINDOW_SEC,
               "n_events_total": len(events_by_pid), "n_skipped": n_skipped, "by_event_type": {}}

    for event_type, recs in results.items():
        delays = np.array([r["delay_to_nadir_sec"] for r in recs])
        amps = np.array([r["amplitude"] for r in recs])
        hit_boundary = np.array([r["hit_search_boundary"] for r in recs])
        summary["by_event_type"][event_type] = {
            "n": len(recs),
            "delay_to_nadir_sec": {
                "mean": float(np.mean(delays)), "median": float(np.median(delays)),
                "p10": float(np.percentile(delays, 10)), "p90": float(np.percentile(delays, 90)),
                "p95": float(np.percentile(delays, 95)),
            },
            "amplitude_pct": {
                "mean": float(np.mean(amps)), "median": float(np.median(amps)),
                "p90": float(np.percentile(amps, 90)),
                "fraction_ge_3pct": float(np.mean(amps >= 3.0)),
            },
            "fraction_hit_search_boundary": float(np.mean(hit_boundary)),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "spo2_event_timing_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
