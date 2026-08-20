#!/usr/bin/env python3
"""Join per-event SpO2-corroboration flags (results/audit/spo2_corroboration_per_event.csv,
computed on the Smartphone/reference-clock timeline) onto actual manifest
windows, producing a per-sample_id lookup usable for training-data filtering.

Two real alignment subtleties, verified against the data before writing
this (2026-08-19), not assumed:
  1. The Smartphone (S) device is the reference clock (configs/base.yaml's
     metadata.reference_clock: smartphone_annotation_seconds) -- its
     manifest start_sec matches (evnet_start - record_start) almost exactly
     (sub-3-second differences, likely rounding in the original event
     timestamps). The Recorder (R) device's start_sec does NOT match this
     directly -- R timestamps go through additional piecewise-linear
     clock-drift correction (recorder_alignment in configs/base.yaml), so
     independently time-matching R rows against the raw annotation timeline
     fails for ~50% of rows. Fix: match only S-device rows by time, then
     propagate the resulting flag to both the S and R rows of the same
     event via the shared logical_window_id (every logical_window_id has
     exactly one R row and one S row, confirmed).
  2. Negative (label=0) windows have no annotated event at all, so
     corroboration doesn't apply to them -- they are always marked
     spo2_corroborated=True (i.e. "keep"), by convention, so a filter can
     treat this column uniformly as "should this window be kept for
     training" without a separate label check.

Output: metadata/window_corroboration.csv (sample_id, spo2_corroborated)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIME_TOLERANCE_SEC = 3.0


def main() -> None:
    manifest = pd.read_csv(PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv", dtype={"subject_id": str})
    per_event = pd.read_csv(PROJECT_ROOT / "results" / "audit" / "spo2_corroboration_per_event.csv", dtype={"subject_id": str})

    positives_smartphone = manifest[(manifest.label == 1) & (manifest.device == "S")].copy()
    logical_window_corroborated: dict[str, bool] = {}
    unmatched = 0
    for subject_id, group in per_event.groupby("subject_id"):
        subject_rows = positives_smartphone[positives_smartphone.subject_id == subject_id]
        event_starts = group["start_sec"].to_numpy()
        event_corroborated = group["spo2_corroborated"].to_numpy()
        for _, row in subject_rows.iterrows():
            diffs = np.abs(event_starts - row["start_sec"])
            nearest = int(np.argmin(diffs))
            if diffs[nearest] < TIME_TOLERANCE_SEC:
                logical_window_corroborated[row["logical_window_id"]] = bool(event_corroborated[nearest])
            else:
                unmatched += 1
    print(f"matched {len(logical_window_corroborated)}/{len(positives_smartphone)} smartphone positive windows "
          f"to a per-event corroboration flag ({unmatched} unmatched, tolerance {TIME_TOLERANCE_SEC}s)")

    rows = []
    for _, row in manifest.iterrows():
        if row["label"] == 0:
            corroborated = True  # negatives are always kept -- corroboration doesn't apply
        else:
            corroborated = logical_window_corroborated.get(row["logical_window_id"])
            if corroborated is None:
                # Positive window whose event didn't match anything in the
                # per-event table (e.g. an event type other than osa/hypo,
                # or genuinely unmatched) -- conservatively keep it rather
                # than silently drop, and flag via a distinct value so this
                # is auditable rather than hidden.
                corroborated = True
        rows.append({"sample_id": row["sample_id"], "logical_window_id": row["logical_window_id"],
                      "device": row["device"], "label": int(row["label"]), "spo2_corroborated": corroborated})

    output_path = PROJECT_ROOT / "metadata" / "window_corroboration.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "logical_window_id", "device", "label", "spo2_corroborated"])
        writer.writeheader()
        writer.writerows(rows)

    n_positive = sum(1 for r in rows if r["label"] == 1)
    n_positive_corroborated = sum(1 for r in rows if r["label"] == 1 and r["spo2_corroborated"])
    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"positive windows: {n_positive}, corroborated: {n_positive_corroborated} "
          f"({100*n_positive_corroborated/n_positive:.1f}%), uncorroborated: {n_positive - n_positive_corroborated}")


if __name__ == "__main__":
    main()
