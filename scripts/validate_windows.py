#!/usr/bin/env python3
"""Independent raw-annotation audit of every materialized positive/negative window."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def overlaps(left, right, tolerance=1e-7):
    return left[0] < right[1] - tolerance and left[1] > right[0] + tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "audit" / "aligned_window_validation.json",
    )
    args = parser.parse_args()
    with (PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (PROJECT_ROOT / "metadata" / "subject_inventory.csv").open(newline="", encoding="utf-8") as handle:
        inventory = {row["subject_id"]: row for row in csv.DictReader(handle)}
    logical = {}
    for row in rows:
        logical.setdefault(row["logical_window_id"], []).append(row)
    errors = []
    negative_overlap_count = 0
    checked_positives = checked_negatives = 0
    for subject_id in sorted({row["subject_id"] for row in rows}):
        info = inventory[subject_id]
        annotation = json.loads(Path(info["annotation_path"]).read_text(encoding="utf-8"))
        origin = float(annotation["record_start"])
        events = [(float(event["evnet_start"]) - origin, float(event["evnet_start"]) - origin + float(event["event_duration"])) for event in annotation["events"]]
        awake = [(float(start) - origin, float(end) - origin) for start, end in annotation["awake_intervals"]]
        common_duration = float(info["common_duration_sec"])
        negatives = []
        for pair in logical.values():
            row = pair[0]
            if row["subject_id"] != subject_id:
                continue
            interval = (float(row.get("reference_start_sec", row["start_sec"])), float(row.get("reference_end_sec", row["end_sec"])))
            if interval[0] < 0 or interval[1] > common_duration + 1e-6 or interval[1] <= interval[0]:
                errors.append(f"coverage:{row['logical_window_id']}")
            if int(row["label"]) == 1:
                checked_positives += 1
                if not any(abs(interval[0] - event[0]) < 1e-6 and abs(interval[1] - event[1]) < 1e-6 for event in events):
                    errors.append(f"positive_annotation_mismatch:{row['logical_window_id']}")
            else:
                checked_negatives += 1
                if any(overlaps(interval, forbidden) for forbidden in events + awake):
                    errors.append(f"negative_forbidden_overlap:{row['logical_window_id']}")
                if any(overlaps(interval, previous) for previous in negatives):
                    negative_overlap_count += 1
                negatives.append(interval)
    for logical_id, pair in logical.items():
        if len(pair) != 2 or {row["device"] for row in pair} != {"R", "S"}:
            errors.append(f"pairing:{logical_id}")
    result = {
        "valid": not errors,
        "errors": errors[:100],
        "total_error_count": len(errors),
        "checked_positive_logical_windows": checked_positives,
        "checked_negative_logical_windows": checked_negatives,
        "negative_negative_overlap_count": negative_overlap_count,
        "negative_forbidden_overlap_count": sum(error.startswith("negative_forbidden_overlap") for error in errors),
        "paired_logical_windows": len(logical),
    }
    target = args.output
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
