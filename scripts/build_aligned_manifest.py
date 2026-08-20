#!/usr/bin/env python3
"""Materialize a paired manifest with device-specific Recorder timestamps."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.metadata import assign_balanced_folds, fold_protocol_rows, write_csv


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = read(PROJECT_ROOT / "metadata" / "dataset_manifest.csv")
    source_subjects = {row["subject_id"] for row in source}
    logical_subject = {row["logical_window_id"]: row["subject_id"] for row in source}
    anchor_rows = read(PROJECT_ROOT / "metadata" / "device_alignment_dense_anchors.csv")
    inventory = {row["subject_id"]: row for row in read(PROJECT_ROOT / "metadata" / "subject_inventory.csv")}
    reliable_by_subject: dict[str, list[dict]] = {}
    for row in anchor_rows:
        if row["reliable"] == "True":
            reliable_by_subject.setdefault(row["subject_id"], []).append(row)
    usable = {
        subject for subject, selected in reliable_by_subject.items()
        if len(selected) >= 3 and float(np.median([float(row["envelope_correlation"]) for row in selected])) >= 0.5
    }
    alignment_segments: dict[str, list[dict]] = {}
    for subject in usable:
        selected = sorted(
            reliable_by_subject[subject],
            key=lambda row: float(row["anchor_phone_sec"]),
        )
        times = np.asarray([float(row["anchor_phone_sec"]) for row in selected])
        lags = np.asarray([float(row["recorder_time_minus_phone_time_sec"]) for row in selected])
        groups: list[list[int]] = [[0]]
        for index in range(1, len(times)):
            previous = groups[-1][-1]
            if float(times[index] - times[previous]) <= 1800.0 and abs(float(lags[index] - lags[previous])) <= 5.0:
                groups[-1].append(index)
            else:
                groups.append([index])
        subject_segments = []
        for group_index, group in enumerate(groups):
            group_times = times[group]
            group_lags = lags[group]
            lower = max(0.0, float(group_times[0]) - 450.0) if group_index == 0 else float(group_times[0])
            phone_duration = float(inventory[subject]["phone_duration_sec"])
            upper = min(phone_duration, float(group_times[-1]) + 450.0) if group_index == len(groups) - 1 else float(group_times[-1])
            if upper > lower:
                subject_segments.append({"lower": lower, "upper": upper, "times": group_times, "lags": group_lags})
        alignment_segments[subject] = subject_segments
    output = []
    excluded_coverage = set()
    for row in source:
        subject = row["subject_id"]
        if subject not in usable:
            continue
        reference_start = float(row["start_sec"])
        reference_end = float(row["end_sec"])
        segment = next(
            (candidate for candidate in alignment_segments[subject] if reference_start >= candidate["lower"] and reference_end <= candidate["upper"]),
            None,
        )
        if segment is None:
            excluded_coverage.add(row["logical_window_id"])
            continue
        new = dict(row)
        new["reference_start_sec"] = reference_start
        new["reference_end_sec"] = reference_end
        if row["device"] == "R":
            lag_start = float(np.interp(reference_start, segment["times"], segment["lags"]))
            lag_end = float(np.interp(reference_end, segment["times"], segment["lags"]))
            start = reference_start + lag_start
            end = reference_end + lag_end
            recorder_duration = float(inventory[subject]["recorder_duration_sec"])
            if start < 0 or end > recorder_duration or end <= start:
                excluded_coverage.add(row["logical_window_id"])
                continue
            new["start_sec"] = round(start, 6)
            new["end_sec"] = round(end, 6)
            new["duration_sec"] = round(end - start, 6)
            new["device_time_offset_start_sec"] = round(lag_start, 6)
            new["device_time_offset_end_sec"] = round(lag_end, 6)
        else:
            new["device_time_offset_start_sec"] = 0.0
            new["device_time_offset_end_sec"] = 0.0
        output.append(new)
    if excluded_coverage:
        output = [row for row in output if row["logical_window_id"] not in excluded_coverage]
    pairs: dict[str, list[dict]] = {}
    for row in output:
        pairs.setdefault(row["logical_window_id"], []).append(row)
    invalid_pairs = [logical_id for logical_id, pair in pairs.items() if len(pair) != 2 or {row["device"] for row in pair} != {"R", "S"}]
    if invalid_pairs:
        raise ValueError(f"Invalid aligned pairs: {invalid_pairs[:5]}")
    output.sort(key=lambda row: (row["subject_id"], row["logical_window_id"], row["device"]))
    subject_summary = []
    for subject in sorted(usable):
        rows = [row for row in output if row["subject_id"] == subject and row["device"] == "R"]
        positives = sum(int(row["label"]) for row in rows)
        subject_summary.append(
            {"subject_id": subject, "positive_logical_windows": positives, "negative_logical_windows": len(rows) - positives,
             "excluded_alignment_coverage_windows": sum(logical_subject[logical_id] == subject for logical_id in excluded_coverage)}
        )
    folds = assign_balanced_folds(subject_summary, n_folds=5, seed=42)
    fold_protocols = fold_protocol_rows(folds, n_folds=5)
    targets = {
        PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv": output,
        PROJECT_ROOT / "metadata" / "subject_window_summary_aligned.csv": subject_summary,
        PROJECT_ROOT / "metadata" / "subject_folds_5cv_aligned.csv": folds,
        PROJECT_ROOT / "metadata" / "fold_protocols_5cv_aligned.csv": fold_protocols,
    }
    for path in targets:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    for path, rows in targets.items():
        write_csv(path, rows)
    validation = {
        "valid": True, "alignment_uses_labels": False, "usable_paired_subjects": len(usable),
        "excluded_alignment_subjects": sorted(source_subjects - usable),
        "excluded_out_of_aligned_coverage_logical_windows": len(excluded_coverage),
        "manifest_rows": len(output), "logical_windows": len(pairs),
        "positive_rows": sum(int(row["label"]) for row in output),
        "negative_rows": sum(1 - int(row["label"]) for row in output),
        "fold_subject_counts": {str(fold): sum(int(row["fold"]) == fold for row in folds) for fold in range(5)},
        "fold_positive_window_counts": {str(fold): sum(int(row["positive_windows"]) for row in folds if int(row["fold"]) == fold) for fold in range(5)},
        "reference_clock": "Smartphone/annotation seconds",
        "recorder_mapping": "piecewise-linear interpolation within continuous reliable unlabeled RMS-envelope lag-anchor segments",
        "alignment_window_inclusion": "both reference endpoints must lie in one segment; segments split at anchor gaps >1800 s or adjacent lag changes >5 s; intervals between segments are excluded",
    }
    path = PROJECT_ROOT / "metadata" / "aligned_metadata_validation.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
