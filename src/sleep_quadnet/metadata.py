"""Build the paired-device window manifest and fixed subject folds.

This module deliberately uses only the Python standard library so the metadata
audit can run before the GPU environment is installed. It never writes to the
raw dataset and never materializes audio clips.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AudioSegment:
    path: str
    sample_rate: int
    frames: int
    duration_sec: float


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def wav_info(path: Path) -> AudioSegment:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV: {path}")
        if handle.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM WAV: {path}")
        rate = int(handle.getframerate())
        frames = int(handle.getnframes())
    return AudioSegment(str(path.resolve()), rate, frames, frames / rate)


def _recorder_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.endswith("_recorder"):
        return (0, path.name)
    try:
        return (int(stem.rsplit("_", 1)[1]), path.name)
    except (IndexError, ValueError):
        return (10_000, path.name)


def recorder_paths(subject_dir: Path, subject_id: str) -> list[Path]:
    paths = sorted(subject_dir.glob(f"{subject_id}_recorder*.wav"), key=_recorder_sort_key)
    return [path for path in paths if path.is_file()]


def merge_intervals(intervals: Iterable[Interval], lower: float, upper: float) -> list[Interval]:
    clipped = [
        Interval(max(lower, item.start), min(upper, item.end))
        for item in intervals
        if item.end > lower and item.start < upper
    ]
    clipped = [item for item in clipped if item.end > item.start]
    clipped.sort(key=lambda item: (item.start, item.end))
    merged: list[Interval] = []
    for item in clipped:
        if not merged or item.start > merged[-1].end:
            merged.append(item)
        else:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, item.end))
    return merged


def complement_intervals(forbidden: Sequence[Interval], lower: float, upper: float) -> list[Interval]:
    allowed: list[Interval] = []
    cursor = lower
    for item in merge_intervals(forbidden, lower, upper):
        if item.start > cursor:
            allowed.append(Interval(cursor, item.start))
        cursor = max(cursor, item.end)
    if cursor < upper:
        allowed.append(Interval(cursor, upper))
    return allowed


def overlaps_any(candidate: Interval, intervals: Sequence[Interval], tolerance: float = 1e-9) -> bool:
    return any(candidate.start < item.end - tolerance and candidate.end > item.start + tolerance for item in intervals)


def choose_interval(
    duration: float,
    forbidden: Sequence[Interval],
    lower: float,
    upper: float,
    rng: random.Random,
) -> Interval | None:
    choices: list[tuple[Interval, float]] = []
    total = 0.0
    for allowed in complement_intervals(forbidden, lower, upper):
        span = allowed.duration - duration
        if span < -1e-9:
            continue
        weight = max(span, 0.0) + 1e-6
        total += weight
        choices.append((allowed, total))
    if not choices:
        return None
    draw = rng.random() * total
    selected = choices[-1][0]
    previous = 0.0
    for allowed, cumulative in choices:
        if draw <= cumulative:
            selected = allowed
            previous = cumulative - (max(allowed.duration - duration, 0.0) + 1e-6)
            break
    local_fraction = 0.0 if selected.duration <= duration else (draw - previous) / max(total - previous, 1e-12)
    local_fraction = min(max(local_fraction, 0.0), 1.0)
    start = selected.start + local_fraction * max(selected.duration - duration, 0.0)
    return Interval(start, start + duration)


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def read_annotation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"record_start", "awake_intervals", "events"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing annotation keys {sorted(missing)} in {path}")
    return data


def subject_inventory(dataset_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for subject_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        subject_id = subject_dir.name
        annotation_path = subject_dir / f"{subject_id}_annotation.json"
        if not annotation_path.exists():
            continue
        phone_path = subject_dir / f"{subject_id}_phone.wav"
        phone = wav_info(phone_path) if phone_path.exists() else None
        recorder = [wav_info(path) for path in recorder_paths(subject_dir, subject_id)]
        annotation = read_annotation(annotation_path)
        sample_rates = sorted({seg.sample_rate for seg in recorder} | ({phone.sample_rate} if phone else set()))
        recorder_duration = sum(seg.duration_sec for seg in recorder)
        rows.append(
            {
                "subject_id": subject_id,
                "annotation_path": str(annotation_path.resolve()),
                "phone_available": bool(phone),
                "recorder_available": bool(recorder),
                "paired_available": bool(phone and recorder),
                "phone_path": phone.path if phone else "",
                "phone_duration_sec": phone.duration_sec if phone else "",
                "recorder_paths_json": json.dumps([seg.path for seg in recorder]),
                "recorder_durations_json": json.dumps([seg.duration_sec for seg in recorder]),
                "recorder_segments": len(recorder),
                "recorder_duration_sec": recorder_duration if recorder else "",
                "common_duration_sec": min(phone.duration_sec, recorder_duration) if phone and recorder else "",
                "sample_rates_json": json.dumps(sample_rates),
                "event_count": len(annotation["events"]),
                "awake_interval_count": len(annotation["awake_intervals"]),
            }
        )
    return rows


def _window_row(
    *,
    logical_window_id: str,
    subject_id: str,
    device: str,
    label: int,
    start: float,
    end: float,
    event_type: str,
    sleep_stage: str,
    paired_positive_id: str,
    inventory: dict,
    negative_overlap_fallback: bool,
) -> dict:
    if device == "S":
        audio_paths = [inventory["phone_path"]]
        audio_durations = [float(inventory["phone_duration_sec"])]
    elif device == "R":
        audio_paths = json.loads(inventory["recorder_paths_json"])
        audio_durations = json.loads(inventory["recorder_durations_json"])
    else:
        raise ValueError(f"Unknown device: {device}")
    return {
        "sample_id": f"{logical_window_id}_{device}",
        "logical_window_id": logical_window_id,
        "subject_id": subject_id,
        "device": device,
        "label": label,
        "event_type": event_type,
        "sleep_stage": sleep_stage,
        "start_sec": round(start, 6),
        "end_sec": round(end, 6),
        "duration_sec": round(end - start, 6),
        "paired_positive_id": paired_positive_id,
        "audio_paths_json": json.dumps(audio_paths),
        "audio_segment_durations_json": json.dumps(audio_durations),
        "raw_sample_rate": 8000,
        "target_sample_rate": 16000,
        "negative_overlap_fallback": negative_overlap_fallback,
    }


def build_window_manifest(
    inventory_rows: Sequence[dict],
    seed: int = 42,
    exclude_wake_events: bool = False,
    avoid_negative_overlap: bool = True,
) -> tuple[list[dict], list[dict]]:
    manifest: list[dict] = []
    subject_summary: list[dict] = []
    for inventory in inventory_rows:
        if not inventory["paired_available"]:
            continue
        subject_id = str(inventory["subject_id"])
        common_duration = float(inventory["common_duration_sec"])
        annotation = read_annotation(Path(inventory["annotation_path"]))
        record_start = float(annotation["record_start"])
        awake = [
            Interval(float(start) - record_start, float(end) - record_start)
            for start, end in annotation["awake_intervals"]
        ]
        all_events: list[tuple[int, dict, Interval]] = []
        for event_index, event in enumerate(annotation["events"]):
            start = float(event["evnet_start"]) - record_start
            interval = Interval(start, start + float(event["event_duration"]))
            all_events.append((event_index, event, interval))
        event_forbidden = [item[2] for item in all_events]
        eligible: list[tuple[int, dict, Interval]] = []
        excluded_coverage = 0
        excluded_wake = 0
        for event_index, event, interval in all_events:
            if interval.start < 0 or interval.end > common_duration or interval.duration <= 0:
                excluded_coverage += 1
                continue
            if exclude_wake_events and str(event.get("sleep_stage", "")) == "W":
                excluded_wake += 1
                continue
            eligible.append((event_index, event, interval))

        rng = random.Random(seed + int(subject_id))
        selected_negatives: list[Interval] = []
        fallback_count = 0
        subject_rows_before = len(manifest)
        for event_index, event, positive in eligible:
            positive_id = stable_id(subject_id, "positive", event_index, positive.start, positive.end)
            negative_forbidden = [*awake, *event_forbidden]
            if avoid_negative_overlap:
                negative_forbidden.extend(selected_negatives)
            negative = choose_interval(positive.duration, negative_forbidden, 0.0, common_duration, rng)
            used_fallback = False
            if negative is None and avoid_negative_overlap:
                negative = choose_interval(positive.duration, [*awake, *event_forbidden], 0.0, common_duration, rng)
                used_fallback = negative is not None
            if negative is None:
                continue
            if used_fallback:
                fallback_count += 1
            selected_negatives.append(negative)
            negative_id = stable_id(subject_id, "negative", event_index, negative.start, negative.end)
            for device in ("R", "S"):
                manifest.append(
                    _window_row(
                        logical_window_id=positive_id,
                        subject_id=subject_id,
                        device=device,
                        label=1,
                        start=positive.start,
                        end=positive.end,
                        event_type=str(event["event_type"]),
                        sleep_stage=str(event.get("sleep_stage", "")),
                        paired_positive_id=positive_id,
                        inventory=inventory,
                        negative_overlap_fallback=False,
                    )
                )
                manifest.append(
                    _window_row(
                        logical_window_id=negative_id,
                        subject_id=subject_id,
                        device=device,
                        label=0,
                        start=negative.start,
                        end=negative.end,
                        event_type="normal",
                        sleep_stage="",
                        paired_positive_id=positive_id,
                        inventory=inventory,
                        negative_overlap_fallback=used_fallback,
                    )
                )
        subject_rows = manifest[subject_rows_before:]
        subject_summary.append(
            {
                "subject_id": subject_id,
                "common_duration_sec": common_duration,
                "raw_events": len(all_events),
                "eligible_events": len(eligible),
                "excluded_outside_common_coverage": excluded_coverage,
                "excluded_wake_events": excluded_wake,
                "positive_logical_windows": sum(row["device"] == "R" and row["label"] == 1 for row in subject_rows),
                "negative_logical_windows": sum(row["device"] == "R" and row["label"] == 0 for row in subject_rows),
                "negative_overlap_fallbacks": fallback_count,
            }
        )
    manifest.sort(key=lambda row: (row["subject_id"], row["logical_window_id"], row["device"]))
    return manifest, subject_summary


def assign_balanced_folds(subject_summary: Sequence[dict], n_folds: int = 5, seed: int = 42) -> list[dict]:
    if len(subject_summary) < n_folds:
        raise ValueError("Not enough subjects for requested folds")
    rng = random.Random(seed)
    items = list(subject_summary)
    rng.shuffle(items)
    items.sort(key=lambda row: int(row["positive_logical_windows"]), reverse=True)
    fold_loads = [0] * n_folds
    fold_counts = [0] * n_folds
    assignments: list[dict] = []
    for item in items:
        fold = min(range(n_folds), key=lambda idx: (fold_loads[idx], fold_counts[idx], idx))
        windows = int(item["positive_logical_windows"])
        assignments.append({"subject_id": item["subject_id"], "fold": fold, "positive_windows": windows})
        fold_loads[fold] += windows
        fold_counts[fold] += 1
    assignments.sort(key=lambda row: row["subject_id"])
    return assignments


def fold_protocol_rows(assignments: Sequence[dict], n_folds: int = 5) -> list[dict]:
    by_fold = {fold: sorted(row["subject_id"] for row in assignments if int(row["fold"]) == fold) for fold in range(n_folds)}
    rows: list[dict] = []
    for test_fold in range(n_folds):
        val_fold = (test_fold + 1) % n_folds
        test_subjects = by_fold[test_fold]
        val_subjects = by_fold[val_fold]
        train_subjects = sorted(
            subject
            for fold, subjects in by_fold.items()
            if fold not in {test_fold, val_fold}
            for subject in subjects
        )
        rows.append(
            {
                "outer_fold": test_fold,
                "validation_fold": val_fold,
                "train_subjects_json": json.dumps(train_subjects),
                "val_subjects_json": json.dumps(val_subjects),
                "test_subjects_json": json.dumps(test_subjects),
            }
        )
    return rows


def validate_metadata(
    manifest: Sequence[dict],
    assignments: Sequence[dict],
    protocols: Sequence[dict],
    inventory_rows: Sequence[dict],
) -> dict:
    errors: list[str] = []
    subject_fold = {row["subject_id"]: int(row["fold"]) for row in assignments}
    if len(subject_fold) != len(assignments):
        errors.append("Duplicate subject in fold assignment")
    paired_subjects = {row["subject_id"] for row in inventory_rows if row["paired_available"]}
    if set(subject_fold) != paired_subjects:
        errors.append("Fold subjects do not exactly equal paired subjects")
    sample_ids = [row["sample_id"] for row in manifest]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("Duplicate sample_id in manifest")
    paired: dict[str, list[dict]] = {}
    for row in manifest:
        paired.setdefault(row["logical_window_id"], []).append(row)
        if float(row["start_sec"]) < 0 or float(row["end_sec"]) <= float(row["start_sec"]):
            errors.append(f"Invalid interval for {row['sample_id']}")
    for window_id, rows in paired.items():
        if {row["device"] for row in rows} != {"R", "S"} or len(rows) != 2:
            errors.append(f"Window {window_id} is not exactly paired across R/S")
            continue
        keys = ("subject_id", "label", "start_sec", "end_sec", "duration_sec")
        if any(rows[0][key] != rows[1][key] for key in keys):
            errors.append(f"Paired window mismatch for {window_id}")
    for protocol in protocols:
        train = set(json.loads(protocol["train_subjects_json"]))
        val = set(json.loads(protocol["val_subjects_json"]))
        test = set(json.loads(protocol["test_subjects_json"]))
        if train & val or train & test or val & test:
            errors.append(f"Subject leakage in outer fold {protocol['outer_fold']}")
        if train | val | test != paired_subjects:
            errors.append(f"Subject loss/duplication in outer fold {protocol['outer_fold']}")
    positives = sum(row["label"] == 1 for row in manifest)
    negatives = sum(row["label"] == 0 for row in manifest)
    return {
        "valid": not errors,
        "errors": errors,
        "paired_subjects": len(paired_subjects),
        "manifest_rows": len(manifest),
        "logical_windows": len(paired),
        "positive_rows": positives,
        "negative_rows": negatives,
        "fold_subject_counts": {
            str(fold): sum(int(row["fold"]) == fold for row in assignments) for fold in range(5)
        },
        "fold_positive_window_counts": {
            str(fold): sum(int(row["positive_windows"]) for row in assignments if int(row["fold"]) == fold)
            for fold in range(5)
        },
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_bundle(
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 42,
    n_folds: int = 5,
    exclude_wake_events: bool = False,
) -> dict:
    expected = [
        output_dir / "subject_inventory.csv",
        output_dir / "dataset_manifest.csv",
        output_dir / "subject_window_summary.csv",
        output_dir / "subject_folds_5cv.csv",
        output_dir / "fold_protocols_5cv.csv",
        output_dir / "metadata_validation.json",
    ]
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError(f"Metadata outputs already exist; refusing to overwrite: {existing}")
    inventory = subject_inventory(dataset_dir)
    manifest, summary = build_window_manifest(
        inventory,
        seed=seed,
        exclude_wake_events=exclude_wake_events,
    )
    assignments = assign_balanced_folds(summary, n_folds=n_folds, seed=seed)
    protocols = fold_protocol_rows(assignments, n_folds=n_folds)
    validation = validate_metadata(manifest, assignments, protocols, inventory)
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation, indent=2))
    write_csv(output_dir / "subject_inventory.csv", inventory)
    write_csv(output_dir / "dataset_manifest.csv", manifest)
    write_csv(output_dir / "subject_window_summary.csv", summary)
    write_csv(output_dir / "subject_folds_5cv.csv", assignments)
    write_csv(output_dir / "fold_protocols_5cv.csv", protocols)
    (output_dir / "metadata_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    durations = [float(row["duration_sec"]) for row in manifest if row["device"] == "R"]
    validation["logical_duration_sec_min"] = min(durations)
    validation["logical_duration_sec_median"] = statistics.median(durations)
    validation["logical_duration_sec_max"] = max(durations)
    return validation
