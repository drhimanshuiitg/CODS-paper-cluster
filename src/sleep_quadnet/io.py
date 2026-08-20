"""Configuration, manifest, and raw-window I/O."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import yaml
from scipy import signal


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict, *extra: object) -> str:
    payload = json.dumps([config, *extra], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def parse_json_list(value: str) -> list:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON list")
    return parsed


def _read_native_window(paths: Sequence[str], durations: Sequence[float], start_sec: float, end_sec: float) -> tuple[np.ndarray, int]:
    if len(paths) != len(durations) or not paths:
        raise ValueError("Audio segment paths/durations are inconsistent")
    pieces: list[np.ndarray] = []
    stream_cursor = 0.0
    sample_rate: int | None = None
    for path_text, duration in zip(paths, durations):
        segment_start = stream_cursor
        segment_end = stream_cursor + float(duration)
        stream_cursor = segment_end
        overlap_start = max(start_sec, segment_start)
        overlap_end = min(end_sec, segment_end)
        if overlap_end <= overlap_start:
            continue
        with sf.SoundFile(path_text, "r") as handle:
            if handle.channels != 1 or handle.subtype not in {"PCM_16", "PCM_S16LE"}:
                raise ValueError(f"Unexpected WAV format: {path_text} ({handle.channels}, {handle.subtype})")
            if sample_rate is None:
                sample_rate = int(handle.samplerate)
            elif sample_rate != int(handle.samplerate):
                raise ValueError(f"Sample-rate mismatch across recorder segments: {paths}")
            local_start = overlap_start - segment_start
            local_end = overlap_end - segment_start
            first_frame = max(0, int(round(local_start * handle.samplerate)))
            last_frame = min(len(handle), int(round(local_end * handle.samplerate)))
            handle.seek(first_frame)
            pieces.append(handle.read(last_frame - first_frame, dtype="float32", always_2d=False))
    if sample_rate is None or not pieces:
        raise ValueError(f"Requested window [{start_sec}, {end_sec}) has no audio coverage")
    audio = np.concatenate(pieces).astype(np.float32, copy=False)
    expected = int(round((end_sec - start_sec) * sample_rate))
    if abs(len(audio) - expected) > 2:
        raise ValueError(f"Window length mismatch: got {len(audio)}, expected {expected}")
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    elif len(audio) > expected:
        audio = audio[:expected]
    return audio, sample_rate


def resample_polyphase(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)


def butter_bandpass(audio: np.ndarray, sample_rate: int, low_hz: float = 20.0, high_hz: float = 4000.0, order: int = 4) -> np.ndarray:
    nyquist = sample_rate / 2.0
    low = max(low_hz / nyquist, 1e-6)
    high = min(high_hz / nyquist, 0.999999)
    if low >= high:
        raise ValueError(f"Invalid bandpass {low_hz}-{high_hz} Hz for fs={sample_rate}")
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32, copy=False)


def load_manifest_window(row: dict, config: dict, preprocessing: str) -> tuple[np.ndarray, int]:
    paths = [str(item) for item in parse_json_list(row["audio_paths_json"])]
    dataset_dir = Path(config["dataset_dir"])
    resolved_paths: list[str] = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            # The aligned manifest is a provenance artifact generated on the
            # original cluster and therefore retains its original absolute
            # prefix. Resolve the stable subject/file suffix against the
            # configured read-only dataset root without rewriting the CSV.
            parts = path.parts
            try:
                data_index = parts.index("Data")
            except ValueError:
                data_index = -1
            if data_index >= 0 and len(parts) > data_index + 2:
                candidate = dataset_dir.joinpath(*parts[data_index + 1 :])
                if candidate.exists():
                    path = candidate
        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest audio file is unavailable: {path_text}; "
                f"also checked configured dataset root {dataset_dir}"
            )
        resolved_paths.append(str(path))
    paths = resolved_paths
    durations = [float(item) for item in parse_json_list(row["audio_segment_durations_json"])]
    audio, native_rate = _read_native_window(paths, durations, float(row["start_sec"]), float(row["end_sec"]))
    target_rate = int(config["audio"]["target_sample_rate"])
    audio = resample_polyphase(audio, native_rate, target_rate)
    if preprocessing in {"filter", "peak_filter"}:
        filter_config = config["audio"]["filter"]
        audio = butter_bandpass(
            audio,
            target_rate,
            low_hz=float(filter_config["low_hz"]),
            high_hz=float(filter_config["high_hz"]),
            order=int(filter_config["order"]),
        )
    if preprocessing in {"peak", "peak_filter"}:
        peak = float(np.max(np.abs(audio)))
        epsilon = float(config["audio"].get("peak_epsilon", 1e-8))
        if peak > epsilon:
            audio = audio / peak
    if not np.isfinite(audio).all() or audio.size == 0:
        raise ValueError(f"Invalid audio after preprocessing for {row['sample_id']}")
    return audio.astype(np.float32, copy=False), target_rate


def manifest_index(rows: Sequence[dict]) -> dict[str, int]:
    result = {row["sample_id"]: index for index, row in enumerate(rows)}
    if len(result) != len(rows):
        raise ValueError("Duplicate sample IDs in manifest")
    return result


def write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_completed_runs(results_root: Path, key_fields: Sequence[str]) -> list[dict]:
    """Scan results_root/runs/*/completion.json for status=="complete" records,
    keeping only the latest-timestamped record per key_fields combination.

    Two unrelated events in this project's history have each left more than
    one "complete" run directory on disk for the same logical
    (representation, classifier, protocol, fold[, ...]) combo: migrating a
    classifier to GPU (evaluation.py/advanced.py's gpu_tag mechanism gives
    the new run a fresh experiment_key, but never deletes the old CPU-era
    completion.json), and a bugfix that bumps a config_hash salt (e.g. the
    Platt-scaling-calibration fix's dimension_control_v1->v2 bump). In both
    cases the newer record is the valid one, so callers that group/average
    over `runs/*/completion.json` MUST use this helper rather than reading
    the raw glob directly, or a duplicated combo silently gets double-
    counted (or, worse, the stale one gets kept) in any aggregate statistic.
    """
    best: dict[tuple, dict] = {}
    for path in sorted(results_root.glob("runs/*/completion.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") != "complete":
            continue
        record["result_dir"] = path.parent
        combo = tuple(record.get(field) for field in key_fields)
        existing = best.get(combo)
        if existing is None or datetime.fromisoformat(record["timestamp"]) > datetime.fromisoformat(existing["timestamp"]):
            best[combo] = record
    return list(best.values())


def append_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    import fcntl

    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
