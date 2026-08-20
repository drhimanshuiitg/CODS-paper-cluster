#!/usr/bin/env python3
"""Stage 0 audit: build raw_audio_inventory.csv and pair_inventory.csv directly
from raw dataset files and the existing aligned manifest -- per
CLAUDE_CODE_MASTER_PROMPT.md Section A. Login-node safe: only WAV header reads
(stdlib `wave`), JSON/CSV parsing, no model computation, no GPU needed."""
import csv
import json
import wave
from pathlib import Path

DATASET_ROOT = Path("/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/Data")
PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
OUT_DIR = PROJECT / "paired_physio_device" / "audit"

# ---------------------------------------------------------------------------
# raw_audio_inventory.csv
# ---------------------------------------------------------------------------

def wav_info(path: Path):
    with wave.open(str(path), "rb") as w:
        return {
            "original_sr": w.getframerate(),
            "channels": w.getnchannels(),
            "sampwidth_bytes": w.getsampwidth(),
            "duration_sec": round(w.getnframes() / w.getframerate(), 3),
        }


def build_raw_audio_inventory():
    rows = []
    subject_dirs = sorted(DATASET_ROOT.iterdir())
    for sdir in subject_dirs:
        if not sdir.is_dir():
            continue
        sid = sdir.name
        device_files = {
            "S": sorted(sdir.glob(f"{sid}_phone.wav")),
            "R": sorted(sdir.glob(f"{sid}_recorder_*.wav")),
        }
        for device, files in device_files.items():
            for f in files:
                try:
                    info = wav_info(f)
                    fmt = "PCM16 mono WAV"
                    status = "read_ok"
                except Exception as e:
                    info = {"original_sr": None, "channels": None,
                             "sampwidth_bytes": None, "duration_sec": None}
                    fmt = "UNREADABLE"
                    status = f"error: {e}"
                rows.append({
                    "subject_id": sid,
                    "device": device,
                    "source_file": str(f),
                    "original_sr": info["original_sr"],
                    "channels": info["channels"],
                    "format": fmt,
                    "duration_sec": info["duration_sec"],
                    "processed_sr": 16000,  # confirmed target rate, src/sleep_quadnet preprocessing
                    "alignment_status": status,
                })
    return rows


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {path}")


# ---------------------------------------------------------------------------
# pair_inventory.csv (from dataset_manifest_aligned.csv, grouped by paired_positive_id)
# ---------------------------------------------------------------------------

def build_pair_inventory():
    manifest_path = PROJECT / "metadata" / "dataset_manifest_aligned.csv"
    with open(manifest_path) as f:
        reader = list(csv.DictReader(f))

    by_pair = {}
    for row in reader:
        pid = row["paired_positive_id"]
        by_pair.setdefault(pid, {})[row["device"]] = row

    rows = []
    for pid, devs in sorted(by_pair.items()):
        r_row = devs.get("R")
        s_row = devs.get("S")
        if r_row is None or s_row is None:
            # negative windows are not necessarily paired 1:1 across devices;
            # only emit a pair_inventory row when both device rows exist for this id
            continue
        # alignment error proxy: difference between device-corrected reference
        # start times already stored per-row (device_time_offset_*), in ms
        try:
            r_off = float(r_row["device_time_offset_start_sec"])
            s_off = float(s_row["device_time_offset_start_sec"])
            err_ms = round(abs(r_off - s_off) * 1000, 2)
        except (KeyError, ValueError):
            err_ms = None
        rows.append({
            "subject_id": r_row["subject_id"],
            "paired_event_id": pid,
            "event_type": r_row["event_type"],
            "event_start": r_row["reference_start_sec"],
            "event_end": r_row["reference_end_sec"],
            "recorder_file": r_row["audio_paths_json"],
            "smartphone_file": s_row["audio_paths_json"],
            "pair_alignment_error_ms": err_ms,
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = build_raw_audio_inventory()
    write_csv(raw_rows, OUT_DIR / "raw_audio_inventory.csv",
              ["subject_id", "device", "source_file", "original_sr", "channels",
               "format", "duration_sec", "processed_sr", "alignment_status"])

    pair_rows = build_pair_inventory()
    write_csv(pair_rows, OUT_DIR / "pair_inventory.csv",
              ["subject_id", "paired_event_id", "event_type", "event_start",
               "event_end", "recorder_file", "smartphone_file",
               "pair_alignment_error_ms"])

    # quick summary printed for the audit report
    n_subjects_with_audio = len({r["subject_id"] for r in raw_rows})
    n_pairs = len(pair_rows)
    n_subjects_paired = len({r["subject_id"] for r in pair_rows})
    print(f"subjects with any raw audio found: {n_subjects_with_audio}")
    print(f"paired positive events (both R and S present): {n_pairs}")
    print(f"subjects contributing >=1 paired positive event: {n_subjects_paired}")


if __name__ == "__main__":
    main()
