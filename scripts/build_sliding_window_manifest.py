#!/usr/bin/env python3
"""Build a load_manifest_window()-compatible manifest for the sliding-window
AHI/severity epochs (metadata/sliding_window_ahi_targets.csv), so the
existing extract_feature_cache() machinery can extract audio features for
them unchanged -- same resumable caching, same GPU-only encoder loading,
same everything, just pointed at a different manifest/cache-root pair.

Each row borrows audio_paths_json/audio_segment_durations_json from a real
manifest row of the same subject+device (those fields describe the audio
*file*, not any specific window -- valid for any window drawn from the same
continuous recording; same technique already used in
scripts/plot_spo2_corroboration_examples.py:build_row()). One row per
(subject, device, epoch) for every subject that has BOTH a real manifest
audio-file template AND sliding-window targets (41/50 subjects -- the same
9 subjects missing from the main classification manifest are missing here
too, for the same underlying reason: no usable device recording).

Output: metadata/sliding_window_audio_manifest.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def available_duration_sec(durations_json: str) -> float:
    return float(sum(json.loads(durations_json)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-manifest", type=Path, default=PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    parser.add_argument("--targets", type=Path, default=PROJECT_ROOT / "metadata" / "sliding_window_ahi_targets.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "metadata" / "sliding_window_audio_manifest.csv")
    args = parser.parse_args()

    main_manifest = pd.read_csv(args.main_manifest, dtype={"subject_id": str})
    targets = pd.read_csv(args.targets, dtype={"subject_id": str})

    template_cols = ["subject_id", "device", "audio_paths_json", "audio_segment_durations_json", "raw_sample_rate", "target_sample_rate"]
    templates = main_manifest[template_cols].drop_duplicates(["subject_id", "device"])

    rows = []
    skipped_subjects = set(targets.subject_id.unique()) - set(templates.subject_id.unique())
    skipped_past_audio_end = 0
    for _, target_row in targets.iterrows():
        subject_id = target_row["subject_id"]
        subject_templates = templates[templates.subject_id == subject_id]
        for _, template in subject_templates.iterrows():
            device = template["device"]
            # Bug fix (2026-08-19): epochs were binned off the SpO2 channel's
            # own timeline (build_sliding_window_ahi_targets.py), which does
            # not necessarily end at the same instant as this device's audio
            # recording -- confirmed for real: job 1548 crashed 149/9950 rows
            # in on a window requesting 2,400,000 samples where only 463,360
            # existed. Drop (not clip) any epoch that runs past this specific
            # device's actual available audio, rather than silently
            # truncating to an inconsistent, shorter-than-labeled clip.
            if target_row["epoch_end_sec"] > available_duration_sec(template["audio_segment_durations_json"]):
                skipped_past_audio_end += 1
                continue
            sample_id = f"SW_{subject_id}_{device}_{int(target_row['epoch_start_sec'])}"
            rows.append({
                "sample_id": sample_id,
                "logical_window_id": f"SW_{subject_id}_{int(target_row['epoch_start_sec'])}",
                "subject_id": subject_id,
                "device": device,
                "start_sec": target_row["epoch_start_sec"],
                "end_sec": target_row["epoch_end_sec"],
                "duration_sec": target_row["epoch_duration_sec"],
                "audio_paths_json": template["audio_paths_json"],
                "audio_segment_durations_json": template["audio_segment_durations_json"],
                "raw_sample_rate": template["raw_sample_rate"],
                "target_sample_rate": template["target_sample_rate"],
                "desat_count": target_row["desat_count"],
                "hypoxic_burden_area_pctmin": target_row["hypoxic_burden_area_pctmin"],
                "epoch_ahi_proxy": target_row["epoch_ahi_proxy"],
                "epoch_hb_proxy": target_row["epoch_hb_proxy"],
                "severity_bin": target_row["severity_bin"],
                "awake_fraction": target_row["awake_fraction"],
                "excluded_from_training": target_row["excluded_from_training"],
            })

    frame = pd.DataFrame(rows)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in sliding-window audio manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} rows ({frame.subject_id.nunique()} subjects x up to 2 devices) to {args.output}")
    if skipped_subjects:
        print(f"skipped {len(skipped_subjects)} subjects with SpO2 targets but no usable audio-file template: {sorted(skipped_subjects)}")
    if skipped_past_audio_end:
        print(f"skipped {skipped_past_audio_end} epoch rows that ran past that device's actual available audio duration")


if __name__ == "__main__":
    main()
