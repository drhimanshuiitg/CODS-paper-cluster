#!/usr/bin/env python3
"""Q1_Paper_Artifact analysis: characterize acoustic domain shift between the
Recorder (Newamy V03) and Smartphone (OPPO Reno8) devices directly from raw
audio -- RMS energy, spectral centroid, spectral bandwidth, spectral
rolloff, zero-crossing rate -- on a representative, stratified (by device,
label) sample of manifest windows. CPU-only, no retraining, no GPU.

Outputs:
  Q1_Paper_Artifact/tables/domain_shift_audio_stats.csv  (per-window stats)
  Q1_Paper_Artifact/analysis/domain_shift_summary.json    (per-device/label summary + Mann-Whitney U tests)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.io import load_manifest_window, load_yaml, read_csv_rows


def stratified_sample(rows: list[dict], per_cell: int, seed: int = 42) -> list[dict]:
    rng = np.random.RandomState(seed)
    strata: dict[tuple, list[dict]] = {}
    for row in rows:
        strata.setdefault((row["device"], row["label"]), []).append(row)
    selected = []
    for key, items in strata.items():
        idx = rng.choice(len(items), size=min(per_cell, len(items)), replace=False)
        selected.extend(items[i] for i in idx)
    return selected


def compute_stats(audio: np.ndarray, sr: int) -> dict:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    return {"rms": rms, "spectral_centroid_hz": centroid, "spectral_bandwidth_hz": bandwidth,
            "spectral_rolloff_hz": rolloff, "zero_crossing_rate": zcr, "spectral_flatness": flatness}


def main() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    rows = read_csv_rows(PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    sample = stratified_sample(rows, per_cell=150)
    print(f"sampling {len(sample)} windows (stratified by device x label)")

    records = []
    for row in sample:
        try:
            audio, sr = load_manifest_window(row, config, "raw")
        except Exception as exc:
            continue
        st = compute_stats(audio, sr)
        records.append({"sample_id": row["sample_id"], "subject_id": row["subject_id"], "device": row["device"],
                         "label": int(row["label"]), "duration_sec": float(row["duration_sec"]), **st})

    df = pd.DataFrame(records)
    out_tables = PROJECT_ROOT / "Q1_Paper_Artifact" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tables / "domain_shift_audio_stats.csv", index=False)

    metrics = ["rms", "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff_hz", "zero_crossing_rate", "spectral_flatness"]
    summary = {}
    r = df[df.device == "R"]
    s = df[df.device == "S"]
    for m in metrics:
        u_stat, p_value = stats.mannwhitneyu(r[m], s[m], alternative="two-sided")
        summary[m] = {
            "recorder_mean": float(r[m].mean()), "recorder_std": float(r[m].std()),
            "smartphone_mean": float(s[m].mean()), "smartphone_std": float(s[m].std()),
            "mann_whitney_u": float(u_stat), "p_value": float(p_value),
            "cliffs_delta": float((2 * u_stat / (len(r) * len(s))) - 1),  # rank-biserial effect size
        }
    out_analysis = PROJECT_ROOT / "Q1_Paper_Artifact" / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)
    (out_analysis / "domain_shift_summary.json").write_text(json.dumps({
        "n_recorder": len(r), "n_smartphone": len(s), "metrics": summary,
        "note": "Mann-Whitney U (two-sided) on window-level statistics, stratified by device x label. "
                "This is a window-level test (not subject-level); treat p-values as indicative of aggregate "
                "acoustic distributional shift, not as a subject-independent significance claim in the sense "
                "used elsewhere in this project's classifier evaluation.",
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
