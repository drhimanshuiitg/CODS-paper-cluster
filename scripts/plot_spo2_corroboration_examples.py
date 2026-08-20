#!/usr/bin/env python3
"""Illustrative figure: spectrogram + SpO2 trace for three concrete cases from
the SpO2-corroboration audit (scripts/audit_spo2_corroboration.py) --
(1) an annotated OSA event with SpO2 corroboration, (2) an annotated
hypopnea with none (plausibly an arousal-only scoring, not an error),
(3) a candidate under-annotated event: a real SpO2 desaturation with no
nearby PSG annotation at all. Reuses the pipeline's own validated audio
alignment (src/sleep_quadnet/io.py::load_manifest_window) rather than
re-deriving raw-audio timing, since the manifest's start_sec was confirmed
(2026-08-19) to be exactly evnet_start - record_start, the same relative
timeline the SpO2 computation uses.

Saves results/audit/figures/spo2_corroboration_examples.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sleep_quadnet.io import load_manifest_window, load_yaml  # noqa: E402
from compute_odi_hypoxic_burden import load_spo2, mask_awake  # noqa: E402
from audit_spo2_corroboration import LAG_TOLERANCE_SEC, detect_desat_windows  # noqa: E402

DATASET_DIR = Path("/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/Data")
CONTEXT_PAD_SEC = 15.0


def build_row(manifest: pd.DataFrame, subject_id: str, device: str, start_sec: float, end_sec: float) -> dict:
    """Build a load_manifest_window()-compatible row for an arbitrary time
    window, by borrowing audio_paths_json/audio_segment_durations_json from
    any real manifest row of the same subject+device (those fields describe
    the audio *file*, not the specific window, so they're valid for any
    window drawn from the same continuous recording)."""
    template = manifest[(manifest.subject_id == subject_id) & (manifest.device == device)].iloc[0]
    row = template.to_dict()
    row["start_sec"] = start_sec
    row["end_sec"] = end_sec
    return row


BASELINE_BUFFER_SEC = 150.0  # extra context fed into detection so the rolling baseline is accurate near the plotted window's edges


def spo2_series_and_detections(subject_id: str, window_start: float, window_end: float):
    annotation = json.loads((DATASET_DIR / subject_id / f"{subject_id}_annotation.json").read_text())
    record_start = float(annotation["record_start"])
    awake_relative = [(s - record_start, e - record_start) for s, e in annotation.get("awake_intervals", [])]
    times_full, values_full = load_spo2(DATASET_DIR / subject_id / f"{subject_id}_SpO2.csv", record_start)

    buffered_mask = (times_full >= window_start - BASELINE_BUFFER_SEC) & (times_full <= window_end + BASELINE_BUFFER_SEC)
    times_buffered, values_buffered = times_full[buffered_mask], values_full[buffered_mask]
    awake_mask_buffered = mask_awake(times_buffered, awake_relative)
    detected = detect_desat_windows(times_buffered, values_buffered, awake_mask_buffered)
    # keep only detections that actually overlap the plotted window
    detected_in_view = [d for d in detected if d["end"] >= window_start and d["start"] <= window_end]

    display_mask = (times_full >= window_start) & (times_full <= window_end)
    return times_full[display_mask], values_full[display_mask], detected_in_view


def plot_example(fig, col, config, manifest, subject_id, device, event_start, event_end, event_label, corroborated, is_annotated):
    window_start = event_start - CONTEXT_PAD_SEC
    window_end = event_end + CONTEXT_PAD_SEC

    row = build_row(manifest, subject_id, device, window_start, window_end)
    audio, sample_rate = load_manifest_window(row, config, "peak")

    ax_spec = fig.add_subplot(2, 3, col)
    spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=256, n_mels=96, fmax=4000)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    duration = len(audio) / sample_rate
    # Manual extent in plain seconds (not librosa's auto mm:ss formatter,
    # which silently picks a different tick style per panel depending on
    # each clip's own duration -- forcing plain seconds keeps all three
    # panels, and their SpO2 panels below, on the same directly-comparable
    # x-axis convention.
    ax_spec.imshow(spec_db, origin="lower", aspect="auto", cmap="magma",
                    extent=(0, duration, 0, sample_rate / 2))
    ax_spec.axvspan(CONTEXT_PAD_SEC, CONTEXT_PAD_SEC + (event_end - event_start), color="white", alpha=0.15, lw=0)
    ax_spec.set_ylim(0, 4000)
    ax_spec.set_xlim(0, duration)
    status = "SpO2-corroborated" if corroborated else ("no SpO2 corroboration" if is_annotated else "NOT annotated by PSG scorer")
    ax_spec.set_title(f"{event_label}\nsubject {subject_id}, {status}", fontsize=9)
    if col == 1:
        ax_spec.set_ylabel("Hz")

    ax_spo2 = fig.add_subplot(2, 3, col + 3)
    times, values, detections = spo2_series_and_detections(subject_id, window_start, window_end)
    relative_times = times - window_start
    ax_spo2.plot(relative_times, values, color="#c0563d" if not corroborated and is_annotated else "#2f5f8f", linewidth=1.4)
    if is_annotated:
        # the annotated PSG event window itself
        ax_spo2.axvspan(CONTEXT_PAD_SEC, CONTEXT_PAD_SEC + (event_end - event_start), color="#2f5f8f", alpha=0.10, lw=0, label="annotated event")
    view_duration = window_end - window_start
    for detection in detections:
        # the actual objectively-detected desaturation window(s) -- may be
        # offset from the annotated window by the circulatory lag tolerance.
        # Detection ran on a wider buffered context for an accurate rolling
        # baseline (see spo2_series_and_detections), so an event's true
        # extent can run past the plotted axis -- clip the drawn span to
        # the visible range or it renders as shading with no underlying
        # SpO2 curve, which reads as a rendering bug rather than real data.
        d_start = max(0.0, detection["start"] - window_start)
        d_end = min(view_duration, detection["end"] - window_start)
        if d_end <= d_start:
            continue
        ax_spo2.axvspan(d_start, d_end, color="#c0563d", alpha=0.22, lw=1.2, label="detected desaturation")
    handles, labels = ax_spo2.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax_spo2.legend(by_label.values(), by_label.keys(), fontsize=6.5, loc="lower left")
    ax_spo2.set_ylim(70, 100)
    ax_spo2.set_xlabel("Time (s)")
    if col == 1:
        ax_spo2.set_ylabel("SpO2 (%)")
    ax_spo2.grid(alpha=0.25)


def main() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    config["project_root"] = str(PROJECT_ROOT)
    manifest = pd.read_csv(PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv", dtype={"subject_id": str})

    fig = plt.figure(figsize=(13, 6.5))

    # Example 1: corroborated OSA (subject 02, R->? device -- use S since phone.wav is a single contiguous file)
    plot_example(fig, 1, config, manifest, "02", "S", 1818.0, 1851.2, "OSA event", corroborated=True, is_annotated=True)
    # Example 2: uncorroborated hypopnea (subject 01)
    plot_example(fig, 2, config, manifest, "01", "S", 9698.0, 9722.1, "Hypopnea event", corroborated=False, is_annotated=True)
    # Example 3: candidate under-annotated event (subject 36) -- real SpO2 desaturation, no nearby PSG annotation
    plot_example(fig, 3, config, manifest, "36", "S", 12578.0, 12614.0, "Unannotated SpO2 desaturation", corroborated=False, is_annotated=False)

    fig.suptitle(
        "SpO2-corroboration audit: three concrete cases\n"
        "spectrogram (top) + SpO2 trace (bottom); blue = PSG-annotated event window, red = objectively-detected SpO2 desaturation window",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir = PROJECT_ROOT / "results" / "audit" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spo2_corroboration_examples.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
