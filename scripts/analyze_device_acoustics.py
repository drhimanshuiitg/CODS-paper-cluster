#!/usr/bin/env python3
"""Paired recorder/smartphone acoustic-domain characterization (P1-A)."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.io import load_manifest_window, load_yaml, read_csv_rows

BANDS = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000))
_CONFIG: dict | None = None


def init_worker(config: dict) -> None:
    global _CONFIG
    _CONFIG = config


def descriptors(audio: np.ndarray, sample_rate: int) -> tuple[dict, np.ndarray, np.ndarray]:
    frequencies, psd = signal.welch(audio, fs=sample_rate, window="hann", nperseg=1024, noverlap=512, scaling="density")
    psd = np.maximum(psd.astype(np.float64), np.finfo(np.float64).tiny)
    total = np.trapz(psd, frequencies)
    weights = psd / psd.sum()
    centroid = float(np.sum(frequencies * weights))
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * weights)))
    cumulative = np.cumsum(psd)
    rolloff = float(frequencies[min(np.searchsorted(cumulative, 0.85 * cumulative[-1]), len(frequencies) - 1)])
    flatness = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    frame_length = max(1, int(round(0.5 * sample_rate)))
    frame_rms = [np.sqrt(np.mean(np.square(audio[start : start + frame_length], dtype=np.float64))) for start in range(0, len(audio), frame_length) if len(audio[start : start + frame_length])]
    noise_floor = float(np.percentile(frame_rms, 10))
    snr_proxy = float(20 * np.log10((rms + 1e-12) / (noise_floor + 1e-12)))
    record = {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff85_hz": rolloff,
        "spectral_flatness": flatness,
        "rms": rms,
        "mean_square_energy": float(np.mean(np.square(audio, dtype=np.float64))),
        "dynamic_range_p995_p005": float(np.percentile(audio, 99.5) - np.percentile(audio, 0.5)),
        "noise_floor_proxy_rms_p10_frames": noise_floor,
        "snr_proxy_db_rms_to_p10_frame": snr_proxy,
        "integrated_psd": float(total),
    }
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high if high < sample_rate / 2 else frequencies <= high)
        record[f"band_energy_{low}_{high}_hz"] = float(np.trapz(psd[mask], frequencies[mask])) if mask.sum() > 1 else 0.0
    return record, frequencies, psd


def process_pair(pair: tuple[dict, dict]):
    if _CONFIG is None:
        raise RuntimeError("Worker not initialized")
    output = []
    psds = []
    frequencies = None
    for row in pair:
        audio, sample_rate = load_manifest_window(row, _CONFIG, "raw")
        values, frequencies, psd = descriptors(audio, sample_rate)
        output.append(
            {"logical_window_id": row["logical_window_id"], "subject_id": row["subject_id"], "device": row["device"],
             "label": int(row["label"]), "duration_sec": float(row["duration_sec"]), **values}
        )
        psds.append(psd)
    return output, frequencies, psds


def exclusive_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite acoustic result: {path}")
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "P1_device_acoustics")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_yaml(args.config)
    rows = read_csv_rows(args.manifest)
    paired: dict[str, dict[str, dict]] = {}
    for row in rows:
        paired.setdefault(row["logical_window_id"], {})[row["device"]] = row
    tasks = [(devices["R"], devices["S"]) for _, devices in sorted(paired.items())]
    metric_rows = []
    psd_subject_sums = {"R": {}, "S": {}}
    psd_subject_counts = {"R": {}, "S": {}}
    frequencies = None
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(config,)) as executor:
        iterator = executor.map(process_pair, tasks, chunksize=8)
        for output, frequencies, psds in tqdm(iterator, total=len(tasks), desc="paired device acoustics", unit="window", dynamic_ncols=True):
            metric_rows.extend(output)
            subject = output[0]["subject_id"]
            for device, psd in zip(("R", "S"), psds):
                if subject not in psd_subject_sums[device]:
                    psd_subject_sums[device][subject] = psd.copy()
                    psd_subject_counts[device][subject] = 1
                else:
                    psd_subject_sums[device][subject] += psd
                    psd_subject_counts[device][subject] += 1
    frame = pd.DataFrame(metric_rows)
    band_columns = [f"band_energy_{low}_{high}_hz" for low, high in BANDS]
    band_energy = frame[["logical_window_id", "subject_id", "device", "label", *band_columns]].copy()
    metric_columns = [column for column in frame.columns if column not in {"logical_window_id", "subject_id", "device", "label", "duration_sec"}]
    subject_means = frame.groupby(["subject_id", "device"], as_index=False)[metric_columns].mean()
    paired_subject = subject_means.pivot(index="subject_id", columns="device", values=metric_columns)
    paired_rows = []
    for metric in metric_columns:
        recorder = paired_subject[(metric, "R")].to_numpy()
        smartphone = paired_subject[(metric, "S")].to_numpy()
        difference = recorder - smartphone
        standard_error = stats.sem(difference)
        interval = stats.t.interval(0.95, len(difference) - 1, loc=np.mean(difference), scale=standard_error)
        t_result = stats.ttest_rel(recorder, smartphone)
        try:
            wilcoxon_p = float(stats.wilcoxon(recorder, smartphone).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        paired_rows.append(
            {"metric": metric, "subjects": len(difference), "recorder_mean": np.mean(recorder), "smartphone_mean": np.mean(smartphone),
             "paired_difference_R_minus_S": np.mean(difference), "difference_std": np.std(difference, ddof=1),
             "ci95_low": interval[0], "ci95_high": interval[1], "paired_t_p_value": t_result.pvalue,
             "wilcoxon_p_value": wilcoxon_p, "unit": "subject_mean"}
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    exclusive_csv(frame, args.output_root / "spectral_statistics.csv")
    exclusive_csv(pd.DataFrame(paired_rows), args.output_root / "paired_device_statistics.csv")
    exclusive_csv(band_energy, args.output_root / "band_energy.csv")
    methods = {
        "native_sample_rate_hz": int(config["audio"]["raw_sample_rate"]),
        "analysis_sample_rate_hz": int(config["audio"]["target_sample_rate"]),
        "analysis_nyquist_hz": int(config["audio"]["target_sample_rate"]) / 2,
        "native_information_nyquist_hz": int(config["audio"]["raw_sample_rate"]) / 2,
        "resampling": config["audio"]["resampling"],
        "normalization": "none (raw PCM amplitude preserved)",
        "snr_note": "RMS-to-low-energy-frame proxy; not absolute SNR",
        "paired_logical_windows": len(tasks),
        "paired_subjects": int(frame["subject_id"].nunique()),
    }
    (args.output_root / "methods.json").write_text(json.dumps(methods, indent=2), encoding="utf-8")
    if frequencies is None:
        raise RuntimeError("No PSDs computed")
    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    psd_figure_path = figure_dir / "mean_psd_recorder_vs_smartphone.pdf"
    band_figure_path = figure_dir / "device_band_energy.pdf"
    if psd_figure_path.exists() or band_figure_path.exists():
        raise FileExistsError("Refusing to overwrite an existing device-acoustics figure")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for device, label in (("R", "Recorder"), ("S", "Smartphone")):
        subject_psds = [total / psd_subject_counts[device][subject] for subject, total in psd_subject_sums[device].items()]
        mean_psd = np.mean(subject_psds, axis=0)
        axis.semilogy(frequencies, mean_psd, label=label)
    axis.set(xlabel="Frequency (Hz)", ylabel="Mean PSD", xlim=(0, 8000))
    axis.axvline(4000, color="gray", linestyle="--", linewidth=0.8, label="Native 8-kHz Nyquist")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(psd_figure_path, bbox_inches="tight")
    plt.close(figure)
    subject_band = frame.groupby(["subject_id", "device"], as_index=False)[band_columns].mean()
    means = subject_band.groupby("device")[band_columns].mean()
    errors = subject_band.groupby("device")[band_columns].sem()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(BANDS))
    width = 0.36
    for offset, (device, label) in zip((-width / 2, width / 2), (("R", "Recorder"), ("S", "Smartphone"))):
        axis.bar(x + offset, means.loc[device], width, yerr=errors.loc[device], label=label, capsize=2)
    axis.set_xticks(x, [f"{low}-{high}" for low, high in BANDS])
    axis.set(xlabel="Frequency band (Hz)", ylabel="Mean integrated PSD (subject means)")
    axis.set_yscale("log")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(band_figure_path, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(methods, indent=2))


if __name__ == "__main__":
    main()
