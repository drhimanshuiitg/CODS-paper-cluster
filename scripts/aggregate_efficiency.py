#!/usr/bin/env python3
"""Combine measured component/fusion cost with cached storage and main performance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.features import FEATURE_DIMENSIONS, cache_paths
from sleep_quadnet.io import load_completed_runs, load_yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_utils import annotate_scatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results" / "P0_efficiency")
    args = parser.parse_args()
    config = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    selection = json.loads((PROJECT_ROOT / "results" / "P0_statistics" / "selection.json").read_text(encoding="utf-8"))
    primary_classifier = selection["primary_classifier"]
    representations = ["classical", "hubert", "wavlm", "wav2vec2", "data2vec_audio", "data2vec_spectrogram", "data2vec_fusion", "full_fusion"]
    measured = {}
    for path in (args.root / "component_runs").glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        measured[item["feature"]] = item
    for path in (args.root / "representation_runs").glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        measured[item["representation"]] = item
    missing = set(representations) - set(measured)
    if missing:
        raise FileNotFoundError(f"Missing efficiency measurements: {sorted(missing)}")
    gap = pd.read_csv(PROJECT_ROOT / "results" / "P0_device_gap" / "device_gap_summary.csv")
    gap = gap[gap["classifier"] == primary_classifier].set_index("representation")
    # Dedup by (representation, classifier, protocol, fold) before filtering
    # to primary_classifier -- otherwise a duplicated combo (stale CPU-era
    # run alongside its GPU-era replacement) would double-count that fold's
    # inference latency below. See load_completed_runs()'s docstring.
    main_records = [
        item for item in load_completed_runs(PROJECT_ROOT / "results" / "P0_device_gap", ("representation", "classifier", "protocol", "fold"))
        if item["classifier"] == primary_classifier
    ]
    rows = []
    for representation in representations:
        item = measured[representation]
        components = config["representations"][representation]
        storage = 0
        for component in components:
            preprocessing = "peak_filter" if component == "classical" else "peak"
            storage += cache_paths(PROJECT_ROOT / "cached_features", component, preprocessing)["features"].stat().st_size
        classifier_times = [record["runtime"]["test_inference_sec_per_clip"] for record in main_records if record["representation"] == representation]
        classifier_latency = sum(classifier_times) / len(classifier_times)
        rows.append(
            {"representation": representation, "feature_dimension": item["feature_dimension"], "encoder_branches": len(components),
             "feature_extraction_latency_mean_sec": item["latency_mean_sec"], "feature_extraction_latency_std_sec": item["latency_std_sec"],
             "downstream_classifier_latency_mean_sec": classifier_latency,
             "total_deployable_latency_sec": item["latency_mean_sec"] + classifier_latency,
             "clips_per_second": item["clips_per_second"], "real_time_factor": item["real_time_factor_mean"],
             "peak_gpu_memory_bytes": item["peak_gpu_memory_bytes"], "peak_cpu_rss_bytes": item["cpu_peak_rss_bytes"],
             "cached_feature_size_bytes": storage, "cross_device_f1": gap.loc[representation, "f1_mean_cross_device"],
             "cross_device_balanced_accuracy": gap.loc[representation, "balanced_accuracy_mean_cross_device"],
             "classifier": primary_classifier, "timed_clips": item["clips"], "warmup_clips": item["warmup_clips"], "batch_size": item["batch_size"]}
        )
    frame = pd.DataFrame(rows)
    args.root.mkdir(parents=True, exist_ok=True)
    for filename, columns in (
        ("runtime_benchmark.csv", [column for column in frame if "latency" in column or column in {"representation", "clips_per_second", "real_time_factor", "timed_clips", "warmup_clips", "batch_size"}]),
        ("memory_benchmark.csv", ["representation", "peak_gpu_memory_bytes", "peak_cpu_rss_bytes"]),
        ("representation_size.csv", ["representation", "feature_dimension", "encoder_branches", "cached_feature_size_bytes"]),
    ):
        target = args.root / filename
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
        frame[columns].to_csv(target, index=False)
    table_path = PROJECT_ROOT / "results" / "tables" / "table4_deployment_cost.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {table_path}")
    frame.to_csv(table_path, index=False)
    figure_path = PROJECT_ROOT / "figures" / "performance_vs_latency.pdf"
    if figure_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {figure_path}")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.set(xlabel="Total latency per clip (s)", ylabel="Mean cross-device balanced accuracy", ylim=(0, 1))
    annotate_scatter(axis, frame["total_deployable_latency_sec"], frame["cross_device_balanced_accuracy"], frame["representation"])
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    print(frame.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
