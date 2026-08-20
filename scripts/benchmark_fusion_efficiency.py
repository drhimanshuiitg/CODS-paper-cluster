#!/usr/bin/env python3
"""Measure end-to-end sequential multi-branch fusion latency with all models resident."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark_efficiency import representative_rows, run_once
from sleep_quadnet.features import FEATURE_DIMENSIONS, _load_model
from sleep_quadnet.io import load_yaml, read_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["data2vec_fusion", "full_fusion"], required=True)
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    config = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    components = config["representations"][args.representation]
    rows = representative_rows(read_csv_rows(PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv"), args.clips)
    device = torch.device("cuda")
    models = {}
    for component in components:
        models[component] = _load_model(component, device, local_files_only=True)

    def extract(row):
        vectors = []
        for component in components:
            _, processor, model, kind = models[component]
            vectors.append(run_once(row, component, config, processor, model, kind, device))
        return np.concatenate(vectors)

    for row in rows[: args.warmup]:
        extract(row)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    timings = []
    durations = []
    for row in tqdm(rows, desc=f"benchmark {args.representation}", unit="clip"):
        start = time.perf_counter()
        vector = extract(row)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        durations.append(float(row["duration_sec"]))
        if vector.shape != (sum(FEATURE_DIMENSIONS[name] for name in components),):
            raise ValueError("Unexpected fusion vector size")
    rss_after = process.memory_info().rss
    record = {
        "representation": args.representation, "components": components,
        "feature_dimension": sum(FEATURE_DIMENSIONS[name] for name in components), "encoders": len(components),
        "clips": len(rows), "warmup_clips": min(args.warmup, len(rows)), "batch_size": 1,
        "latency_mean_sec": statistics.mean(timings), "latency_std_sec": statistics.stdev(timings),
        "clips_per_second": len(timings) / sum(timings),
        "real_time_factor_mean": statistics.mean(value / duration for value, duration in zip(timings, durations)),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "cpu_rss_before_bytes": rss_before, "cpu_rss_after_bytes": rss_after, "cpu_rss_delta_bytes": rss_after - rss_before,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "timings_sec": timings, "audio_durations_sec": durations,
    }
    root = PROJECT_ROOT / "results" / "P0_efficiency" / "representation_runs"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{args.representation}.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {target}")
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
