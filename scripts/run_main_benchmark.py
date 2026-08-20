#!/usr/bin/env python3
"""Run or resume the fold-wise main benchmark grid."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.evaluation import iter_grid, run_fold
from sleep_quadnet.io import load_yaml


def csv_values(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def _run_one(job: dict) -> dict:
    """Top-level, picklable worker entry point for ProcessPoolExecutor.

    Each combo is independent: run_fold's fit cache and result-dir keys are
    content-addressed, so concurrent workers racing on the same fit_key at
    worst duplicate a fit into a retry-suffixed directory rather than
    corrupting anything (see evaluation.py's _new_run_dir/_completed_key
    scheme). This function only re-packages run_fold's kwargs so the call
    is a single picklable argument.
    """
    representation, classifier, protocol, fold = job["combo"]
    result = run_fold(
        config=job["config"], manifest_path=job["manifest"], fold_path=job["fold_file"], cache_root=job["cache_root"],
        results_root=job["results_root"], representation=representation, classifier=classifier,
        protocol=protocol, fold=fold, preprocessing_override=job["preprocessing"],
        filter_uncorroborated_training=job["filter_uncorroborated_training"],
    )
    return {"representation": representation, "classifier": classifier, "protocol": protocol, "fold": fold, "status": result["status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    parser.add_argument("--fold-file", type=Path, default=PROJECT_ROOT / "metadata" / "subject_folds_5cv_aligned.csv")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "cached_features")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "P0_device_gap")
    parser.add_argument("--representations", default="classical")
    parser.add_argument("--classifiers", default="svm_rbf")
    parser.add_argument("--protocols", default="R_R,S_S,R_S,S_R")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--preprocessing", choices=["raw", "peak", "filter", "peak_filter"])
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of fold-runs to fit concurrently (separate processes). "
             "Each worker can itself use multiple threads (RF/XGBoost n_jobs, BLAS via "
             "OMP_NUM_THREADS/MKL_NUM_THREADS) -- size --workers so workers * threads-per-worker "
             "<= the job's allocated CPUs. Default 1 preserves the original sequential behavior.",
    )
    parser.add_argument(
        "--filter-uncorroborated-training", action="store_true",
        help="Drop positive training/validation windows whose annotated event has no corroborating "
             "SpO2 desaturation (metadata/window_corroboration.csv, from "
             "scripts/build_window_corroboration.py). The test set is never filtered. Use a dedicated "
             "--results-root (never results/P0_device_gap) so this ablation can't collide with the "
             "baseline benchmark's dedup keys.",
    )
    args = parser.parse_args()
    if args.filter_uncorroborated_training and args.results_root.name == "P0_device_gap":
        raise SystemExit(
            "--filter-uncorroborated-training must use a dedicated --results-root, not P0_device_gap: "
            "mixing filtered and unfiltered runs under the same (representation, classifier, protocol, fold) "
            "key would silently corrupt load_completed_runs()'s dedup logic for the baseline benchmark."
        )
    config = load_yaml(args.config)
    jobs = list(
        iter_grid(
            config,
            csv_values(args.representations),
            csv_values(args.classifiers),
            csv_values(args.protocols),
            csv_values(args.folds, int),
        )
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    completed = skipped = 0

    if args.workers <= 1:
        for combo in tqdm(jobs, desc="benchmark", unit="fold-run", dynamic_ncols=True):
            result = run_fold(
                config=config, manifest_path=args.manifest, fold_path=args.fold_file, cache_root=args.cache_root,
                results_root=args.results_root, representation=combo[0], classifier=combo[1],
                protocol=combo[2], fold=combo[3], preprocessing_override=args.preprocessing,
                filter_uncorroborated_training=args.filter_uncorroborated_training,
            )
            status = result["status"]
            completed += status != "skipped_complete"
            skipped += status == "skipped_complete"
            tqdm.write(json.dumps({"representation": combo[0], "classifier": combo[1], "protocol": combo[2], "fold": combo[3], "status": status}))
    else:
        payloads = [
            {
                "combo": combo, "config": config, "manifest": args.manifest, "fold_file": args.fold_file,
                "cache_root": args.cache_root, "results_root": args.results_root, "preprocessing": args.preprocessing,
                "filter_uncorroborated_training": args.filter_uncorroborated_training,
            }
            for combo in jobs
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_one, payload) for payload in payloads]
            with tqdm(total=len(futures), desc="benchmark", unit="fold-run", dynamic_ncols=True) as bar:
                for future in as_completed(futures):
                    record = future.result()
                    completed += record["status"] != "skipped_complete"
                    skipped += record["status"] == "skipped_complete"
                    tqdm.write(json.dumps(record))
                    bar.update(1)

    print(json.dumps({"total": len(jobs), "completed": completed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
