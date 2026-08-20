#!/usr/bin/env python3
"""First cut at the sliding-window AHI/severity target: binary "severe vs.
not-severe" classification of 5-minute whole-night epochs from HuBERT
audio features, using ODI/hypoxic-burden-derived ground truth (not
annotation-privileged windows -- see scripts/build_sliding_window_ahi_targets.py).

Binarized (severe vs {normal,mild,moderate}) rather than full 4-class, so
this reuses the existing binary build_estimator/select_estimator/
probability/metrics machinery in evaluation.py unchanged -- same GPU-only
classifiers, same calibration handling, same everything. Full 4-class
severity is a natural follow-up once this baseline is established.

Subject-disjoint 5-fold CV, same fold assignments as the main pipeline
(metadata/subject_folds_5cv_aligned.csv) for consistency, same 4 device
protocols (R_R/S_S/R_S/S_R). Epochs with excluded_from_training=True
(>=50% awake) are dropped from all splits -- they're not meaningful
"did an apnea happen here" examples.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.evaluation import (
    GPU_ACCELERATED_CLASSIFIERS,
    build_estimator,
    metrics,
    probability,
    protocol_devices,
    select_estimator,
)
from sleep_quadnet.io import config_hash, file_sha256, load_yaml


def load_data(manifest_path: Path, cache_root: Path):
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    manifest = manifest[~manifest["excluded_from_training"]].reset_index(drop=True)
    features = np.load(cache_root / "hubert" / "peak" / "features.npy", mmap_mode="r")
    sample_ids = json.loads((cache_root / "hubert" / "peak" / "sample_ids.json").read_text(encoding="utf-8"))
    id_to_row = {sid: i for i, sid in enumerate(sample_ids)}
    feature_index = manifest["sample_id"].map(id_to_row)
    if feature_index.isna().any():
        raise ValueError("Some manifest rows have no matching cached feature row")
    manifest["feature_row"] = feature_index.astype(int)
    manifest["label"] = (manifest["severity_bin"] == "severe").astype(int)
    return manifest, features


def split_indices(manifest: pd.DataFrame, folds: pd.DataFrame, fold: int, protocol: str):
    train_devices, test_devices = protocol_devices(protocol)
    fold_map = dict(zip(folds["subject_id"], folds["fold"]))
    manifest = manifest.assign(cv_fold=manifest["subject_id"].map(fold_map))
    test_subjects = set(manifest.loc[manifest["cv_fold"] == fold, "subject_id"])
    train_val_subjects = sorted(set(manifest["subject_id"]) - test_subjects)
    rng = np.random.RandomState(1000 + fold)
    rng.shuffle(train_val_subjects)
    n_val = max(1, int(round(len(train_val_subjects) * 0.2)))
    val_subjects = set(train_val_subjects[:n_val])
    train_subjects = set(train_val_subjects[n_val:])

    def rows_for(subjects: set, devices: set) -> np.ndarray:
        mask = manifest["subject_id"].isin(subjects) & manifest["device"].isin(devices)
        return manifest.index[mask].to_numpy()

    train_idx = rows_for(train_subjects, train_devices)
    val_idx = rows_for(val_subjects, train_devices)
    test_idx = rows_for(test_subjects, test_devices)
    return train_idx, val_idx, test_idx, {
        "train_subjects": sorted(train_subjects), "val_subjects": sorted(val_subjects), "test_subjects": sorted(test_subjects),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata" / "sliding_window_audio_manifest.csv")
    parser.add_argument("--fold-file", type=Path, default=PROJECT_ROOT / "metadata" / "subject_folds_5cv_aligned.csv")
    parser.add_argument("--cache-root", type=Path, default=Path("/scratch/pkdas/IEEE_healthcomm_workshop/cached_features_sliding_window"))
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "P3_sliding_window_severity")
    parser.add_argument("--classifiers", default="svm_rbf,mlp,random_forest,xgboost")
    parser.add_argument("--protocols", default="R_R,S_S,R_S,S_R")
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["project_root"] = str(PROJECT_ROOT)
    manifest, features = load_data(args.manifest, args.cache_root)
    folds_frame = pd.read_csv(args.fold_file, dtype={"subject_id": str})
    print(f"epochs after excluding awake-dominant: {len(manifest)}, positive (severe) rate: {manifest['label'].mean():.3f}")

    args.results_root.mkdir(parents=True, exist_ok=True)
    runs_dir = args.results_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for protocol in args.protocols.split(","):
        for fold in [int(f) for f in args.folds.split(",")]:
            train_idx, val_idx, test_idx, subject_sets = split_indices(manifest, folds_frame, fold, protocol)
            if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
                print(f"skip {protocol} fold={fold}: empty split")
                continue
            x_train, x_val, x_test = features[train_idx], features[val_idx], features[test_idx]
            y_train = manifest.loc[train_idx, "label"].to_numpy()
            y_val = manifest.loc[val_idx, "label"].to_numpy()
            y_test = manifest.loc[test_idx, "label"].to_numpy()
            if len(set(y_train)) < 2 or len(set(y_val)) < 2 or len(set(y_test)) < 2:
                print(f"skip {protocol} fold={fold}: single-class split")
                continue
            for classifier in args.classifiers.split(","):
                gpu_tag = ("gpu_v1",) if classifier in GPU_ACCELERATED_CLASSIFIERS else ()
                key = config_hash("sliding_window_severity_v1", file_sha256(args.manifest), file_sha256(args.fold_file), config, protocol, fold, classifier, *gpu_tag)
                run_dir = runs_dir / key
                if (run_dir / "completion.json").exists():
                    print(f"skip (complete) {protocol} fold={fold} classifier={classifier}")
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                candidates = config["classifiers"][classifier]
                selected, tuning = select_estimator(classifier, candidates, int(config["seed"]) + fold * 100, x_train, y_train, x_val, y_val)
                estimator = build_estimator(classifier, selected, int(config["seed"]) + fold * 1000)
                estimator.fit(np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val]))
                probs = probability(estimator, x_test)
                record = {
                    "status": "complete", "experiment_key": key, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "protocol": protocol, "fold": fold, "classifier": classifier, "feature": "hubert",
                    "target": "severity_bin==severe (binarized)", "selected_hyperparameters": selected,
                    "subjects": subject_sets, "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
                    "metrics": metrics(y_test, probs), "runtime_sec": time.perf_counter() - started,
                }
                (run_dir / "completion.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                results.append(record)
                print(f"{protocol} fold={fold} {classifier}: BA={record['metrics']['balanced_accuracy']:.4f} F1={record['metrics']['f1']:.4f}")

    print(json.dumps({"completed_this_run": len(results)}, indent=2))


if __name__ == "__main__":
    main()
