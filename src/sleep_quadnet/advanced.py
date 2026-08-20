"""Fold-local PCA and CORAL experiments with strict subject isolation."""

from __future__ import annotations

import csv
import gzip
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

from .evaluation import (
    GPU_ACCELERATED_CLASSIFIERS,
    MASTER_FIELDS,
    build_estimator,
    load_representation,
    metrics,
    parse_folds,
    probability,
    protocol_devices,
    select_estimator,
    split_indices,
    take_features,
)
from .io import append_csv, config_hash, file_sha256, read_csv_rows


def _calibration_safe_candidates(classifier: str, candidates: list[dict]) -> list[dict]:
    """Strip svm_rbf's probability=True (Platt scaling) for the PCA/CORAL paths.

    Diagnosed empirically (2026-08-18): on the PCA-reduced (384-1536d) and
    CORAL-aligned feature spaces, SVC's internal Platt-scaling calibration
    (fit via a 5-fold CV on the training data) collapsed every test-set
    probability into a narrow band just under 0.5 (observed range
    0.447-0.484 on one run), so metrics()'s fixed >=0.5 threshold predicted
    the negative class 100% of the time -- despite ROC-AUC of 0.51-0.61
    showing the underlying ranking still carried real signal. This is a
    known SVM failure mode: Platt's separately-fit sigmoid can drift away
    from the SVM's own decision boundary when the classes are only weakly
    separable, which the higher-dimensional (768-3840d), non-PCA-reduced
    main-benchmark/ablation representations apparently give the SVM enough
    separation to avoid.

    Without probability=True, evaluation.py:probability() falls back to
    decision_function() + a manual sigmoid, which maps decision_function==0
    (the SVM's actual, class_weight-adjusted decision boundary) to exactly
    probability==0.5 -- consistent with metrics()'s fixed threshold, and
    unaffected by a separately-miscalibrated Platt curve. This only changes
    how a decision score is turned into a probability for thresholding; it
    does not change what boundary the SVM itself learns (C/gamma/class_weight
    are untouched). Only applied within advanced.py (PCA/CORAL) -- left
    untouched for run_main_benchmark's P0_device_gap/P0_ablation, which do
    not show this collapse on their un-reduced feature spaces.
    """
    if classifier != "svm_rbf":
        return candidates
    return [{**candidate, "probability": False} for candidate in candidates]


def _safe_run_dir(root: Path, key: str) -> tuple[Path, bool]:
    base = root / "runs" / key
    completion = base / "completion.json"
    if completion.exists() and json.loads(completion.read_text(encoding="utf-8")).get("status") == "complete":
        return base, True
    try:
        base.mkdir(parents=True)
        return base, False
    except FileExistsError:
        pass
    # A concurrent worker (e.g. run_dimension_control.py/run_coral.py under
    # xargs -P) won the race on `base` between the exists()/mkdir() check
    # above -- fall through to a retry-suffixed directory rather than
    # crashing, mirroring evaluation.py::_new_run_dir's fix for the same
    # pattern.
    retry = root / "runs" / f"{key}_retry_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    retry.mkdir(parents=True)
    return retry, False


def _write_predictions(run_dir: Path, run_id: str, rows: list[dict], indices: np.ndarray, probabilities: np.ndarray, context: dict) -> None:
    labels = np.asarray([int(rows[int(i)]["label"]) for i in indices])
    predictions = (probabilities >= 0.5).astype(np.int8)
    with gzip.open(run_dir / "window_predictions.csv.gz", "xt", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "fold", "protocol", "representation", "classifier", "sample_id", "logical_window_id", "subject_id", "device", "label", "probability", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, index in enumerate(indices):
            row = rows[int(index)]
            writer.writerow(
                {"run_id": run_id, **context, "sample_id": row["sample_id"], "logical_window_id": row["logical_window_id"],
                 "subject_id": row["subject_id"], "device": row["device"], "label": int(labels[position]),
                 "probability": float(probabilities[position]), "prediction": int(predictions[position])}
            )


def _log(config: dict, record: dict, run_dir: Path, subject_sets: dict) -> None:
    train_devices, test_devices = protocol_devices(record["protocol"])
    append_csv(
        Path(config["logging"]["master_log"]),
        [{
            "run_id": record["run_id"], "timestamp": record["timestamp"], "git_commit": config["logging"]["git_commit_fallback"],
            "seed": record["seed"], "fold": record["fold"], "train_subjects": json.dumps(subject_sets["train_subjects"]),
            "val_subjects": json.dumps(subject_sets["val_subjects"]), "test_subjects": json.dumps(subject_sets["test_subjects"]),
            "device_train": "+".join(sorted(train_devices)), "device_test": "+".join(sorted(test_devices)),
            "representation": record["representation"], "classifier": record["classifier"],
            "hyperparameters": json.dumps(record["selected_hyperparameters"], sort_keys=True),
            "preprocessing": record.get("preprocessing", "peak"), "feature_dimension": record["feature_dimension"],
            "metrics": json.dumps(record["metrics"], sort_keys=True), "runtime": json.dumps(record["runtime"], sort_keys=True),
            "hardware": json.dumps(record["hardware"], sort_keys=True), "experiment_key": record["experiment_key"], "result_dir": str(run_dir),
        }],
        MASTER_FIELDS,
    )


def run_pca_fold(
    *, config: dict, manifest_path: Path, fold_path: Path, cache_root: Path, results_root: Path,
    protocol: str, fold: int, classifier: str = "svm_rbf", dimensions: tuple[int, ...] = (1536, 768, 384),
    representation: str = "full_fusion",
) -> list[dict]:
    rows = read_csv_rows(manifest_path)
    folds = parse_folds(fold_path)
    train_idx, val_idx, test_idx, subject_sets = split_indices(rows, folds, fold, protocol)
    arrays, native_dimension = load_representation(cache_root, representation, config)
    x_train = take_features(arrays, train_idx)
    x_val = take_features(arrays, val_idx)
    x_test = take_features(arrays, test_idx)
    y_train = np.asarray([int(rows[int(i)]["label"]) for i in train_idx], dtype=np.int8)
    y_val = np.asarray([int(rows[int(i)]["label"]) for i in val_idx], dtype=np.int8)
    y_test = np.asarray([int(rows[int(i)]["label"]) for i in test_idx], dtype=np.int8)
    max_dimension = max(dimensions)
    if max_dimension >= min(x_train.shape):
        raise ValueError(f"PCA dimension {max_dimension} invalid for train shape {x_train.shape}")
    # Target-aware refit fix (2026-08-19): for cross-device protocols, a PCA
    # fit only on source-device train+val data collapses cross-device test
    # predictions to a single class (root-caused and confirmed via smoke
    # test: BA +3.37pt from this fix on one combo). The refit-stage PCA --
    # the one that actually produces z_test -- now also sees unlabeled
    # target-device validation features, exactly mirroring how
    # run_coral_fold already builds its target covariance from
    # target_val_idx below. Matched-device protocols (R_R/S_S) have no
    # cross-device gap to bridge, so target_val_idx is empty there and the
    # refit fit is unchanged from before.
    train_devices, test_devices = protocol_devices(protocol)
    target_devices = test_devices - train_devices
    val_subjects = set(subject_sets["val_subjects"])
    target_val_idx = np.asarray(
        [i for i, row in enumerate(rows) if row["subject_id"] in val_subjects and row["device"] in target_devices],
        dtype=np.int64,
    )
    x_target_val = take_features(arrays, target_val_idx) if len(target_val_idx) else np.empty((0, x_train.shape[1]), dtype=x_train.dtype)
    started = time.perf_counter()
    tuning_pca = PCA(n_components=max_dimension, svd_solver="randomized", iterated_power=2, random_state=int(config["seed"]) + fold)
    z_train = tuning_pca.fit_transform(x_train)
    z_val = tuning_pca.transform(x_val)
    candidates = _calibration_safe_candidates(classifier, config["classifiers"][classifier])
    selected_by_dimension = {}
    tuning_by_dimension = {}
    for dimension in dimensions:
        selected, tuning = select_estimator(
            classifier, candidates, int(config["seed"]) + fold * 100,
            z_train[:, :dimension], y_train, z_val[:, :dimension], y_val,
        )
        selected_by_dimension[dimension] = selected
        tuning_by_dimension[dimension] = tuning
    x_fit = np.concatenate([x_train, x_val])
    y_fit = np.concatenate([y_train, y_val])
    refit_pca = PCA(n_components=max_dimension, svd_solver="randomized", iterated_power=2, random_state=int(config["seed"]) + fold + 1000)
    refit_pca.fit(np.concatenate([x_fit, x_target_val]) if len(target_val_idx) else x_fit)
    z_fit = refit_pca.transform(x_fit)
    z_test = refit_pca.transform(x_test)
    results = []
    for dimension in dimensions:
        # representation is included in the key so full_fusion_v2 (or any
        # other base representation) gets its own experiment identity rather
        # than colliding with existing full_fusion PCA runs, which predate
        # this parameter and would otherwise share a hash (config_hash did
        # not previously vary by representation here, since only "full_fusion"
        # was ever passed). Salt bumped v1->v2 for the probability=True Platt-
        # scaling fix above (see _calibration_safe_candidates) -- the old v1
        # keys' completion.json files are all degenerate (F1=0, BA=0.5, 100%
        # negative predictions despite real ROC-AUC) and must not be treated
        # as already-complete under the fixed logic.
        # gpu_tag mirrors evaluation.py::run_fold's own mechanism (bug fix,
        # 2026-08-19): without it, every existing dimension_control_v2 key --
        # all of which default to classifier="svm_rbf", a GPU_ACCELERATED
        # classifier -- would keep matching its pre-migration CPU-fit
        # completion.json forever, silently skipping the GPU rerun entirely.
        gpu_tag = ("gpu_v1",) if classifier in GPU_ACCELERATED_CLASSIFIERS else ()
        # v2->v3: target-aware refit-PCA fix above (2026-08-19) changes what
        # z_test actually is for cross-device protocols -- old v2 keys are
        # the pre-fix collapsed runs and must not be treated as complete.
        key = config_hash("dimension_control_v3", file_sha256(manifest_path), file_sha256(fold_path), config, representation, protocol, fold, classifier, dimension, *gpu_tag)
        run_dir, complete = _safe_run_dir(results_root, key)
        if complete:
            results.append({"status": "skipped_complete", "experiment_key": key, "result_dir": str(run_dir)})
            continue
        estimator = build_estimator(classifier, selected_by_dimension[dimension], int(config["seed"]) + fold * 1000)
        fit_started = time.perf_counter()
        estimator.fit(z_fit[:, :dimension], y_fit)
        probabilities = probability(estimator, z_test[:, :dimension])
        runtime = {"total_fold_pca_and_classifier_sec": time.perf_counter() - started, "classifier_refit_sec": time.perf_counter() - fit_started}
        representation_label = f"{representation}_pca_{dimension}"
        run_id = run_dir.name
        _write_predictions(run_dir, run_id, rows, test_idx, probabilities, {"fold": fold, "protocol": protocol, "representation": representation_label, "classifier": classifier})
        # Persist the fitted PCA+classifier pipeline under the shared
        # checkpoints tree (kept off /home via project_root's checkpoints
        # symlink) rather than duplicating a large binary into every run
        # directory; the run directory keeps only a small pointer.
        pipeline_dir = Path(config["project_root"]) / "checkpoints" / "pca_cache" / key
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = pipeline_dir / "pipeline.joblib"
        joblib.dump(
            {"pca_mean": refit_pca.mean_, "pca_components": refit_pca.components_[:dimension], "pca_explained_variance": refit_pca.explained_variance_[:dimension], "classifier": estimator},
            pipeline_path,
        )
        (run_dir / "pipeline_ref.json").write_text(
            json.dumps({"experiment_key": key, "pipeline_path": str(pipeline_path)}, indent=2), encoding="utf-8"
        )
        record = {
            "status": "complete", "experiment_key": key, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": int(config["seed"]), "fold": fold, "protocol": protocol, "representation": representation_label,
            "base_representation": representation,
            "classifier": classifier, "selected_hyperparameters": selected_by_dimension[dimension], "tuning": tuning_by_dimension[dimension],
            "native_feature_dimension": native_dimension, "feature_dimension": dimension, "preprocessing": "peak",
            "pca_fit_scope": (
                "training subjects for tuning; training+validation subjects (source) plus unlabeled "
                "target-device validation subjects (cross-device protocols only) for refit; never test"
            ),
            "target_aware_refit": bool(len(target_val_idx)),
            "target_val_rows": int(len(target_val_idx)),
            "subjects": subject_sets, "metrics": metrics(y_test, probabilities), "runtime": runtime,
            "hardware": {"host": socket.gethostname(), "platform": platform.platform(), "slurm_job_id": os.getenv("SLURM_JOB_ID", "")},
        }
        (run_dir / "completion.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        _log(config, record, run_dir, subject_sets)
        results.append(record)
    return results


def coral_transform(source: np.ndarray, target_reference: np.ndarray, regularization: float = 1e-5):
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target_reference.mean(axis=0, keepdims=True)
    source_cov = np.cov(source - source_mean, rowvar=False)
    target_cov = np.cov(target_reference - target_mean, rowvar=False)
    scale = max(float(np.trace(source_cov) / source_cov.shape[0]), float(np.trace(target_cov) / target_cov.shape[0]), 1e-12)
    source_cov += np.eye(source_cov.shape[0]) * regularization * scale
    target_cov += np.eye(target_cov.shape[0]) * regularization * scale
    source_values, source_vectors = linalg.eigh(source_cov, check_finite=False)
    target_values, target_vectors = linalg.eigh(target_cov, check_finite=False)
    source_inverse_root = (source_vectors * (1.0 / np.sqrt(np.maximum(source_values, 1e-12)))) @ source_vectors.T
    target_root = (target_vectors * np.sqrt(np.maximum(target_values, 1e-12))) @ target_vectors.T
    transform = source_inverse_root @ target_root
    return (source - source_mean) @ transform + target_mean, source_mean, target_mean, transform


def run_coral_fold(
    *, config: dict, manifest_path: Path, fold_path: Path, cache_root: Path, results_root: Path,
    representation: str, protocol: str, fold: int, classifier: str = "svm_rbf",
) -> dict:
    if protocol not in {"R_S", "S_R"}:
        raise ValueError("CORAL is only defined here for R_S and S_R")
    # Salt bumped v1->v2 for the same probability=True Platt-scaling fix as
    # run_pca_fold (see _calibration_safe_candidates) -- CORAL's aligned
    # feature space is similarly weakly-separable and would be expected to
    # hit the same calibration collapse. No v1 CORAL runs existed yet
    # (P1_domain_adaptation was 0/30 when this fix was made), so this bump
    # is precautionary/consistency-preserving rather than invalidating any
    # existing result.
    # gpu_tag: same bug fix and rationale as run_pca_fold above -- without
    # it, all 20 existing coral_v2 completions (classifier defaults to the
    # GPU_ACCELERATED "svm_rbf") would keep matching their pre-migration
    # CPU-fit completion.json forever.
    gpu_tag = ("gpu_v1",) if classifier in GPU_ACCELERATED_CLASSIFIERS else ()
    key = config_hash("coral_v2", file_sha256(manifest_path), file_sha256(fold_path), config, representation, protocol, fold, classifier, *gpu_tag)
    run_dir, complete = _safe_run_dir(results_root, key)
    if complete:
        return {"status": "skipped_complete", "experiment_key": key, "result_dir": str(run_dir)}
    rows = read_csv_rows(manifest_path)
    folds = parse_folds(fold_path)
    train_idx, val_idx, test_idx, subject_sets = split_indices(rows, folds, fold, protocol)
    source_device = "R" if protocol == "R_S" else "S"
    target_device = "S" if protocol == "R_S" else "R"
    val_subjects = set(subject_sets["val_subjects"])
    target_val_idx = np.asarray([i for i, row in enumerate(rows) if row["subject_id"] in val_subjects and row["device"] == target_device], dtype=np.int64)
    arrays, dimension = load_representation(cache_root, representation, config)
    x_train = take_features(arrays, train_idx).astype(np.float64)
    x_source_val = take_features(arrays, val_idx).astype(np.float64)
    x_target_val = take_features(arrays, target_val_idx).astype(np.float64)
    y_train = np.asarray([int(rows[int(i)]["label"]) for i in train_idx], dtype=np.int8)
    y_val = np.asarray([int(rows[int(i)]["label"]) for i in val_idx], dtype=np.int8)
    started = time.perf_counter()
    aligned_train, source_mean, target_mean, transform = coral_transform(x_train, x_target_val)
    aligned_source_val = (x_source_val - source_mean) @ transform + target_mean
    candidates = _calibration_safe_candidates(classifier, config["classifiers"][classifier])
    selected, tuning = select_estimator(
        classifier, candidates, int(config["seed"]) + fold * 100,
        aligned_train.astype(np.float32), y_train, aligned_source_val.astype(np.float32), y_val,
    )
    fit_idx = np.concatenate([train_idx, val_idx])
    x_source_fit = take_features(arrays, fit_idx).astype(np.float64)
    y_fit = np.asarray([int(rows[int(i)]["label"]) for i in fit_idx], dtype=np.int8)
    aligned_fit, source_mean, target_mean, transform = coral_transform(x_source_fit, x_target_val)
    estimator = build_estimator(classifier, selected, int(config["seed"]) + fold * 1000)
    estimator.fit(aligned_fit.astype(np.float32), y_fit)
    # Bug fix (2026-08-18): x_test previously went into probability() raw/
    # un-aligned, even though the classifier was fit on CORAL-aligned
    # (whitened-and-recolored) features -- a train/test feature-distribution
    # mismatch that would tend to make CORAL look worse than it actually is.
    # Apply the SAME (train+val-fit) source_mean/transform/target_mean used
    # to build aligned_fit above -- test features still never fit the
    # alignment (only .fit_idx/x_target_val did, via coral_transform above),
    # they are only .transform()-equivalent-ed through it here.
    x_test = take_features(arrays, test_idx).astype(np.float64)
    aligned_test = (x_test - source_mean) @ transform + target_mean
    probabilities = probability(estimator, aligned_test.astype(np.float32))
    y_test = np.asarray([int(rows[int(i)]["label"]) for i in test_idx], dtype=np.int8)
    adapted_representation = f"{representation}_coral"
    run_id = run_dir.name
    _write_predictions(run_dir, run_id, rows, test_idx, probabilities, {"fold": fold, "protocol": protocol, "representation": adapted_representation, "classifier": classifier})
    # Same rationale as run_pca_fold: keep the fitted CORAL transform +
    # classifier binary in the shared checkpoints tree (off /home), and
    # leave only a small pointer in the run directory.
    pipeline_dir = Path(config["project_root"]) / "checkpoints" / "coral_cache" / key
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = pipeline_dir / "pipeline.joblib"
    joblib.dump(
        {"source_mean": source_mean, "target_validation_mean": target_mean, "coral_transform": transform, "classifier": estimator},
        pipeline_path,
    )
    (run_dir / "pipeline_ref.json").write_text(
        json.dumps({"experiment_key": key, "pipeline_path": str(pipeline_path)}, indent=2), encoding="utf-8"
    )
    record = {
        "status": "complete", "experiment_key": key, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": int(config["seed"]), "fold": fold, "protocol": protocol, "representation": adapted_representation,
        "base_representation": representation, "classifier": classifier, "selected_hyperparameters": selected, "tuning": tuning,
        "feature_dimension": dimension, "preprocessing": "peak", "subjects": subject_sets,
        "adaptation_scope": "CORAL target covariance uses unlabeled target-device validation subjects only; test features never fit the alignment (only transformed through the train+val-fit source_mean/transform/target_mean before scoring)",
        "target_validation_device": target_device, "source_device": source_device,
        "metrics": metrics(y_test, probabilities), "runtime": {"total_sec": time.perf_counter() - started},
        "hardware": {"host": socket.gethostname(), "platform": platform.platform(), "slurm_job_id": os.getenv("SLURM_JOB_ID", "")},
    }
    (run_dir / "completion.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    _log(config, record, run_dir, subject_sets)
    return record
