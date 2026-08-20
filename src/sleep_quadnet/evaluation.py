"""Leakage-safe, fold-wise downstream evaluation on cached representations."""

from __future__ import annotations

import csv
import gzip
import json
import os
import pickle
import platform
import socket
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_DIMENSIONS, cache_paths
from .io import append_csv, config_hash, file_sha256, read_csv_rows


PRIMARY_METRICS = ("f1", "balanced_accuracy", "roc_auc", "mcc", "sensitivity", "specificity")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_folds(path: Path) -> dict[str, int]:
    rows = read_csv_rows(path)
    result = {row["subject_id"].zfill(2): int(row["fold"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate subjects in fold file")
    return result


def protocol_devices(protocol: str) -> tuple[set[str], set[str]]:
    mapping = {
        "R_R": ({"R"}, {"R"}),
        "S_S": ({"S"}, {"S"}),
        "R_S": ({"R"}, {"S"}),
        "S_R": ({"S"}, {"R"}),
        "RS_RS": ({"R", "S"}, {"R", "S"}),
    }
    if protocol not in mapping:
        raise ValueError(f"Unknown protocol: {protocol}")
    return mapping[protocol]


def split_indices(
    rows: Sequence[dict], folds: dict[str, int], fold: int, protocol: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    train_devices, test_devices = protocol_devices(protocol)
    test_subjects = {subject for subject, value in folds.items() if value == fold}
    val_subjects = {subject for subject, value in folds.items() if value == (fold + 1) % 5}
    train_subjects = set(folds) - test_subjects - val_subjects
    train = np.asarray(
        [i for i, row in enumerate(rows) if row["subject_id"] in train_subjects and row["device"] in train_devices],
        dtype=np.int64,
    )
    val = np.asarray(
        [i for i, row in enumerate(rows) if row["subject_id"] in val_subjects and row["device"] in train_devices],
        dtype=np.int64,
    )
    test = np.asarray(
        [i for i, row in enumerate(rows) if row["subject_id"] in test_subjects and row["device"] in test_devices],
        dtype=np.int64,
    )
    subject_sets = {
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
    }
    if train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects:
        raise AssertionError("Subject leakage in split construction")
    for name, indices in (("train", train), ("validation", val), ("test", test)):
        if not len(indices) or len({rows[i]["label"] for i in indices}) != 2:
            raise ValueError(f"{name} split is empty or single-class: fold={fold}, protocol={protocol}")
    return train, val, test, subject_sets


def _load_window_corroboration(config: dict) -> dict[str, bool]:
    """sample_id -> spo2_corroborated, from metadata/window_corroboration.csv
    (scripts/build_window_corroboration.py). Negatives are always True
    (kept) by construction of that file; only positives can be False."""
    path = Path(config["project_root"]) / "metadata" / "window_corroboration.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/build_window_corroboration.py first")
    lookup: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lookup[row["sample_id"]] = row["spo2_corroborated"] == "True"
    return lookup


def filter_uncorroborated(
    train_idx: np.ndarray, val_idx: np.ndarray, rows: Sequence[dict], corroboration_lookup: dict[str, bool]
) -> tuple[np.ndarray, np.ndarray]:
    """Drop positive-label training/validation windows whose annotated event
    has no corroborating SpO2 desaturation, per the SpO2-corroboration audit
    (results/audit/spo2_corroboration_per_event.csv). Negatives are never
    filtered (corroboration doesn't apply to them). Test indices are never
    passed to this function -- the test set is always the full, unfiltered
    data, so a filtered-vs-baseline comparison is a fair apples-to-apples
    evaluation on the same test set, isolating the effect of cleaner
    training labels rather than an easier test set."""

    def keep(index: int) -> bool:
        row = rows[index]
        if int(row["label"]) == 0:
            return True
        return corroboration_lookup.get(row["sample_id"], True)

    train_filtered = np.asarray([i for i in train_idx if keep(i)], dtype=np.int64)
    val_filtered = np.asarray([i for i in val_idx if keep(i)], dtype=np.int64)
    for name, indices in (("train", train_filtered), ("validation", val_filtered)):
        if not len(indices) or len({rows[i]["label"] for i in indices}) != 2:
            raise ValueError(f"{name} split became empty or single-class after corroboration filtering")
    return train_filtered, val_filtered


def validate_manifest_and_folds(manifest_path: Path, fold_path: Path) -> dict:
    rows = read_csv_rows(manifest_path)
    folds = parse_folds(fold_path)
    errors: list[str] = []
    if set(row["subject_id"] for row in rows) != set(folds):
        errors.append("Manifest and fold subject sets differ")
    fold_values = sorted(set(folds.values()))
    if fold_values != list(range(5)):
        errors.append(f"Expected folds 0..4, found {fold_values}")
    by_window: dict[str, list[dict]] = {}
    for row in rows:
        by_window.setdefault(row["logical_window_id"], []).append(row)
    for logical_id, pair in by_window.items():
        if len(pair) != 2 or {row["device"] for row in pair} != {"R", "S"}:
            errors.append(f"Logical window is not an R/S pair: {logical_id}")
            break
        start_key = "reference_start_sec" if "reference_start_sec" in pair[0] else "start_sec"
        end_key = "reference_end_sec" if "reference_end_sec" in pair[0] else "end_sec"
        invariant = (pair[0]["subject_id"], pair[0]["label"], pair[0][start_key], pair[0][end_key])
        if invariant != (pair[1]["subject_id"], pair[1]["label"], pair[1][start_key], pair[1][end_key]):
            errors.append(f"R/S window mismatch: {logical_id}")
            break
    split_counts: list[dict] = []
    for fold in range(5):
        for protocol in ("R_R", "S_S", "R_S", "S_R", "RS_RS"):
            try:
                train, val, test, subjects = split_indices(rows, folds, fold, protocol)
            except Exception as error:
                errors.append(str(error))
                continue
            split_counts.append(
                {
                    "fold": fold,
                    "protocol": protocol,
                    "train_rows": len(train),
                    "val_rows": len(val),
                    "test_rows": len(test),
                    **{key: len(value) for key, value in subjects.items()},
                }
            )
    return {
        "valid": not errors,
        "errors": errors,
        "subjects": len(folds),
        "manifest_rows": len(rows),
        "logical_windows": len(by_window),
        "fold_counts": {str(fold): sum(value == fold for value in folds.values()) for fold in range(5)},
        "split_counts": split_counts,
    }


def load_representation(
    cache_root: Path, representation: str, config: dict, preprocessing_override: str | None = None
) -> tuple[list[np.ndarray], int]:
    if representation not in config["representations"]:
        raise ValueError(f"Unknown representation: {representation}")
    components = config["representations"][representation]
    arrays: list[np.ndarray] = []
    for component in components:
        preprocessing = preprocessing_override or ("peak_filter" if component == "classical" else "peak")
        paths = cache_paths(cache_root, component, preprocessing)
        if not paths["metadata"].exists():
            raise FileNotFoundError(f"Missing cache metadata: {paths['metadata']}")
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        complete = np.load(paths["complete"], mmap_mode="r")
        if metadata.get("status") != "complete" or not bool(np.asarray(complete).all()):
            raise RuntimeError(f"Incomplete feature cache: {paths['root']}")
        arrays.append(np.load(paths["features"], mmap_mode="r"))
    row_counts = {array.shape[0] for array in arrays}
    if len(row_counts) != 1:
        raise ValueError(f"Feature row-count mismatch for {representation}")
    dimension = sum(array.shape[1] for array in arrays)
    return arrays, dimension


def take_features(arrays: Sequence[np.ndarray], indices: np.ndarray) -> np.ndarray:
    if len(arrays) == 1:
        return np.asarray(arrays[0][indices], dtype=np.float32)
    return np.concatenate([np.asarray(array[indices], dtype=np.float32) for array in arrays], axis=1)


# Classifiers whose build_estimator() behavior was changed to run on GPU.
# Included as an extra config_hash argument in run_fold() so that ONLY these
# classifiers' experiment/fit keys change on each addition -- any classifier
# NOT (yet) in this set keeps a byte-identical key to before, so its
# already-completed results stay correctly recognized as valid rather than
# needlessly re-running. random_forest/svm_rbf use cuML via a subprocess
# bridge to an isolated venv; xgboost uses its native device="cuda"; mlp
# uses a from-scratch PyTorch reimplementation (see TorchMLPClassifier --
# not bit-identical to sklearn's MLPClassifier, a faithful approximation).
GPU_ACCELERATED_CLASSIFIERS = {"xgboost", "random_forest", "svm_rbf", "mlp"}


def _require_gpu(classifier: str) -> None:
    """xgboost's device="cuda" and TorchMLPClassifier both silently fall back
    to CPU (with only a warning, in xgboost's case) when no GPU is visible --
    confirmed empirically (2026-08-19): a job missing --gres=gpu would
    otherwise quietly compute on CPU under a "GPU-accelerated" label rather
    than failing. random_forest/svm_rbf don't need this check -- their cuML
    subprocess backend already raises hard (CUDARuntimeError) with no GPU."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{classifier} is GPU-accelerated and requires a GPU (torch.cuda.is_available() is False) -- "
            "this job's SLURM script is missing --gres=gpu. No CPU fallback by design."
        )

# cuML (cuml-cu12) requires newer numpy/scipy/scikit-learn than this project
# is pinned to (which sklearn's MLPClassifier/StandardScaler still depend on
# staying stable for), so it lives in a separate, isolated venv rather than
# being installed here. GPUSubprocessEstimator below bridges to it by
# shelling out per fit/score call -- slower than an in-process call would be,
# but keeps the two dependency stacks from ever conflicting.
_GPU_VENV_PYTHON = "/scratch/pkdas/IEEE_healthcomm_workshop/gpu_classifier_test/bin/python"
_GPU_WORKER_SCRIPT = "/scratch/pkdas/IEEE_healthcomm_workshop/gpu_classifier_test/cuml_worker.py"
_GPU_WORKER_TMP = Path("/scratch/pkdas/IEEE_healthcomm_workshop/gpu_worker_tmp")
# Generous relative to the ~130 min *whole-fold* figure in
# slurm/09s_dimension_control.sbatch's comment (which covers CPU-side PCA
# plus multiple calls) -- a single fit/score subprocess call should never
# legitimately take this long. Prevents a hung/deadlocked cuML subprocess
# from blocking the parent forever.
_GPU_WORKER_TIMEOUT_SEC = 3600


class GPUSubprocessEstimator:
    """Mimics enough of the sklearn estimator interface (fit + predict_proba
    OR decision_function) for select_estimator()/probability() to use it as
    a drop-in, while the actual computation runs in the isolated cuML venv
    via subprocess. `has_predict_proba` controls which of the two interfaces
    is exposed -- True for random_forest (cuML RF has predict_proba, like
    sklearn's), False for svm_rbf (cuML's SVC has no predict_proba at all,
    only decision_function -- which evaluation.py:probability() already
    falls back to with a manual sigmoid for any non-predict_proba estimator,
    so no change needed there)."""

    def __init__(self, classifier: str, hyperparameters: dict, has_predict_proba: bool):
        self.classifier = classifier
        self.hyperparameters = dict(hyperparameters)
        self._has_predict_proba = has_predict_proba
        self._model_path: Path | None = None
        self._work_dir = _GPU_WORKER_TMP / uuid.uuid4().hex
        if has_predict_proba:
            # Only exposed on RF instances, so probability()'s hasattr check
            # takes this branch for RF and the decision_function branch for
            # SVC (see class docstring).
            self.predict_proba = self._predict_proba  # type: ignore[method-assign]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPUSubprocessEstimator":
        self._work_dir.mkdir(parents=True, exist_ok=True)
        data_path = self._work_dir / "train.npz"
        model_path = self._work_dir / "model.pkl"
        np.savez(data_path, X=np.asarray(X, dtype=np.float32), y=np.asarray(y))
        try:
            subprocess.run(
                [_GPU_VENV_PYTHON, _GPU_WORKER_SCRIPT, "fit", self.classifier, json.dumps(self.hyperparameters), str(data_path), str(model_path)],
                check=True, capture_output=True, text=True, timeout=_GPU_WORKER_TIMEOUT_SEC,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"cuML fit subprocess failed for {self.classifier} (exit {error.returncode}): {error.stderr}") from error
        self._model_path = model_path
        data_path.unlink(missing_ok=True)
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        assert self._model_path is not None, "fit() must be called before scoring"
        data_path = self._work_dir / f"score_{uuid.uuid4().hex}.npz"
        output_path = self._work_dir / f"out_{uuid.uuid4().hex}.npz"
        np.savez(data_path, X=np.asarray(X, dtype=np.float32))
        try:
            subprocess.run(
                [_GPU_VENV_PYTHON, _GPU_WORKER_SCRIPT, "score", str(self._model_path), str(data_path), str(output_path)],
                check=True, capture_output=True, text=True, timeout=_GPU_WORKER_TIMEOUT_SEC,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"cuML score subprocess failed for {self.classifier} (exit {error.returncode}): {error.stderr}") from error
        result = np.load(output_path)
        array = result["probability"] if "probability" in result else result["decision"]
        data_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return array

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = self._score(X)
        return np.stack([1.0 - prob, prob], axis=1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._score(X)

    def cleanup(self) -> None:
        for path in self._work_dir.glob("*"):
            path.unlink(missing_ok=True)
        if self._work_dir.exists():
            self._work_dir.rmdir()


def _cleanup_estimator(estimator) -> None:
    """Reclaim a GPUSubprocessEstimator's gpu_worker_tmp/<uuid> directory.

    Safe to call unconditionally on any estimator select_estimator() built
    for a tuning candidate that was NOT selected as the winner (or even one
    that was -- select_estimator's own candidates are always discarded and
    re-fit from scratch by the caller's own build_estimator() call, so no
    candidate estimator returned from select_estimator is ever reused). Do
    NOT call this on the final refit's estimator -- its _model_path is
    depended on by the cached classifier.joblib/pipeline.joblib under
    checkpoints/{downstream_fit_cache,pca_cache,coral_cache}.
    """
    target = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    cleanup = getattr(target, "cleanup", None)
    if cleanup is not None:
        cleanup()


class TorchMLPClassifier:
    """GPU-accelerated approximation of sklearn.neural_network.MLPClassifier.

    Not bit-identical to sklearn's implementation -- different weight
    initialization and a different (though closely comparable) Adam
    implementation -- but built to faithfully reproduce the same
    architecture and hyperparameters actually configured in
    configs/base.yaml: `hidden_layer_sizes`, `alpha` (L2, mapped to Adam's
    weight_decay), `max_iter` (epoch cap), `early_stopping`. Values NOT
    present in configs/base.yaml (learning_rate_init, validation_fraction,
    n_iter_no_change, tol, batch_size, activation=relu) are set to match
    sklearn's own MLPClassifier defaults as closely as practical -- these
    are background-knowledge defaults, not project-confirmed choices (see
    CODEBASE_ARCHITECTURE_AND_EXPERIMENT_LEARNING_GUIDE.md Step 52).
    PyTorch is already a live-venv dependency (used for feature extraction),
    so unlike random_forest/svm_rbf this needs no isolated-venv bridge --
    lives and runs in-process.
    """

    def __init__(
        self, hidden_layer_sizes, alpha: float, max_iter: int, early_stopping: bool, random_state: int,
        learning_rate_init: float = 0.001, validation_fraction: float = 0.1, n_iter_no_change: int = 10, tol: float = 1e-4,
    ):
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.learning_rate_init = learning_rate_init
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        # No silent CPU fallback: the whole point of this class existing is
        # GPU acceleration; a job that lands here without a GPU allocated
        # should fail loudly (matching random_forest/svm_rbf/xgboost's own
        # hard failures in that situation), not quietly compute on CPU under
        # a name that says otherwise.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TorchMLPClassifier requires a GPU (torch.cuda.is_available() is False) -- "
                "this job's SLURM script is missing --gres=gpu. No CPU fallback by design."
            )
        device = torch.device("cuda")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        if self.early_stopping:
            idx = rng.permutation(n)
            n_val = max(1, int(n * self.validation_fraction))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
        else:
            train_idx, val_idx = np.arange(n), np.asarray([], dtype=int)

        sizes = [X.shape[1], *self.hidden_layer_sizes, 1]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        loss_fn = nn.BCEWithLogitsLoss()

        X_train_t = torch.from_numpy(X[train_idx]).to(device)
        y_train_t = torch.from_numpy(y[train_idx]).to(device)
        if len(val_idx):
            X_val_t = torch.from_numpy(X[val_idx]).to(device)
            y_val_t = torch.from_numpy(y[val_idx]).to(device)

        batch_size = min(200, len(train_idx))
        best_score, best_state, no_improve = -np.inf, None, 0

        for _epoch in range(self.max_iter):
            model.train()
            perm = torch.randperm(len(train_idx), device=device)
            for start in range(0, len(train_idx), batch_size):
                batch_idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                logits = model(X_train_t[batch_idx]).squeeze(-1)
                loss_fn(logits, y_train_t[batch_idx]).backward()
                optimizer.step()

            if self.early_stopping:
                model.eval()
                with torch.no_grad():
                    val_pred = (torch.sigmoid(model(X_val_t).squeeze(-1)) >= 0.5).float()
                    score = (val_pred == y_val_t).float().mean().item()
                if score > best_score + self.tol:
                    best_score, best_state, no_improve = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        # Stored on CPU so joblib.dump/load (fit-cache reuse, possibly from a
        # different process/job later) never depends on a live CUDA context
        # being present at pickle time; predict_proba() moves it back to
        # whichever device is available when actually scoring.
        self._model = model.eval().cpu()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TorchMLPClassifier.predict_proba requires a GPU -- no CPU fallback by design.")
        device = torch.device("cuda")
        model = self._model.to(device)
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(X).to(device)).squeeze(-1)).cpu().numpy()
        return np.stack([1.0 - prob, prob], axis=1)


def build_estimator(name: str, parameters: dict, seed: int):
    parameters = dict(parameters)
    if name == "random_forest":
        # GPU acceleration (2026-08-19): cuML's RandomForestClassifier via the
        # isolated-venv subprocess bridge (see GPUSubprocessEstimator). No
        # n_jobs concept on GPU -- dropped, everything else passes through.
        parameters.pop("n_jobs", None)
        return GPUSubprocessEstimator("random_forest", {**parameters, "random_state": seed}, has_predict_proba=True)
    if name == "xgboost":
        from xgboost import XGBClassifier

        parameters.setdefault("eval_metric", "logloss")
        parameters.setdefault("random_state", seed)
        # GPU acceleration (2026-08-19): xgboost's tree_method="hist" (already
        # configured) supports both CPU and GPU depending on `device` -- this
        # is the modern, well-supported GPU path (not the deprecated
        # gpu_hist). Does not change what's learned in principle, only where
        # the histogram computation runs; see GPU_ACCELERATED_CLASSIFIERS
        # above for why this forces a fresh experiment key.
        # NOTE: xgboost with device="cuda" and no GPU present only warns and
        # silently runs on CPU -- confirmed empirically -- hence the explicit
        # check here rather than trusting xgboost's own behavior.
        _require_gpu("xgboost")
        parameters.setdefault("device", "cuda")
        return XGBClassifier(**parameters)
    if name == "svm_rbf":
        # GPU acceleration (2026-08-19): cuML's SVC via the same bridge.
        # probability=True (Platt scaling) has no cuML equivalent -- dropped;
        # cuML's SVC only exposes decision_function, which
        # evaluation.py:probability() already handles via its manual-sigmoid
        # fallback for any non-predict_proba estimator (this is also the
        # exact mechanism the Platt-scaling calibration-collapse fix in
        # advanced.py relies on, so behavior here is consistent with that).
        parameters.pop("probability", None)
        return make_pipeline(StandardScaler(), GPUSubprocessEstimator("svm_rbf", {**parameters, "random_state": seed}, has_predict_proba=False))
    if name == "mlp":
        # GPU acceleration (2026-08-19): TorchMLPClassifier (see class docstring
        # for exactly which hyperparameters carry over and which are
        # sklearn-default approximations). Not bit-identical to
        # sklearn.neural_network.MLPClassifier -- forces a fresh experiment
        # key via GPU_ACCELERATED_CLASSIFIERS.
        return make_pipeline(StandardScaler(), TorchMLPClassifier(random_state=seed, **parameters))
    raise ValueError(f"Unknown classifier: {name}")


def probability(estimator, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(features)[:, 1], dtype=np.float64)
    decision = np.asarray(estimator.decision_function(features), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -30, 30)))


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = (np.asarray(probabilities) >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    # Fold-level splits are guaranteed multi-class by split_indices()'s own
    # assertion, but per-subject slices (a single test subject's windows)
    # are not -- roc_auc_score raises ValueError if a slice happens to be
    # single-class, same edge case sensitivity/specificity already guard
    # below with a NaN fallback rather than a crash.
    try:
        roc_auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(labels)),
    }


def select_estimator(name: str, candidates: Sequence[dict], seed: int, x_train, y_train, x_val, y_val):
    best = None
    records: list[dict] = []
    for candidate_index, parameters in enumerate(candidates):
        estimator = build_estimator(name, parameters, seed + candidate_index)
        start = time.perf_counter()
        estimator.fit(x_train, y_train)
        val_prob = probability(estimator, x_val)
        score = balanced_accuracy_score(y_val, val_prob >= 0.5)
        # Every candidate here is a pure tuning throwaway: the winning
        # hyperparameters get refit from scratch by the caller, so no
        # candidate estimator (winner or not) is ever reused past this
        # loop. Reclaim its gpu_worker_tmp/<uuid> directory immediately
        # rather than leaking it (see _cleanup_estimator docstring).
        _cleanup_estimator(estimator)
        records.append(
            {
                "candidate": candidate_index,
                "hyperparameters": parameters,
                "validation_balanced_accuracy": float(score),
                "fit_sec": time.perf_counter() - start,
            }
        )
        rank = (float(score), -candidate_index)
        if best is None or rank > best[0]:
            best = (rank, parameters)
    assert best is not None
    return dict(best[1]), records


MASTER_FIELDS = [
    "run_id", "timestamp", "git_commit", "seed", "fold", "train_subjects", "val_subjects", "test_subjects",
    "device_train", "device_test", "representation", "classifier", "hyperparameters", "preprocessing",
    "feature_dimension", "metrics", "runtime", "hardware", "experiment_key", "result_dir",
]


def _completed_key(results_root: Path, key: str) -> Path | None:
    for path in results_root.glob(f"runs/{key}*/completion.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if record.get("status") == "complete" and record.get("experiment_key") == key:
            return path.parent
    return None


def _new_run_dir(results_root: Path, key: str) -> Path:
    base = results_root / "runs" / key
    try:
        base.mkdir(parents=True)
        return base
    except FileExistsError:
        pass
    # A concurrent worker (e.g. --workers > 1) won the race on `base` between
    # our _completed_key() check and this mkdir() -- fall through to a
    # retry-suffixed directory rather than crashing.
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    retry = results_root / "runs" / f"{key}_retry_{suffix}_{os.getpid()}"
    retry.mkdir(parents=True)
    return retry


def run_fold(
    *, config: dict, manifest_path: Path, fold_path: Path, cache_root: Path, results_root: Path,
    representation: str, classifier: str, protocol: str, fold: int, preprocessing_override: str | None = None,
    filter_uncorroborated_training: bool = False,
) -> dict:
    seed = int(config["seed"])
    # gpu_tag is empty for every classifier not in GPU_ACCELERATED_CLASSIFIERS,
    # so their config_hash call is byte-identical to before this change --
    # only classifiers whose build_estimator() behavior actually changed get
    # a new key, forcing exactly (and only) their stale results to re-run.
    gpu_tag = ("gpu_v1",) if classifier in GPU_ACCELERATED_CLASSIFIERS else ()
    # Same idiom for the SpO2-corroboration training-label-quality ablation
    # (2026-08-19): empty when disabled, so every existing call site/cache
    # entry is completely unaffected; only runs that opt in get a distinct
    # key, and this ablation is additionally routed to its own results root
    # by the caller (never results/P0_device_gap) so it can never collide
    # with -- or get silently deduped against -- the baseline benchmark.
    corrob_tag = ("corrob_filtered_v1",) if filter_uncorroborated_training else ()
    key = config_hash(
        "main_benchmark_v1", file_sha256(manifest_path), file_sha256(fold_path), config,
        representation, classifier, protocol, fold, preprocessing_override, *gpu_tag, *corrob_tag,
    )
    completed = _completed_key(results_root, key)
    if completed is not None:
        return {"status": "skipped_complete", "result_dir": str(completed), "experiment_key": key}
    run_dir = _new_run_dir(results_root, key)
    rows = read_csv_rows(manifest_path)
    folds = parse_folds(fold_path)
    train_idx, val_idx, test_idx, subject_sets = split_indices(rows, folds, fold, protocol)
    if filter_uncorroborated_training:
        corroboration_lookup = _load_window_corroboration(config)
        train_idx, val_idx = filter_uncorroborated(train_idx, val_idx, rows, corroboration_lookup)
        subject_sets = {
            **subject_sets,
            "corroboration_filtered_train_rows": int(len(train_idx)),
            "corroboration_filtered_val_rows": int(len(val_idx)),
        }
    arrays, dimension = load_representation(cache_root, representation, config, preprocessing_override)
    if arrays[0].shape[0] != len(rows):
        raise ValueError("Manifest/cache row mismatch")
    started = time.perf_counter()
    train_devices, _ = protocol_devices(protocol)
    preprocessing_name = preprocessing_override or ("peak_filter" if representation == "classical" else "peak")
    fit_key = config_hash(
        "downstream_fit_v1", file_sha256(manifest_path), file_sha256(fold_path), config,
        representation, classifier, fold, sorted(train_devices), preprocessing_name, *gpu_tag, *corrob_tag,
    )
    fit_root = Path(config["project_root"]) / "checkpoints" / "downstream_fit_cache"
    fit_dir = fit_root / fit_key
    fit_model_path = fit_dir / "classifier.joblib"
    fit_record_path = fit_dir / "completion.json"
    reused_fit = False
    for candidate_record_path in sorted(fit_root.glob(f"{fit_key}*/completion.json")):
        candidate_model_path = candidate_record_path.parent / "classifier.joblib"
        if not candidate_model_path.exists():
            continue
        candidate_record = json.loads(candidate_record_path.read_text(encoding="utf-8"))
        if candidate_record.get("status") == "complete" and candidate_record.get("fit_key") == fit_key:
            fit_dir = candidate_record_path.parent
            fit_model_path = candidate_model_path
            fit_record_path = candidate_record_path
            break
    if fit_record_path.exists() and fit_model_path.exists():
        fit_record = json.loads(fit_record_path.read_text(encoding="utf-8"))
        if fit_record.get("status") != "complete" or fit_record.get("fit_key") != fit_key:
            raise RuntimeError(f"Invalid downstream fit cache: {fit_dir}")
        estimator = joblib.load(fit_model_path)
        selected = fit_record["selected_hyperparameters"]
        tuning = fit_record["tuning"]
        refit_sec = float(fit_record["refit_sec"])
        reused_fit = True
    else:
        try:
            fit_dir.mkdir(parents=True)
        except FileExistsError:
            # fit_key deliberately excludes protocol (it keys only on
            # sorted(train_devices), and e.g. R_R/R_S share train_devices={"R"}),
            # so two concurrent --workers jobs for the same
            # representation/classifier/fold can legitimately race here.
            # Attempt-then-fallback (rather than exists()-then-mkdir) closes
            # the check-then-create window; only actual collision falls
            # through to a retry-suffixed directory.
            retry_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            fit_dir = fit_dir.parent / f"{fit_key}_retry_{retry_suffix}_{os.getpid()}"
            fit_model_path = fit_dir / "classifier.joblib"
            fit_record_path = fit_dir / "completion.json"
            fit_dir.mkdir(parents=True)
        x_train = take_features(arrays, train_idx)
        x_val = take_features(arrays, val_idx)
        y_train = np.asarray([int(rows[i]["label"]) for i in train_idx], dtype=np.int8)
        y_val = np.asarray([int(rows[i]["label"]) for i in val_idx], dtype=np.int8)
        selected, tuning = select_estimator(
            classifier, config["classifiers"][classifier], seed + fold * 100, x_train, y_train, x_val, y_val
        )
        del x_train, x_val
        fit_idx = np.concatenate([train_idx, val_idx])
        x_fit = take_features(arrays, fit_idx)
        y_fit = np.asarray([int(rows[i]["label"]) for i in fit_idx], dtype=np.int8)
        estimator = build_estimator(classifier, selected, seed + fold * 1000)
        refit_started = time.perf_counter()
        estimator.fit(x_fit, y_fit)
        refit_sec = time.perf_counter() - refit_started
        del x_fit
        joblib.dump(estimator, fit_model_path)
        fit_record_path.write_text(
            json.dumps(
                {"status": "complete", "fit_key": fit_key, "representation": representation, "classifier": classifier,
                 "fold": fold, "train_devices": sorted(train_devices), "preprocessing": preprocessing_name,
                 "selected_hyperparameters": selected, "tuning": tuning, "refit_sec": refit_sec,
                 "train_subjects": subject_sets["train_subjects"], "val_subjects": subject_sets["val_subjects"]},
                indent=2,
            ),
            encoding="utf-8",
        )
    x_test = take_features(arrays, test_idx)
    inference_started = time.perf_counter()
    probabilities = probability(estimator, x_test)
    inference_sec = time.perf_counter() - inference_started
    labels = np.asarray([int(rows[i]["label"]) for i in test_idx], dtype=np.int8)
    result_metrics = metrics(labels, probabilities)
    predictions = (probabilities >= 0.5).astype(np.int8)
    run_id = run_dir.name
    with gzip.open(run_dir / "window_predictions.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        fieldnames = ["run_id", "fold", "protocol", "representation", "classifier", "sample_id", "logical_window_id", "subject_id", "device", "label", "probability", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, index in enumerate(test_idx):
            row = rows[int(index)]
            writer.writerow(
                {
                    "run_id": run_id, "fold": fold, "protocol": protocol, "representation": representation,
                    "classifier": classifier, "sample_id": row["sample_id"], "logical_window_id": row["logical_window_id"],
                    "subject_id": row["subject_id"], "device": row["device"], "label": int(labels[position]),
                    "probability": float(probabilities[position]), "prediction": int(predictions[position]),
                }
            )
    subject_rows: list[dict] = []
    test_subject_ids = np.asarray([rows[int(i)]["subject_id"] for i in test_idx])
    for subject in sorted(set(test_subject_ids)):
        mask = test_subject_ids == subject
        subject_rows.append({"run_id": run_id, "subject_id": subject, **metrics(labels[mask], probabilities[mask])})
    with (run_dir / "subject_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0]))
        writer.writeheader()
        writer.writerows(subject_rows)
    # The fitted classifier itself already lives in the shared, deduped
    # downstream_fit_cache (fit_model_path). Nothing reads a per-run copy
    # back, so store a lightweight pointer instead of duplicating a
    # potentially large binary (e.g. svm_rbf with cache_size=4096) into
    # every run directory.
    (run_dir / "classifier_ref.json").write_text(
        json.dumps({"fit_key": fit_key, "classifier_path": str(fit_model_path)}, indent=2),
        encoding="utf-8",
    )
    runtime = {
        "total_sec": time.perf_counter() - started,
        "refit_sec": refit_sec,
        "test_inference_sec": inference_sec,
        "test_inference_sec_per_clip": inference_sec / len(test_idx),
        "downstream_fit_reused": reused_fit,
    }
    device_train, device_test = protocol_devices(protocol)
    record = {
        "status": "complete", "experiment_key": key, "run_id": run_id, "timestamp": utc_now(),
        "seed": seed, "fold": fold, "protocol": protocol, "representation": representation,
        "classifier": classifier, "selected_hyperparameters": selected, "tuning": tuning,
        "feature_dimension": dimension, "preprocessing": preprocessing_name,
        "downstream_fit_cache": str(fit_dir), "downstream_fit_reused": reused_fit,
        "subjects": subject_sets, "row_counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "metrics": result_metrics, "runtime": runtime,
        "hardware": {"host": socket.gethostname(), "platform": platform.platform(), "slurm_job_id": os.getenv("SLURM_JOB_ID", "")},
    }
    (run_dir / "completion.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    append_csv(
        Path(config["logging"]["master_log"]),
        [{
            "run_id": run_id, "timestamp": record["timestamp"], "git_commit": config["logging"]["git_commit_fallback"],
            "seed": seed, "fold": fold, "train_subjects": json.dumps(subject_sets["train_subjects"]),
            "val_subjects": json.dumps(subject_sets["val_subjects"]), "test_subjects": json.dumps(subject_sets["test_subjects"]),
            "device_train": "+".join(sorted(device_train)), "device_test": "+".join(sorted(device_test)),
            "representation": representation, "classifier": classifier, "hyperparameters": json.dumps(selected, sort_keys=True),
            "preprocessing": record["preprocessing"], "feature_dimension": dimension,
            "metrics": json.dumps(result_metrics, sort_keys=True), "runtime": json.dumps(runtime, sort_keys=True),
            "hardware": json.dumps(record["hardware"], sort_keys=True), "experiment_key": key, "result_dir": str(run_dir),
        }],
        MASTER_FIELDS,
    )
    return record


def iter_grid(config: dict, representations: Iterable[str], classifiers: Iterable[str], protocols: Iterable[str], folds: Iterable[int]):
    for representation in representations:
        for classifier in classifiers:
            for protocol in protocols:
                for fold in folds:
                    yield representation, classifier, protocol, fold
