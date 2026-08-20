#!/usr/bin/env python3
"""Isolated cuML fit/score worker used by sleep_quadnet.evaluation.

The main project pins an older scientific Python stack, so RAPIDS lives in
``.gpu_classifier_venv`` and communicates through NumPy archives. This
worker intentionally fails if CUDA is unavailable; it never falls back to a
CPU estimator.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC


def require_gpu() -> None:
    count = cp.cuda.runtime.getDeviceCount()
    if count < 1:
        raise RuntimeError("cuML worker has no visible CUDA GPU")


def numpy_output(value) -> np.ndarray:
    if isinstance(value, cp.ndarray):
        value = cp.asnumpy(value)
    return np.asarray(value)


def fit(classifier: str, parameters_json: str, data_path: Path, model_path: Path) -> None:
    parameters = json.loads(parameters_json)
    with np.load(data_path) as data:
        features = np.asarray(data["X"], dtype=np.float32, order="C")
        labels = np.asarray(data["y"], dtype=np.int32)

    if classifier == "random_forest":
        estimator = RandomForestClassifier(output_type="numpy", **parameters)
    elif classifier == "svm_rbf":
        estimator = SVC(kernel="rbf", probability=False, output_type="numpy", **parameters)
    else:
        raise ValueError(f"Unsupported cuML classifier: {classifier}")

    estimator.fit(features, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"classifier": classifier, "estimator": estimator}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def score(model_path: Path, data_path: Path, output_path: Path) -> None:
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    with np.load(data_path) as data:
        features = np.asarray(data["X"], dtype=np.float32, order="C")

    estimator = payload["estimator"]
    if payload["classifier"] == "random_forest":
        probabilities = numpy_output(estimator.predict_proba(features))
        probability = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
        np.savez(output_path, probability=np.asarray(probability, dtype=np.float64))
    else:
        decision = numpy_output(estimator.decision_function(features)).reshape(-1)
        np.savez(output_path, decision=np.asarray(decision, dtype=np.float64))


def main() -> None:
    require_gpu()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: cuml_worker.py {fit|score} ...")
    command = sys.argv[1]
    if command == "fit" and len(sys.argv) == 6:
        fit(sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5]))
    elif command == "score" and len(sys.argv) == 5:
        score(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(f"Invalid arguments for {command}: {sys.argv[1:]}")


if __name__ == "__main__":
    main()
