#!/usr/bin/env python3
"""Gate SSL launch on a complete leakage-free cheap baseline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    errors = []
    complete_path = PROJECT_ROOT / "cached_features" / "classical" / "peak_filter" / "complete.npy"
    if not complete_path.exists() or not bool(np.load(complete_path, mmap_mode="r").all()):
        errors.append("Classical cache is absent or incomplete")
    records = []
    for path in (PROJECT_ROOT / "results" / "P0_device_gap" / "runs").glob("*/completion.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete" and record["representation"] == "classical" and record["classifier"] == "svm_rbf" and record["protocol"] in {"R_R", "S_S", "R_S", "S_R"}:
            record["directory"] = path.parent
            records.append(record)
    keys = {(record["protocol"], int(record["fold"])) for record in records}
    expected = {(protocol, fold) for protocol in ("R_R", "S_S", "R_S", "S_R") for fold in range(5)}
    if keys != expected:
        errors.append(f"Baseline fold keys differ: missing={sorted(expected - keys)}, extra={sorted(keys - expected)}")
    prediction_sets = {}
    for record in records:
        subjects = record["subjects"]
        train, val, test = map(set, (subjects["train_subjects"], subjects["val_subjects"], subjects["test_subjects"]))
        if train & val or train & test or val & test:
            errors.append(f"Subject leakage in {record['run_id']}")
        with gzip.open(record["directory"] / "window_predictions.csv.gz", "rt", encoding="utf-8") as handle:
            predictions = pd.read_csv(handle, dtype={"subject_id": str})
        predictions["subject_id"] = predictions["subject_id"].str.zfill(2)
        if set(predictions["subject_id"]) != test:
            errors.append(f"Prediction/test subject mismatch in {record['run_id']}")
        expected_device = record["protocol"].split("_")[1]
        if set(predictions["device"]) != {expected_device}:
            errors.append(f"Test device mismatch in {record['run_id']}")
        if not np.isfinite(predictions["probability"]).all():
            errors.append(f"Non-finite probabilities in {record['run_id']}")
        prediction_sets[(record["protocol"], record["fold"])] = set(predictions["sample_id"])
    for protocol in ("R_R", "S_S", "R_S", "S_R"):
        sets = [prediction_sets.get((protocol, fold), set()) for fold in range(5)]
        if sum(len(item) for item in sets) != len(set().union(*sets)):
            errors.append(f"Duplicate test samples across folds for {protocol}")
    result = {
        "valid": not errors, "errors": errors, "completed_fold_runs": len(records),
        "expected_fold_runs": 20, "protocols": ["R_R", "S_S", "R_S", "S_R"],
        "folds": 5, "representation": "classical", "classifier": "svm_rbf",
        "gate": "SSL extraction may launch only when valid=true",
    }
    target = PROJECT_ROOT / "results" / "audit" / "phase1_baseline_validation.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
