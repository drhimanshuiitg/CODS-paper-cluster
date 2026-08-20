#!/usr/bin/env python3
"""Validation-only top-3 representation selection after the SVM ablation stage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_REPRESENTATIONS = {"hubert", "wavlm", "wav2vec2", "data2vec_audio", "data2vec_spectrogram", "data2vec_fusion", "full_fusion"}


def scan(root: Path):
    for path in root.glob("runs/*/completion.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete":
            yield record


def main() -> None:
    records = list(scan(PROJECT_ROOT / "results" / "P0_device_gap")) + list(scan(PROJECT_ROOT / "results" / "P0_ablation"))
    rows = []
    for record in records:
        if record["classifier"] != "svm_rbf" or record["protocol"] not in {"R_S", "S_R"}:
            continue
        if record["representation"] == "classical":
            continue
        rows.append(
            {"representation": record["representation"], "protocol": record["protocol"], "fold": record["fold"],
             "validation_balanced_accuracy": max(item["validation_balanced_accuracy"] for item in record["tuning"])}
        )
    frame = pd.DataFrame(rows)
    counts = frame.groupby("representation").size()
    complete = counts[counts == 10].index
    frame = frame[frame["representation"].isin(complete)]
    summary = frame.groupby("representation", as_index=False).agg(
        validation_balanced_accuracy_mean=("validation_balanced_accuracy", "mean"),
        validation_balanced_accuracy_std=("validation_balanced_accuracy", "std"),
        validation_runs=("validation_balanced_accuracy", "size"),
    ).sort_values(["validation_balanced_accuracy_mean", "validation_balanced_accuracy_std", "representation"], ascending=[False, True, True])
    if len(summary) < 3:
        raise ValueError("Fewer than three complete ablation representations")
    top3 = summary.head(3)["representation"].tolist()
    root = PROJECT_ROOT / "results" / "P0_ablation"
    csv_path = root / "validation_only_representation_selection.csv"
    json_path = root / "selection.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Refusing to overwrite ablation selection")
    summary.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"primary_classifier": "svm_rbf", "top3_representations": top3, "selection_basis": "mean cross-device validation balanced accuracy"}, indent=2), encoding="utf-8")
    print(json.dumps({"top3_representations": top3}, indent=2))


if __name__ == "__main__":
    main()
