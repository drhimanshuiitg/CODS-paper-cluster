#!/usr/bin/env python3
"""Q1_Paper_Artifact: aggregate every completed fold-run across every
results root into one master ledger. Window-level metrics only (the unit
actually stored in each completion.json); subject-level aggregation for
significance testing lives separately in results/*_statistics/*.csv and is
referenced, not duplicated, here."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.io import load_completed_runs

TRAIN_TEST = {
    "R_R": ("R", "R"), "S_S": ("S", "S"), "R_S": ("R", "S"), "S_R": ("S", "R"), "RS_RS": ("R+S", "R+S"),
}

ROOTS = [
    ("P0_device_gap", "main_benchmark", ("representation", "classifier", "protocol", "fold")),
    ("P0_ablation", "leave_one_encoder_out_ablation", ("representation", "classifier", "protocol", "fold")),
    ("P1_dimension_control", "pca_pre_fix", ("representation", "classifier", "protocol", "fold", "feature_dimension")),
    ("P1_dimension_control_v3", "pca_post_fix", ("representation", "classifier", "protocol", "fold", "feature_dimension")),
    ("P1_domain_adaptation", "coral", ("representation", "classifier", "protocol", "fold")),
    ("P2_label_quality_ablation", "spo2_corroboration_filter_ablation", ("representation", "classifier", "protocol", "fold")),
    ("P3_sliding_window_severity", "sliding_window_severity", ("protocol", "classifier", "fold")),
]

METRIC_COLS = ["accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity", "f1", "macro_f1", "mcc", "cohen_kappa", "roc_auc", "tn", "fp", "fn", "tp"]


def main() -> None:
    rows = []
    for root_name, experiment_family, key_fields in ROOTS:
        root = PROJECT_ROOT / "results" / root_name
        if not root.exists():
            continue
        records = load_completed_runs(root, key_fields)
        for r in records:
            protocol = r.get("protocol", "")
            train_dev, test_dev = TRAIN_TEST.get(protocol, ("", ""))
            m = r.get("metrics", {})
            row = {
                "experiment_family": experiment_family,
                "results_root": root_name,
                "representation": r.get("representation"),
                "base_representation": r.get("base_representation", r.get("representation")),
                "feature": r.get("feature"),
                "classifier": r.get("classifier"),
                "protocol": protocol,
                "train_domain": train_dev,
                "test_domain": test_dev,
                "fold": r.get("fold"),
                "feature_dimension": r.get("feature_dimension"),
                "n_test": (m.get("tn", 0) + m.get("fp", 0) + m.get("fn", 0) + m.get("tp", 0)) or None,
                "evaluation_unit": "window-level (per-fold test set)",
                "run_id": r.get("run_id"),
                "experiment_key": r.get("experiment_key"),
                "timestamp": r.get("timestamp"),
            }
            for col in METRIC_COLS:
                row[col] = m.get(col)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "Q1_Paper_Artifact" / "MASTER_RESULTS.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
    print(df.groupby("experiment_family").size())


if __name__ == "__main__":
    main()
