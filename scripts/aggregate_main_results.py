#!/usr/bin/env python3
"""Materialize plan-required P0-A CSVs and publication-ready benchmark figure."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.evaluation import PRIMARY_METRICS
from sleep_quadnet.io import load_completed_runs


def exclusive_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) == list(frame.columns) and existing.equals(frame):
            return
        raise FileExistsError(f"Refusing to overwrite an existing different result: {path}")
    frame.to_csv(path, index=False)


def load_records(root: Path) -> list[dict]:
    # Dedup by (representation, classifier, protocol, fold): the GPU
    # migration and earlier calibration-fix salt bumps can each leave more
    # than one "complete" completion.json for the same logical combo on
    # disk (a stale pre-fix/pre-migration one alongside the current one).
    # load_completed_runs() keeps only the latest-timestamped record per
    # combo -- see its docstring for why this must not be a raw glob.
    records = load_completed_runs(root, ("representation", "classifier", "protocol", "fold"))
    if not records:
        raise FileNotFoundError(f"No completed fold runs in {root}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "P0_device_gap")
    parser.add_argument("--figure", type=Path, default=PROJECT_ROOT / "figures" / "cross_device_robustness.pdf")
    args = parser.parse_args()
    records = load_records(args.results_root)
    fold_rows = []
    subject_frames = []
    prediction_frames = []
    for record in records:
        base = {
            "run_id": record["run_id"], "fold": record["fold"], "protocol": record["protocol"],
            "representation": record["representation"], "classifier": record["classifier"],
            "feature_dimension": record["feature_dimension"], "preprocessing": record["preprocessing"],
            **record["row_counts"], **record["runtime"], **record["metrics"],
        }
        fold_rows.append(base)
        subject = pd.read_csv(Path(record["result_dir"]) / "subject_metrics.csv")
        for column in ("fold", "protocol", "representation", "classifier", "feature_dimension"):
            subject[column] = base[column]
        subject_frames.append(subject)
        with gzip.open(Path(record["result_dir"]) / "window_predictions.csv.gz", "rt", encoding="utf-8") as handle:
            prediction_frames.append(pd.read_csv(handle, dtype={"subject_id": str}))
    fold_frame = pd.DataFrame(fold_rows).sort_values(["representation", "classifier", "protocol", "fold"])
    subject_frame = pd.concat(subject_frames, ignore_index=True).sort_values(
        ["representation", "classifier", "protocol", "fold", "subject_id"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary_rows = []
    for keys, group in fold_frame.groupby(["representation", "classifier", "protocol"], sort=True):
        row = dict(zip(("representation", "classifier", "protocol"), keys))
        row["folds"] = len(group)
        row["feature_dimension"] = int(group["feature_dimension"].iloc[0])
        for metric in PRIMARY_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    gap_rows = []
    for (representation, classifier), group in summary.groupby(["representation", "classifier"], sort=True):
        indexed = group.set_index("protocol")
        if not {"R_R", "S_S", "R_S", "S_R"}.issubset(indexed.index):
            continue
        row = {"representation": representation, "classifier": classifier, "feature_dimension": int(group["feature_dimension"].iloc[0])}
        for metric in PRIMARY_METRICS:
            values = {protocol: float(indexed.loc[protocol, f"{metric}_mean"]) for protocol in ("R_R", "S_S", "R_S", "S_R")}
            row.update({f"{metric}_{protocol}": value for protocol, value in values.items()})
            row[f"{metric}_gap_R_to_S"] = values["R_R"] - values["R_S"]
            row[f"{metric}_gap_S_to_R"] = values["S_S"] - values["S_R"]
            row[f"{metric}_mean_matched"] = np.mean([values["R_R"], values["S_S"]])
            row[f"{metric}_mean_cross_device"] = np.mean([values["R_S"], values["S_R"]])
            row[f"{metric}_mean_device_gap"] = row[f"{metric}_mean_matched"] - row[f"{metric}_mean_cross_device"]
        gap_rows.append(row)
    gap = pd.DataFrame(gap_rows)
    exclusive_csv(fold_frame, args.results_root / "fold_metrics.csv")
    exclusive_csv(subject_frame, args.results_root / "subject_level_predictions.csv")
    exclusive_csv(summary, args.results_root / "protocol_summary.csv")
    exclusive_csv(gap, args.results_root / "device_gap_summary.csv")
    confusion = fold_frame.groupby(["representation", "classifier", "protocol"], as_index=False)[["tn", "fp", "fn", "tp"]].sum()
    exclusive_csv(confusion, args.results_root / "confusion_matrices" / "aggregate_counts.csv")

    for (representation, classifier), group in predictions.groupby(["representation", "classifier"], sort=True):
        roc_path = args.results_root / "roc_curves" / f"{representation}__{classifier}.pdf"
        matrix_path = args.results_root / "confusion_matrices" / f"{representation}__{classifier}.pdf"
        roc_path.parent.mkdir(parents=True, exist_ok=True)
        if roc_path.exists() or matrix_path.exists():
            raise FileExistsError(f"Refusing to overwrite ROC/confusion figure for {representation}/{classifier}")
        figure, axis = plt.subplots(figsize=(5, 5))
        for protocol, protocol_group in group.groupby("protocol"):
            false_positive, true_positive, _ = roc_curve(protocol_group["label"], protocol_group["probability"])
            axis.plot(false_positive, true_positive, label=f"{protocol} (AUC={auc(false_positive, true_positive):.3f})")
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(roc_path, bbox_inches="tight")
        plt.close(figure)
        protocols_present = [name for name in ("R_R", "S_S", "R_S", "S_R") if name in set(group["protocol"])]
        figure, axes = plt.subplots(1, len(protocols_present), figsize=(3.2 * len(protocols_present), 3))
        axes = np.atleast_1d(axes)
        for axis, protocol in zip(axes, protocols_present):
            protocol_group = group[group["protocol"] == protocol]
            matrix = confusion_matrix(protocol_group["label"], protocol_group["prediction"], labels=[0, 1])
            ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Event"]).plot(ax=axis, colorbar=False)
            axis.set_title(protocol)
        figure.tight_layout()
        figure.savefig(matrix_path, bbox_inches="tight")
        plt.close(figure)

    complete = summary[summary["folds"] == 5]
    if not complete.empty:
        plot = complete.pivot_table(index=["representation", "classifier"], columns="protocol", values="balanced_accuracy_mean")
        plot = plot[[column for column in ("R_R", "S_S", "R_S", "S_R") if column in plot]]
        figure, axis = plt.subplots(figsize=(max(8, 0.45 * len(plot)), 5))
        plot.plot(kind="bar", ax=axis, width=0.82)
        axis.set_ylabel("Balanced accuracy")
        axis.set_xlabel("Representation / classifier")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Protocol", ncol=4, fontsize=8)
        figure.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        if args.figure.exists():
            raise FileExistsError(f"Refusing to overwrite figure: {args.figure}")
        figure.savefig(args.figure, bbox_inches="tight")
        plt.close(figure)
    print(json.dumps({"completed_fold_runs": len(fold_frame), "summary_rows": len(summary), "complete_configurations": int((summary["folds"] == 5).sum())}, indent=2))


if __name__ == "__main__":
    main()
