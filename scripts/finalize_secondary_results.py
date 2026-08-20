#!/usr/bin/env python3
"""Aggregate preprocessing, PCA, CORAL, paper tables, and remaining vector figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.evaluation import PRIMARY_METRICS
from sleep_quadnet.io import load_completed_runs, read_csv_rows

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_utils import annotate_scatter


def exclusive(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    frame.to_csv(path, index=False)


def aggregate_preprocessing() -> pd.DataFrame:
    rows = []
    # Keyed on (preprocessing, protocol, fold) -- classifier is always
    # svm_rbf for this phase (no CLI flag varies it today).
    for record in load_completed_runs(PROJECT_ROOT / "results" / "P1_preprocessing", ("preprocessing", "protocol", "fold")):
        rows.append({"preprocessing": record["preprocessing"], "protocol": record["protocol"], "fold": record["fold"], **record["metrics"]})
    folds = pd.DataFrame(rows)
    if folds.empty:
        raise ValueError("No preprocessing runs")
    summary = folds.groupby(["preprocessing", "protocol"], as_index=False).agg(
        folds=("fold", "size"), **{f"{metric}_mean": (metric, "mean") for metric in PRIMARY_METRICS},
        **{f"{metric}_std": (metric, "std") for metric in PRIMARY_METRICS},
    )
    output = []
    for preprocessing, group in summary.groupby("preprocessing"):
        indexed = group.set_index("protocol")
        if not {"R_R", "S_S", "R_S", "S_R"}.issubset(indexed.index):
            continue
        row = {"preprocessing": preprocessing}
        for metric in PRIMARY_METRICS:
            matched = np.mean([indexed.loc["R_R", f"{metric}_mean"], indexed.loc["S_S", f"{metric}_mean"]])
            cross = np.mean([indexed.loc["R_S", f"{metric}_mean"], indexed.loc["S_R", f"{metric}_mean"]])
            row[f"{metric}_mean_matched"] = matched
            row[f"{metric}_mean_cross_device"] = cross
            row[f"{metric}_mean_device_gap"] = matched - cross
        output.append(row)
    result = pd.DataFrame(output)
    exclusive(result, PROJECT_ROOT / "results" / "P1_preprocessing" / "preprocessing_ablation.csv")
    return result


def aggregate_dimension() -> pd.DataFrame:
    rows = []
    # Keyed on (representation, protocol, fold) -- "representation" already
    # encodes the PCA dimension (e.g. full_fusion_pca_384), so this triple is
    # the true logical identity; classifier is always svm_rbf here today
    # (no CLI flag varies it) -- revisit this key if that ever changes.
    for record in load_completed_runs(PROJECT_ROOT / "results" / "P1_dimension_control", ("representation", "protocol", "fold")):
        rows.append({"representation": record["representation"], "protocol": record["protocol"], "fold": record["fold"], "feature_dimension": record["feature_dimension"], **record["metrics"]})
    fold_frame = pd.DataFrame(rows)
    if fold_frame.empty:
        raise ValueError("No dimension-control runs")
    main = pd.read_csv(PROJECT_ROOT / "results" / "P0_device_gap" / "fold_metrics.csv")
    selection = json.loads((PROJECT_ROOT / "results" / "P0_statistics" / "selection.json").read_text(encoding="utf-8"))
    classifier = selection["primary_classifier"]
    native = main[(main["classifier"] == classifier) & main["protocol"].isin(["R_S", "S_R"]) & main["representation"].isin([selection["best_single_encoder"], "data2vec_fusion", "full_fusion"])]
    native = native[["representation", "protocol", "fold", "feature_dimension", *PRIMARY_METRICS]]
    fold_frame = pd.concat([fold_frame, native], ignore_index=True)
    summary = fold_frame.groupby(["representation", "feature_dimension", "protocol"], as_index=False).agg(
        folds=("fold", "size"), **{f"{metric}_mean": (metric, "mean") for metric in PRIMARY_METRICS},
        **{f"{metric}_std": (metric, "std") for metric in PRIMARY_METRICS},
    )
    exclusive(fold_frame, PROJECT_ROOT / "results" / "P1_dimension_control" / "dimension_control_fold_metrics.csv")
    exclusive(summary, PROJECT_ROOT / "results" / "P1_dimension_control" / "dimension_control_summary.csv")
    plot = summary.groupby(["representation", "feature_dimension"], as_index=False)["balanced_accuracy_mean"].mean().sort_values("feature_dimension")
    figure_path = PROJECT_ROOT / "figures" / "performance_vs_feature_dimension.pdf"
    if figure_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {figure_path}")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.set(xlabel="Feature dimension", ylabel="Mean cross-device balanced accuracy", ylim=(0, 1))
    annotate_scatter(axis, plot["feature_dimension"], plot["balanced_accuracy_mean"], plot["representation"])
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    return summary


def aggregate_coral() -> pd.DataFrame:
    rows = []
    # Keyed on (representation, protocol, fold) -- "representation" already
    # encodes the CORAL-adapted identity (e.g. full_fusion_coral); same
    # svm_rbf-only caveat as aggregate_dimension() above.
    for record in load_completed_runs(PROJECT_ROOT / "results" / "P1_domain_adaptation", ("representation", "protocol", "fold")):
        rows.append({"representation": record["representation"], "base_representation": record["base_representation"], "protocol": record["protocol"], "fold": record["fold"], "feature_dimension": record["feature_dimension"], "adaptation_scope": record["adaptation_scope"], **record["metrics"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No CORAL runs")
    summary = frame.groupby(["representation", "base_representation", "protocol", "feature_dimension", "adaptation_scope"], as_index=False).agg(
        folds=("fold", "size"), **{f"{metric}_mean": (metric, "mean") for metric in PRIMARY_METRICS},
        **{f"{metric}_std": (metric, "std") for metric in PRIMARY_METRICS},
    )
    exclusive(summary, PROJECT_ROOT / "results" / "P1_domain_adaptation" / "coral_results.csv")
    return summary


def make_table1() -> None:
    manifest = read_csv_rows(PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    folds = {row["subject_id"]: int(row["fold"]) for row in read_csv_rows(PROJECT_ROOT / "metadata" / "subject_folds_5cv_aligned.csv")}
    rows = []
    for outer in range(5):
        val_fold = (outer + 1) % 5
        subject_sets = {
            "train": {subject for subject, fold in folds.items() if fold not in {outer, val_fold}},
            "validation": {subject for subject, fold in folds.items() if fold == val_fold},
            "test": {subject for subject, fold in folds.items() if fold == outer},
        }
        for device in ("R", "S"):
            for split, subjects in subject_sets.items():
                selected = [row for row in manifest if row["device"] == device and row["subject_id"] in subjects]
                rows.append({"outer_fold": outer, "device": device, "split": split, "subjects": len(subjects), "subject_ids": json.dumps(sorted(subjects)), "clips": len(selected), "positive_clips": sum(int(row["label"]) for row in selected), "negative_clips": sum(1 - int(row["label"]) for row in selected), "patient_disjoint": "yes"})
    exclusive(pd.DataFrame(rows), PROJECT_ROOT / "results" / "tables" / "table1_dataset_protocol.csv")


def make_tables() -> None:
    tables = PROJECT_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    gap = pd.read_csv(PROJECT_ROOT / "results" / "P0_device_gap" / "device_gap_summary.csv")
    selection = json.loads((PROJECT_ROOT / "results" / "P0_statistics" / "selection.json").read_text(encoding="utf-8"))
    gap = gap[gap["classifier"] == selection["primary_classifier"]]
    columns = ["representation", "classifier", "balanced_accuracy_R_R", "balanced_accuracy_S_S", "balanced_accuracy_R_S", "balanced_accuracy_S_R", "balanced_accuracy_mean_cross_device", "balanced_accuracy_mean_device_gap"]
    exclusive(gap[columns], tables / "table2_main_benchmark.csv")
    ablation = pd.read_csv(PROJECT_ROOT / "results" / "P0_ablation" / "ablation_summary.csv")
    exclusive(ablation, tables / "table3_fusion_ablation.csv")
    significance = pd.read_csv(PROJECT_ROOT / "results" / "P0_statistics" / "significance_summary.csv")
    exclusive(significance, tables / "table5_statistical_comparison.csv")


def make_system_figure() -> None:
    path = PROJECT_ROOT / "figures" / "sleep_quadnet_system_overview.pdf"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    # Explicit line breaks, not wrap=True: matplotlib's Text(wrap=True)
    # wraps against the *figure* width, not the local box width, so long
    # labels (e.g. "Matched and cross-device evaluation") overflowed their
    # box and collided with the connecting arrows -- a confirmed rendering
    # bug in the previous version of this figure.
    labels = ["Same held-out\npatient", "Paired Recorder\n+ Smartphone", "Measured device\nacoustic shift", "SSL / handcrafted\nrepresentation", "Respiratory-event\nclassifier", "Matched and\ncross-device\nevaluation"]
    figure, axis = plt.subplots(figsize=(11, 2.5))
    axis.set_xlim(0, len(labels)); axis.set_ylim(0, 1); axis.axis("off")
    for index, label in enumerate(labels):
        box = FancyBboxPatch((index + 0.06, 0.3), 0.82, 0.4, boxstyle="round,pad=0.03", facecolor="#e8f0fa", edgecolor="#285f8f")
        axis.add_patch(box)
        axis.text(index + 0.47, 0.5, label, ha="center", va="center", fontsize=7.5, linespacing=1.4)
        if index + 1 < len(labels):
            axis.annotate("", xy=(index + 1.04, 0.5), xytext=(index + 0.9, 0.5), arrowprops={"arrowstyle": "->", "color": "#333333"})
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    preprocessing = aggregate_preprocessing()
    dimension = aggregate_dimension()
    coral = aggregate_coral()
    make_table1()
    make_tables()
    make_system_figure()
    print(json.dumps({"preprocessing_rows": len(preprocessing), "dimension_rows": len(dimension), "coral_rows": len(coral)}, indent=2))


if __name__ == "__main__":
    main()
