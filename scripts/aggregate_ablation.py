#!/usr/bin/env python3
"""Aggregate cross-device encoder ablations and paired subject contributions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.io import load_completed_runs


def scan(root: Path):
    # Dedup by (representation, classifier, protocol, fold) *within this
    # root* before concatenating with the other root below -- a duplicated
    # combo (stale CPU-era or pre-calibration-fix run alongside its
    # GPU-era/post-fix replacement) must resolve to exactly one record
    # first, or a later drop_duplicates() across the concatenated frame
    # could keep the wrong (glob-order-first, not latest) one. See
    # load_completed_runs()'s docstring for the two known root causes.
    records = load_completed_runs(root, ("representation", "classifier", "protocol", "fold"))
    for record in records:
        record["directory"] = record["result_dir"]
    return records


def main() -> None:
    selection = json.loads((PROJECT_ROOT / "results" / "P0_ablation" / "selection.json").read_text(encoding="utf-8"))
    top3 = set(selection["top3_representations"])
    records = scan(PROJECT_ROOT / "results" / "P0_device_gap") + scan(PROJECT_ROOT / "results" / "P0_ablation")
    records = [record for record in records if record["protocol"] in {"R_S", "S_R"} and (record["classifier"] == "svm_rbf" or record["representation"] in top3)]
    fold_rows = []
    subject_frames = []
    for record in records:
        fold_rows.append(
            {"run_id": record["run_id"], "representation": record["representation"], "classifier": record["classifier"],
             "protocol": record["protocol"], "fold": record["fold"], "feature_dimension": record["feature_dimension"], **record["metrics"]}
        )
        subject_path = record["directory"] / "subject_metrics.csv"
        if subject_path.exists():
            frame = pd.read_csv(subject_path, dtype={"subject_id": str})
            frame["subject_id"] = frame["subject_id"].str.zfill(2)
            for key in ("representation", "classifier", "protocol", "fold"):
                frame[key] = fold_rows[-1][key]
            subject_frames.append(frame)
    # No drop_duplicates() needed here -- scan() above already dedupes each
    # root to one record per (representation, classifier, protocol, fold)
    # via load_completed_runs(), keeping the latest-timestamped record
    # rather than whichever the filesystem glob happened to list first.
    fold_frame = pd.DataFrame(fold_rows)
    protocol_summary = fold_frame.groupby(["representation", "classifier", "protocol"], as_index=False).agg(
        folds=("fold", "size"), feature_dimension=("feature_dimension", "first"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), roc_auc_mean=("roc_auc", "mean"), mcc_mean=("mcc", "mean"),
    )
    subject = pd.concat(subject_frames, ignore_index=True)
    svm_subject = subject[subject["classifier"] == "svm_rbf"]
    paired = svm_subject.groupby(["representation", "subject_id"], as_index=False)[["balanced_accuracy", "f1"]].mean()
    ci_rows = []
    for representation, group in paired.groupby("representation"):
        for metric in ("balanced_accuracy", "f1"):
            values = group[metric].to_numpy()
            ci = stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values))
            ci_rows.append({"representation": representation, "metric": metric, "subject_mean": np.mean(values), "subject_std": np.std(values, ddof=1), "ci95_low": ci[0], "ci95_high": ci[1], "subjects": len(values)})
    ci_frame = pd.DataFrame(ci_rows)
    summary_rows = []
    for (representation, classifier), group in protocol_summary.groupby(["representation", "classifier"]):
        indexed = group.set_index("protocol")
        if not {"R_S", "S_R"}.issubset(indexed.index):
            continue
        summary_rows.append(
            {"representation": representation, "classifier": classifier,
             "feature_dimension": int(group["feature_dimension"].iloc[0]), "folds_per_protocol": int(group["folds"].min()),
             "balanced_accuracy_R_S": indexed.loc["R_S", "balanced_accuracy_mean"],
             "balanced_accuracy_S_R": indexed.loc["S_R", "balanced_accuracy_mean"],
             "balanced_accuracy_mean_cross_device": np.mean([indexed.loc["R_S", "balanced_accuracy_mean"], indexed.loc["S_R", "balanced_accuracy_mean"]]),
             "f1_R_S": indexed.loc["R_S", "f1_mean"], "f1_S_R": indexed.loc["S_R", "f1_mean"],
             "f1_mean_cross_device": np.mean([indexed.loc["R_S", "f1_mean"], indexed.loc["S_R", "f1_mean"]])}
        )
    summary = pd.DataFrame(summary_rows)
    ci_wide = ci_frame.pivot(index="representation", columns="metric", values=["subject_mean", "ci95_low", "ci95_high"])
    ci_wide.columns = [f"{metric}_{stat}" for stat, metric in ci_wide.columns]
    ci_wide = ci_wide.reset_index()
    summary = summary.merge(ci_wide, on="representation", how="left")
    full = paired[paired["representation"] == "full_fusion"].set_index("subject_id")
    contribution_rows = []
    for representation, group in paired.groupby("representation"):
        if not representation.startswith("full_minus_"):
            continue
        candidate = group.set_index("subject_id")
        common = full.index.intersection(candidate.index)
        for metric in ("balanced_accuracy", "f1"):
            difference = full.loc[common, metric] - candidate.loc[common, metric]
            ci = stats.t.interval(0.95, len(difference) - 1, loc=difference.mean(), scale=stats.sem(difference))
            contribution_rows.append({"removed_configuration": representation, "metric": metric, "full_minus_ablation_difference": difference.mean(), "ci95_low": ci[0], "ci95_high": ci[1], "subjects": len(difference)})
    root = PROJECT_ROOT / "results" / "P0_ablation"
    outputs = {
        root / "ablation_fold_metrics.csv": fold_frame,
        root / "ablation_summary.csv": summary,
        root / "encoder_contribution.csv": pd.DataFrame(contribution_rows),
    }
    for path, frame in outputs.items():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        frame.to_csv(path, index=False)
    plot = ci_frame[ci_frame["metric"] == "balanced_accuracy"].sort_values("subject_mean", ascending=False)
    figure_path = PROJECT_ROOT / "figures" / "encoder_ablation_cross_device.pdf"
    if figure_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {figure_path}")
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(plot))
    error = np.vstack([plot["subject_mean"] - plot["ci95_low"], plot["ci95_high"] - plot["subject_mean"]])
    axis.errorbar(x, plot["subject_mean"], yerr=error, fmt="o", capsize=3)
    axis.set_xticks(x, plot["representation"], rotation=45, ha="right")
    axis.set(ylabel="Mean cross-device subject balanced accuracy", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"fold_rows": len(fold_frame), "summary_rows": len(summary), "top3": sorted(top3)}, indent=2))


if __name__ == "__main__":
    main()
