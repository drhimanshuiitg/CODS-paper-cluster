#!/usr/bin/env python3
"""Paired subject-level bootstrap comparison: target-aware PCA refit fix
(results/P1_dimension_control_v3) vs. the pre-fix, source-only-fit PCA
(results/P1_dimension_control), on the exact same (representation,
classifier, protocol, fold, dimension) combos. Test set is identical in
both arms (PCA/CORAL never touch it during fitting), so any paired
difference is attributable to the refit-scope change alone.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.evaluation import metrics
from sleep_quadnet.io import load_completed_runs

METRICS = ("f1", "balanced_accuracy", "roc_auc", "mcc")


def records(root: Path) -> list[dict]:
    output = load_completed_runs(root, ("representation", "classifier", "protocol", "fold", "feature_dimension"))
    for record in output:
        record["directory"] = record["result_dir"]
    return output


def load_predictions_for_key(all_records: list[dict], base_representation: str, classifier: str, protocol: str, fold: int, dimension: int) -> pd.DataFrame:
    matches = [
        r for r in all_records
        if r["base_representation"] == base_representation and r["classifier"] == classifier
        and r["protocol"] == protocol and r["fold"] == fold and r["feature_dimension"] == dimension
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 match for {(base_representation, classifier, protocol, fold, dimension)}, got {len(matches)}")
    with gzip.open(matches[0]["directory"] / "window_predictions.csv.gz", "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle, dtype={"subject_id": str})
    frame["subject_id"] = frame["subject_id"].str.zfill(2)
    return frame


def subject_scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        result = metrics(group["label"].to_numpy(), group["probability"].to_numpy())
        rows.append({"subject_id": subject, **{name: float(result[name]) for name in METRICS}})
    return pd.DataFrame(rows).set_index("subject_id")


def align_pair(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["subject_id", "logical_window_id", "label"]
    common = left[keys].merge(right[keys], on=keys, how="inner")
    left_aligned = common.merge(left, on=keys, how="left")
    right_aligned = common.merge(right, on=keys, how="left")
    if len(left_aligned) != len(left) or len(right_aligned) != len(right):
        raise ValueError("Fixed vs baseline PCA test sets do not align -- expected identical test sets")
    return left_aligned, right_aligned


def bootstrap_difference(left: pd.DataFrame, right: pd.DataFrame, rng: np.random.Generator, iterations: int):
    left, right = align_pair(left, right)
    left_scores = subject_scores(left)
    right_scores = subject_scores(right)
    common = left_scores.index.intersection(right_scores.index)
    difference = left_scores.loc[common, list(METRICS)] - right_scores.loc[common, list(METRICS)]
    draws = rng.integers(0, len(common), size=(iterations, len(common)))
    return {name: difference[name].to_numpy()[draws].mean(axis=1) for name in METRICS}


def interval(values: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def save_exclusive(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite statistical result: {path}")
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT / "results" / "P1_dimension_control")
    parser.add_argument("--fixed-root", type=Path, default=PROJECT_ROOT / "results" / "P1_dimension_control_v3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "P1_statistics_pca_fix")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_records = records(args.baseline_root)
    fixed_records = records(args.fixed_root)
    baseline_keys = {(r["base_representation"], r["classifier"], r["protocol"], r["fold"], r["feature_dimension"]) for r in baseline_records}
    fixed_keys = {(r["base_representation"], r["classifier"], r["protocol"], r["fold"], r["feature_dimension"]) for r in fixed_records}
    overlap = sorted(baseline_keys & fixed_keys)
    rng = np.random.default_rng(args.seed)

    rows = []
    for base_representation, classifier, protocol, fold, dimension in tqdm(overlap, desc="pca fix vs baseline", unit="combo"):
        fixed_frame = load_predictions_for_key(fixed_records, base_representation, classifier, protocol, fold, dimension)
        baseline_frame = load_predictions_for_key(baseline_records, base_representation, classifier, protocol, fold, dimension)
        distributions = bootstrap_difference(fixed_frame, baseline_frame, rng, args.iterations)
        fixed_point = subject_scores(fixed_frame).mean().to_dict()
        baseline_point = subject_scores(baseline_frame).mean().to_dict()
        for name, values in distributions.items():
            mean, std, low, high = interval(values)
            p_value = min(1.0, 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
            rows.append({
                "representation": base_representation, "classifier": classifier, "protocol": protocol, "fold": fold,
                "dimension": dimension, "metric": name, "fixed_point": fixed_point[name], "baseline_point": baseline_point[name],
                "point_difference": fixed_point[name] - baseline_point[name], "bootstrap_mean_difference": mean,
                "bootstrap_std": std, "ci95_low": low, "ci95_high": high, "p_value_two_sided": p_value,
                "bootstrap_iterations": args.iterations, "unit": "subject",
                "interpretation": "CI excludes zero" if low > 0 or high < 0 else "CI includes zero",
            })

    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    save_exclusive(frame, args.output_root / "pca_fix_vs_baseline.csv")

    summary = frame[frame["metric"].isin(["balanced_accuracy", "f1"])]
    print(json.dumps({"overlap_combos": len(overlap)}, indent=2))
    for metric in ("balanced_accuracy", "f1"):
        m = summary[summary["metric"] == metric]
        sig = m[(m["ci95_low"] > 0) | (m["ci95_high"] < 0)]
        print(
            f"{metric:20s} n={len(m):3d}  mean_diff={m['point_difference'].mean():+.4f}  "
            f"significant(CI excl. 0)={len(sig)}/{len(m)}  "
            f"significant_positive={int((sig['point_difference'] > 0).sum())}  "
            f"significant_negative={int((sig['point_difference'] < 0).sum())}  "
            f"collapsed_baseline_rows(BA==0.5 exactly)={int((m['baseline_point'] == 0.5).sum()) if metric=='balanced_accuracy' else '-'}"
        )


if __name__ == "__main__":
    main()
