#!/usr/bin/env python3
"""Subject-level bootstrap confidence intervals and paired comparisons for P0-C."""

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
    # Dedup by (representation, classifier, protocol, fold) -- without this,
    # choose_primary_classifier() below would silently average in stale
    # CPU-era (or pre-calibration-fix) duplicate runs alongside their
    # GPU-era/post-fix replacements, which could change which classifier is
    # selected as the paper's primary one. See load_completed_runs()'s
    # docstring for the two known root causes.
    output = load_completed_runs(root, ("representation", "classifier", "protocol", "fold"))
    for record in output:
        record["directory"] = record["result_dir"]
    return output


def choose_primary_classifier(all_records: list[dict]) -> tuple[str, pd.DataFrame]:
    rows = []
    for record in all_records:
        if record["protocol"] not in {"R_S", "S_R"}:
            continue
        rows.append(
            {
                "classifier": record["classifier"],
                "representation": record["representation"],
                "protocol": record["protocol"],
                "fold": record["fold"],
                "validation_balanced_accuracy": max(item["validation_balanced_accuracy"] for item in record["tuning"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No cross-device validation records available")
    summary = frame.groupby("classifier", as_index=False).agg(
        validation_balanced_accuracy_mean=("validation_balanced_accuracy", "mean"),
        validation_balanced_accuracy_std=("validation_balanced_accuracy", "std"),
        validation_runs=("validation_balanced_accuracy", "size"),
    )
    summary = summary.sort_values(
        ["validation_balanced_accuracy_mean", "validation_balanced_accuracy_std", "classifier"],
        ascending=[False, True, True],
    )
    return str(summary.iloc[0]["classifier"]), summary


def load_predictions(all_records: list[dict], representation: str, classifier: str, protocol: str) -> pd.DataFrame:
    selected = [
        record for record in all_records
        if record["representation"] == representation and record["classifier"] == classifier and record["protocol"] == protocol
    ]
    if {int(record["fold"]) for record in selected} != set(range(5)):
        raise ValueError(f"Incomplete 5-fold predictions for {representation}/{classifier}/{protocol}")
    frames = []
    for record in selected:
        with gzip.open(record["directory"] / "window_predictions.csv.gz", "rt", encoding="utf-8") as handle:
            frames.append(pd.read_csv(handle, dtype={"subject_id": str}))
    frame = pd.concat(frames, ignore_index=True)
    frame["subject_id"] = frame["subject_id"].str.zfill(2)
    if frame.duplicated(["sample_id"]).any():
        raise ValueError("Duplicate test prediction IDs across folds")
    return frame


def subject_scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        result = metrics(group["label"].to_numpy(), group["probability"].to_numpy())
        rows.append({"subject_id": subject, **{name: float(result[name]) for name in METRICS}})
    return pd.DataFrame(rows).set_index("subject_id")


def bootstrap_single(frame: pd.DataFrame, rng: np.random.Generator, iterations: int) -> dict[str, np.ndarray]:
    scores = subject_scores(frame)
    draws = rng.integers(0, len(scores), size=(iterations, len(scores)))
    return {name: scores[name].to_numpy()[draws].mean(axis=1) for name in METRICS}


def align_pair(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["subject_id", "logical_window_id", "label"]
    common = left[keys].merge(right[keys], on=keys, how="inner")
    left_aligned = common.merge(left, on=keys, how="left")
    right_aligned = common.merge(right, on=keys, how="left")
    if len(left_aligned) != len(left) or len(right_aligned) != len(right):
        raise ValueError("Paired comparison prediction sets do not align")
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
    parser.add_argument("--main-root", type=Path, default=PROJECT_ROOT / "results" / "P0_device_gap")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "P0_statistics")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    all_records = records(args.main_root)
    classifier, classifier_summary = choose_primary_classifier(all_records)
    # Extended 2026-08-19 (ARA Level 2 review finding F05/G1): wavlm_large and
    # hear were added to the main benchmark after this candidate list was
    # last set, so C01/C08's "best/worst representation" point-estimate
    # claims were never actually run through this significance machinery
    # against them. full_fusion_plus_hear is not a single encoder so it does
    # not belong in candidate_singles, but is added to `principal` below
    # (with its own comparison spec) for the same reason.
    candidate_singles = ["hubert", "wavlm", "wavlm_large", "wav2vec2", "data2vec_audio", "data2vec_spectrogram", "hear"]
    validation_rows = []
    for representation in candidate_singles:
        selected = [r for r in all_records if r["representation"] == representation and r["classifier"] == classifier and r["protocol"] in {"R_S", "S_R"}]
        if len(selected) == 10:
            validation_rows.append(
                (representation, np.mean([max(item["validation_balanced_accuracy"] for item in r["tuning"]) for r in selected]))
            )
    if not validation_rows:
        raise ValueError("No complete individual-encoder validation records")
    best_single = max(validation_rows, key=lambda item: (item[1], item[0]))[0]
    principal = ["classical", best_single, "data2vec_fusion", "full_fusion", "full_fusion_plus_hear"]
    rng = np.random.default_rng(args.seed)
    ci_rows = []
    prediction_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for representation in principal:
        for protocol in ("R_R", "S_S", "R_S", "S_R"):
            frame = load_predictions(all_records, representation, classifier, protocol)
            prediction_cache[(representation, protocol)] = frame
            bootstrap = bootstrap_single(frame, rng, args.iterations)
            point = subject_scores(frame).mean().to_dict()
            for name, values in bootstrap.items():
                mean, std, low, high = interval(values)
                ci_rows.append(
                    {"representation": representation, "classifier": classifier, "protocol": protocol, "metric": name,
                     "point_estimate": point[name], "bootstrap_mean": mean, "bootstrap_std": std,
                     "ci95_low": low, "ci95_high": high, "bootstrap_iterations": args.iterations, "unit": "subject"}
                )
    comparisons = []
    specs = []
    for protocol in ("R_S", "S_R"):
        specs.extend(
            [
                (f"full_fusion_vs_data2vec_fusion_{protocol}", ("full_fusion", protocol), ("data2vec_fusion", protocol)),
                (f"full_fusion_vs_{best_single}_{protocol}", ("full_fusion", protocol), (best_single, protocol)),
                (f"{best_single}_vs_classical_{protocol}", (best_single, protocol), ("classical", protocol)),
                # Added 2026-08-19 (F05/G1): direct significance test for C08's
                # "HeAR fused in makes full_fusion worse" claim, previously a
                # point-estimate-only comparison.
                (f"full_fusion_plus_hear_vs_full_fusion_{protocol}", ("full_fusion_plus_hear", protocol), ("full_fusion", protocol)),
            ]
        )
    specs.extend(
        [
            ("full_fusion_R_R_vs_R_S", ("full_fusion", "R_R"), ("full_fusion", "R_S")),
            ("full_fusion_S_S_vs_S_R", ("full_fusion", "S_S"), ("full_fusion", "S_R")),
        ]
    )
    for comparison, left_key, right_key in tqdm(specs, desc="paired subject bootstrap", unit="comparison"):
        distributions = bootstrap_difference(prediction_cache[left_key], prediction_cache[right_key], rng, args.iterations)
        left_point = subject_scores(prediction_cache[left_key]).mean().to_dict()
        right_point = subject_scores(prediction_cache[right_key]).mean().to_dict()
        for name, values in distributions.items():
            mean, std, low, high = interval(values)
            p_value = min(1.0, 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
            comparisons.append(
                {"comparison": comparison, "left": f"{left_key[0]}/{left_key[1]}", "right": f"{right_key[0]}/{right_key[1]}",
                 "classifier": classifier, "metric": name, "point_difference": left_point[name] - right_point[name],
                 "bootstrap_mean_difference": mean, "bootstrap_std": std, "ci95_low": low, "ci95_high": high,
                 "p_value_two_sided": p_value, "bootstrap_iterations": args.iterations, "unit": "subject",
                 "interpretation": "CI excludes zero" if low > 0 or high < 0 else "CI includes zero"}
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_exclusive(pd.DataFrame(ci_rows), args.output_root / "confidence_intervals.csv")
    comparison_frame = pd.DataFrame(comparisons)
    save_exclusive(comparison_frame, args.output_root / "bootstrap_differences.csv")
    save_exclusive(comparison_frame, args.output_root / "significance_summary.csv")
    save_exclusive(classifier_summary, args.output_root / "validation_only_classifier_selection.csv")
    (args.output_root / "selection.json").write_text(
        json.dumps({"primary_classifier": classifier, "best_single_encoder": best_single, "selection_basis": "mean cross-device validation balanced accuracy", "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"primary_classifier": classifier, "best_single_encoder": best_single, "confidence_interval_rows": len(ci_rows), "comparison_rows": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
