#!/usr/bin/env python3
"""Paired subject-level bootstrap significance for the leave-one-encoder-out
ablation (results/P0_ablation) against the full_fusion baseline
(results/P0_device_gap), same test set both arms, same primary classifier
selection logic as run_statistics.py."""
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
ABLATION_REPS = ("full_minus_hubert", "full_minus_wavlm", "full_minus_wav2vec2", "full_minus_data2vec_audio", "full_minus_data2vec_spectrogram", "full_minus_data2vec")
CLASSIFIERS = ("svm_rbf", "mlp", "random_forest", "xgboost")
PROTOCOLS = ("R_R", "S_S", "R_S", "S_R")


def records(root: Path) -> list[dict]:
    output = load_completed_runs(root, ("representation", "classifier", "protocol", "fold"))
    for record in output:
        record["directory"] = record["result_dir"]
    return output


def load_predictions(all_records: list[dict], representation: str, classifier: str, protocol: str) -> pd.DataFrame:
    selected = [r for r in all_records if r["representation"] == representation and r["classifier"] == classifier and r["protocol"] == protocol]
    have = {int(r["fold"]) for r in selected}
    if have != set(range(5)):
        raise ValueError(f"Incomplete 5-fold predictions for {representation}/{classifier}/{protocol}: have {sorted(have)}")
    frames = []
    for record in selected:
        with gzip.open(record["directory"] / "window_predictions.csv.gz", "rt", encoding="utf-8") as handle:
            frames.append(pd.read_csv(handle, dtype={"subject_id": str}))
    frame = pd.concat(frames, ignore_index=True)
    frame["subject_id"] = frame["subject_id"].str.zfill(2)
    return frame


def subject_scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        result = metrics(group["label"].to_numpy(), group["probability"].to_numpy())
        rows.append({"subject_id": subject, **{name: float(result[name]) for name in METRICS}})
    return pd.DataFrame(rows).set_index("subject_id")


def align_pair(left, right):
    keys = ["subject_id", "logical_window_id", "label"]
    common = left[keys].merge(right[keys], on=keys, how="inner")
    left_aligned = common.merge(left, on=keys, how="left")
    right_aligned = common.merge(right, on=keys, how="left")
    if len(left_aligned) != len(left) or len(right_aligned) != len(right):
        raise ValueError("Ablation vs baseline test sets do not align")
    return left_aligned, right_aligned


def bootstrap_difference(left, right, rng, iterations):
    left, right = align_pair(left, right)
    left_scores, right_scores = subject_scores(left), subject_scores(right)
    common = left_scores.index.intersection(right_scores.index)
    difference = left_scores.loc[common, list(METRICS)] - right_scores.loc[common, list(METRICS)]
    draws = rng.integers(0, len(common), size=(iterations, len(common)))
    return {name: difference[name].to_numpy()[draws].mean(axis=1) for name in METRICS}


def interval(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def save_exclusive(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT / "results" / "P0_device_gap")
    parser.add_argument("--ablation-root", type=Path, default=PROJECT_ROOT / "results" / "P0_ablation")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "P0_ablation_statistics")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_records = records(args.baseline_root)
    ablation_records = records(args.ablation_root)
    rng = np.random.default_rng(args.seed)
    combos = [(rep, clf, proto) for rep in ABLATION_REPS for clf in CLASSIFIERS for proto in PROTOCOLS]

    rows, skipped = [], []
    for representation, classifier, protocol in tqdm(combos, desc="ablation vs full_fusion", unit="combo"):
        try:
            ablation_frame = load_predictions(ablation_records, representation, classifier, protocol)
            baseline_frame = load_predictions(baseline_records, "full_fusion", classifier, protocol)
        except ValueError as exc:
            skipped.append({"representation": representation, "classifier": classifier, "protocol": protocol, "reason": str(exc)})
            continue
        distributions = bootstrap_difference(ablation_frame, baseline_frame, rng, args.iterations)
        ablation_point = subject_scores(ablation_frame).mean().to_dict()
        baseline_point = subject_scores(baseline_frame).mean().to_dict()
        for name, values in distributions.items():
            mean, std, low, high = interval(values)
            p_value = min(1.0, 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
            rows.append({
                "representation": representation, "classifier": classifier, "protocol": protocol, "metric": name,
                "ablation_point": ablation_point[name], "full_fusion_point": baseline_point[name],
                "point_difference": ablation_point[name] - baseline_point[name], "bootstrap_mean_difference": mean,
                "bootstrap_std": std, "ci95_low": low, "ci95_high": high, "p_value_two_sided": p_value,
                "bootstrap_iterations": args.iterations, "unit": "subject",
                "interpretation": "CI excludes zero" if low > 0 or high < 0 else "CI includes zero",
            })

    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    save_exclusive(frame, args.output_root / "ablation_vs_full_fusion.csv")
    if skipped:
        save_exclusive(pd.DataFrame(skipped), args.output_root / "skipped_combos.csv")

    ba = frame[frame["metric"] == "balanced_accuracy"]
    sig = ba[(ba["ci95_low"] > 0) | (ba["ci95_high"] < 0)]
    print(json.dumps({"total_combos": len(combos), "compared": len(combos) - len(skipped), "skipped": len(skipped)}, indent=2))
    print(f"balanced_accuracy: n={len(ba)} mean_diff={ba['point_difference'].mean():+.4f} significant={len(sig)}/{len(ba)} "
          f"positive={int((sig['point_difference']>0).sum())} negative={int((sig['point_difference']<0).sum())}")


if __name__ == "__main__":
    main()
