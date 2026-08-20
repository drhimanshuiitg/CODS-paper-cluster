#!/usr/bin/env python3
"""ARA Level 2 review finding F03: C09 claimed hubert_odi_hb is "statistically
indistinguishable" from hubert alone with no direct paired-bootstrap test
cited. This script runs that specific test."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.evaluation import metrics
from sleep_quadnet.io import load_completed_runs

METRICS = ("f1", "balanced_accuracy", "roc_auc", "mcc")
CLASSIFIERS = ("svm_rbf", "mlp", "random_forest", "xgboost")
PROTOCOLS = ("R_S", "S_R")


def records(root: Path) -> list[dict]:
    output = load_completed_runs(root, ("representation", "classifier", "protocol", "fold"))
    for r in output:
        r["directory"] = r["result_dir"]
    return output


def load_predictions(all_records, representation, classifier, protocol) -> pd.DataFrame:
    selected = [r for r in all_records if r["representation"] == representation and r["classifier"] == classifier and r["protocol"] == protocol]
    have = {int(r["fold"]) for r in selected}
    if have != set(range(5)):
        raise ValueError(f"Incomplete 5-fold predictions for {representation}/{classifier}/{protocol}: have {sorted(have)}")
    frames = []
    for r in selected:
        with gzip.open(r["directory"] / "window_predictions.csv.gz", "rt", encoding="utf-8") as handle:
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
        raise ValueError("hubert_odi_hb vs hubert test sets do not align")
    return left_aligned, right_aligned


def bootstrap_difference(left, right, rng, iterations):
    left, right = align_pair(left, right)
    left_scores, right_scores = subject_scores(left), subject_scores(right)
    common = left_scores.index.intersection(right_scores.index)
    diff = left_scores.loc[common, list(METRICS)] - right_scores.loc[common, list(METRICS)]
    draws = rng.integers(0, len(common), size=(iterations, len(common)))
    return {name: diff[name].to_numpy()[draws].mean(axis=1) for name in METRICS}


def interval(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    root = PROJECT_ROOT / "results" / "P0_device_gap"
    all_records = records(root)
    rng = np.random.default_rng(42)
    rows = []
    for classifier in CLASSIFIERS:
        for protocol in PROTOCOLS:
            try:
                hub = load_predictions(all_records, "hubert", classifier, protocol)
                odi = load_predictions(all_records, "hubert_odi_hb", classifier, protocol)
            except ValueError as exc:
                print(f"skip {classifier}/{protocol}: {exc}")
                continue
            dist = bootstrap_difference(odi, hub, rng, 2000)
            odi_point = subject_scores(odi).mean().to_dict()
            hub_point = subject_scores(hub).mean().to_dict()
            for name, values in dist.items():
                mean, std, low, high = interval(values)
                p = min(1.0, 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
                rows.append({
                    "classifier": classifier, "protocol": protocol, "metric": name,
                    "hubert_odi_hb_point": odi_point[name], "hubert_point": hub_point[name],
                    "point_difference": odi_point[name] - hub_point[name],
                    "ci95_low": low, "ci95_high": high, "p_value_two_sided": p,
                    "interpretation": "CI excludes zero" if low > 0 or high < 0 else "CI includes zero",
                })
    frame = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "results" / "P0_statistics_hubert_vs_hubert_odi_hb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hubert_odi_hb_vs_hubert.csv"
    if out_path.exists():
        raise FileExistsError(out_path)
    frame.to_csv(out_path, index=False)
    ba = frame[frame.metric == "balanced_accuracy"]
    sig = ba[(ba.ci95_low > 0) | (ba.ci95_high < 0)]
    print(json.dumps({"n": len(ba), "significant": len(sig), "mean_point_diff": float(ba.point_difference.mean())}, indent=2))
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
