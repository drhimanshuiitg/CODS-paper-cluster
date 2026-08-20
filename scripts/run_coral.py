#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.advanced import run_coral_fold
from sleep_quadnet.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True)
    parser.add_argument("--protocol", choices=["R_S", "S_R"], required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    # Added 2026-08-19 (ARA Level 2 review finding F04/G3): CORAL was only
    # ever run with svm_rbf, giving it much narrower evidentiary coverage
    # than the PCA-fix comparison it's rhetorically paired against (C05 vs
    # C06). run_coral_fold already accepted a classifier parameter
    # (default svm_rbf); this CLI just never exposed it.
    parser.add_argument("--classifier", default="svm_rbf")
    args = parser.parse_args()
    result = run_coral_fold(
        config=load_yaml(PROJECT_ROOT / "configs" / "base.yaml"), manifest_path=PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv",
        fold_path=PROJECT_ROOT / "metadata" / "subject_folds_5cv_aligned.csv", cache_root=PROJECT_ROOT / "cached_features",
        results_root=PROJECT_ROOT / "results" / "P1_domain_adaptation", representation=args.representation,
        protocol=args.protocol, fold=args.fold, classifier=args.classifier,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
