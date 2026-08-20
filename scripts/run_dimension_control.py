#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.advanced import run_pca_fold
from sleep_quadnet.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["R_S", "S_R"], required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--representation", default="full_fusion", help="Base representation to run PCA over (must be a concatenation-style entry in configs/base.yaml).")
    parser.add_argument("--classifier", default="svm_rbf")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "P1_dimension_control")
    parser.add_argument(
        "--dimensions", default="1536,768,384",
        help="Comma-separated PCA target dimensions. Must each be strictly less than the representation's "
             "native dimension (e.g. data2vec_fusion is exactly 1536-d natively, so 1536 is not a real "
             "reduction and run_pca_fold rejects it -- pass e.g. --dimensions 768,384 for that case).",
    )
    args = parser.parse_args()
    result = run_pca_fold(
        config=load_yaml(PROJECT_ROOT / "configs" / "base.yaml"), manifest_path=PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv",
        fold_path=PROJECT_ROOT / "metadata" / "subject_folds_5cv_aligned.csv", cache_root=PROJECT_ROOT / "cached_features",
        results_root=args.results_root, protocol=args.protocol, fold=args.fold, representation=args.representation,
        classifier=args.classifier, dimensions=tuple(int(x) for x in args.dimensions.split(",")),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
