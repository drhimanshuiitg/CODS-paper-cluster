#!/usr/bin/env python3
"""Independently validate the materialized paired manifest and every fold/protocol split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.evaluation import validate_manifest_and_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    parser.add_argument("--folds", type=Path, default=PROJECT_ROOT / "metadata" / "subject_folds_5cv_aligned.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "audit" / "aligned_split_validation.json")
    args = parser.parse_args()
    result = validate_manifest_and_folds(args.manifest, args.folds)
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous != result:
            raise FileExistsError(f"Refusing to overwrite different validation output: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
