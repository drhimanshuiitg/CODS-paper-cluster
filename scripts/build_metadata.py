#!/usr/bin/env python3
"""Create the fixed paired-device manifest and five-fold split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.metadata import write_metadata_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/scratch/pkdas/IEEE_healthcomm_workshop/dataset/V5/Data"),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "metadata")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--exclude-wake-events", action="store_true")
    args = parser.parse_args()
    summary = write_metadata_bundle(
        args.dataset_dir,
        args.output_dir,
        seed=args.seed,
        n_folds=args.folds,
        exclude_wake_events=args.exclude_wake_events,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
