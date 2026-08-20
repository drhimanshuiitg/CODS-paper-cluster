#!/usr/bin/env python3
"""Resume-safe feature extraction entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.features import FEATURE_DIMENSIONS, extract_feature_cache
from sleep_quadnet.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "cached_features")
    parser.add_argument("--feature", choices=sorted(FEATURE_DIMENSIONS), required=True)
    parser.add_argument("--preprocessing", choices=["raw", "peak", "filter", "peak_filter"])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = extract_feature_cache(
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        config=load_yaml(args.config),
        feature=args.feature,
        preprocessing=args.preprocessing,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
