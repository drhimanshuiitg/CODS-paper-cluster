#!/usr/bin/env python3
"""Compact progress bars for feature caches and experiment fold runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_REPRESENTATIONS = {
    "hubert", "wavlm", "wav2vec2", "data2vec_audio",
    "data2vec_spectrogram", "data2vec_fusion", "full_fusion",
}


def bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done:,}/{total:,} ({100 * done / total if total else 0:.2f}%)"


def expected_ablation_runs() -> int:
    """Stage 1 has 60 new leave-out runs; Stage 2 reuses main runs when possible."""
    selection_path = PROJECT_ROOT / "results" / "P0_ablation" / "selection.json"
    if not selection_path.exists():
        return 60
    selected = json.loads(selection_path.read_text(encoding="utf-8"))["top3_representations"]
    new_representations = sum(name not in BASE_REPRESENTATIONS for name in selected)
    # SVM is already present from Stage 1 for a leave-out representation, so
    # Stage 2 adds three classifiers x two protocols x five folds.
    return 60 + new_representations * 3 * 2 * 5


def main() -> None:
    main_features = [
        ("classical", "peak_filter"), ("hubert", "peak"), ("wavlm", "peak"),
        ("wav2vec2", "peak"), ("data2vec_audio", "peak"),
        ("data2vec_spectrogram", "peak"),
    ]
    preprocessing_features = [("classical", name) for name in ("raw", "peak", "filter", "peak_filter")]
    reference = PROJECT_ROOT / "cached_features" / "classical" / "peak_filter" / "complete.npy"
    rows = len(np.load(reference, mmap_mode="r")) if reference.exists() else 0
    for label, specs in (("main feature caches", main_features), ("preprocessing caches", preprocessing_features)):
        done = 0
        for feature, preprocessing in specs:
            path = PROJECT_ROOT / "cached_features" / feature / preprocessing / "complete.npy"
            if path.exists():
                done += int(np.load(path, mmap_mode="r").sum())
        print(f"{label}: {bar(done, rows * len(specs))}")
    for path in sorted((PROJECT_ROOT / "cached_features").glob("*/*/complete.npy")):
        complete = np.load(path, mmap_mode="r")
        print(f"feature {path.parent.relative_to(PROJECT_ROOT / 'cached_features')}: {bar(int(complete.sum()), len(complete))}")
    expected = {
        "P0_device_gap": 8 * 4 * 4 * 5 + 8 * 1 * 1 * 5,
        "P0_ablation": expected_ablation_runs(),
        "P1_preprocessing": 4 * 4 * 5,
        "P1_dimension_control": 3 * 2 * 5,
        "P1_domain_adaptation": 3 * 2 * 5,
    }
    for experiment, total in expected.items():
        completed = 0
        for path in (PROJECT_ROOT / "results" / experiment / "runs").glob("*/completion.json"):
            try:
                completed += json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
            except Exception:
                pass
        print(f"runs {experiment}: {bar(completed, total)}")


if __name__ == "__main__":
    main()
