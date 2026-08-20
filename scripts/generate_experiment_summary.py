#!/usr/bin/env python3
"""Generate a factual final project summary only from materialized results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_REPRESENTATIONS = {
    "hubert", "wavlm", "wav2vec2", "data2vec_audio",
    "data2vec_spectrogram", "data2vec_fusion", "full_fusion",
}


def completed_runs(root: Path) -> int:
    count = 0
    for path in root.glob("runs/*/completion.json"):
        try:
            count += json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
        except Exception:
            pass
    return count


def expected_ablation_runs(result_root: Path) -> int:
    """Count only newly materialized runs; eligible main-benchmark runs are reused."""
    selection_path = result_root / "P0_ablation" / "selection.json"
    if not selection_path.exists():
        return 60
    selected = json.loads(selection_path.read_text(encoding="utf-8"))["top3_representations"]
    new_representations = sum(name not in BASE_REPRESENTATIONS for name in selected)
    return 60 + new_representations * 3 * 2 * 5


def main() -> None:
    result_root = PROJECT_ROOT / "results"
    experiments = {
        "P0_device_gap": 680,
        "P0_ablation": expected_ablation_runs(result_root),
        "P1_preprocessing": 80,
        "P1_dimension_control": 30,
        "P1_domain_adaptation": 30,
    }
    status = {name: {"completed_fold_runs": completed_runs(result_root / name), "expected_fold_runs": expected} for name, expected in experiments.items()}
    status["P0_statistics"] = {"complete": (result_root / "P0_statistics" / "confidence_intervals.csv").exists()}
    status["P0_efficiency"] = {"complete": (result_root / "P0_efficiency" / "runtime_benchmark.csv").exists()}
    status["P1_device_acoustics"] = {"complete": (result_root / "P1_device_acoustics" / "spectral_statistics.csv").exists()}
    tables = sorted(str(path.relative_to(PROJECT_ROOT)) for path in (result_root / "tables").glob("*.csv")) if (result_root / "tables").exists() else []
    figures = sorted(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "figures").glob("*.pdf"))
    numeric_lines = []
    gap_path = result_root / "P0_device_gap" / "device_gap_summary.csv"
    if gap_path.exists():
        gap = pd.read_csv(gap_path)
        selection_path = result_root / "P0_statistics" / "selection.json"
        classifier = json.loads(selection_path.read_text(encoding="utf-8"))["primary_classifier"] if selection_path.exists() else None
        if classifier:
            gap = gap[gap["classifier"] == classifier]
        gap = gap.sort_values("balanced_accuracy_mean_cross_device", ascending=False)
        for _, row in gap.head(5).iterrows():
            numeric_lines.append(
                f"- {row['representation']} / {row['classifier']}: mean cross-device BA "
                f"{row['balanced_accuracy_mean_cross_device']:.4f}, F1 {row['f1_mean_cross_device']:.4f}, "
                f"mean BA device gap {row['balanced_accuracy_mean_device_gap']:.4f}."
            )
    if not numeric_lines:
        numeric_lines = ["- No final benchmark metric is reported because the aggregate result does not yet exist."]
    failed_or_pending = []
    for name, item in status.items():
        if "expected_fold_runs" in item and item["completed_fold_runs"] < item["expected_fold_runs"]:
            failed_or_pending.append(f"{name}: {item['completed_fold_runs']}/{item['expected_fold_runs']} fold-runs")
        elif item.get("complete") is False:
            failed_or_pending.append(f"{name}: pending")
    alignment = json.loads((PROJECT_ROOT / "metadata" / "aligned_metadata_validation.json").read_text(encoding="utf-8"))
    text = f"""# Sleep-QuadNet experiment summary

## 1. Experiments defined by the authoritative strategy

P0-A matched/cross-device benchmark; P0-B Data2Vec and leave-one-encoder-out fusion ablations; P0-C subject-level bootstrap robustness; P0-D computational cost; P1-A paired device acoustics; P1-B preprocessing ablation; P1-C fold-local PCA dimension control; P1-D validation-target CORAL; and the combined-device recovery protocol where useful. The optional learnable P2 fusion was intentionally not started before the reviewer-critical experiments.

## 2. Previous work available

The original project supplied a v4 runner and configuration, but no valid results, checkpoints, feature caches, logs, extracted labelled clips, or SLURM scripts were present. Its supplied cross-device mode was not patient-disjoint.

## 3. Newly implemented

Annotation-driven paired window materialization, label-free device-time alignment, fixed shared five-fold subject splits, independent leakage/window audits, resumable feature caches, validation-only classifier tuning, all plan-required experiment runners, append-only master logging, SLURM workflows, statistical/efficiency/acoustics analyses, and vector table/figure generation.

## 4. Completion status

```json
{json.dumps(status, indent=2)}
```

## 5. Failed or pending

{chr(10).join(f'- {item}' for item in failed_or_pending) if failed_or_pending else '- None.'}

## 6. Main numerical results

{chr(10).join(numeric_lines)}

Dataset/evaluation facts: 50 annotation subjects; 43 with both device files; 41 with reliable synchronized regions; {alignment['logical_windows']:,} paired logical windows; {alignment['manifest_rows']:,} device rows; five fixed patient-disjoint folds.

## 7. Generated tables

{chr(10).join(f'- `{item}`' for item in tables) if tables else '- Pending.'}

## 8. Generated figures

{chr(10).join(f'- `{item}`' for item in figures) if figures else '- Pending.'}

## 9. Result locations

Structured experiment outputs are under `results/`; feature caches under `cached_features/`; model artifacts inside each experiment's `runs/` directories; Hugging Face checkpoints under `checkpoints/huggingface/`; logs under `logs/`; figures under `figures/`; immutable configuration snapshots under `results/configs/`.

## 10. Methodological or implementation issues

The nominal plan described 50 dual-device subjects, but only 43 have both recordings and only 41 have reliably alignable regions. Subjects 34 and 35 are excluded from controlled device comparisons. The original clip-generation policy was unavailable, so the replacement policy is explicit: annotation-duration positive windows plus duration-matched event/wake-free negatives. Alignment is label-free but subject-specific; uncertain recording intervals are excluded. Results must therefore be interpreted for these two evaluated devices and the 41-subject synchronized subset. No result is fabricated when a required output is absent.
"""
    target = result_root / "experiment_summary.md"
    status_path = result_root / "experiment_status.json"
    if target.exists() or status_path.exists():
        raise FileExistsError("Refusing to overwrite final experiment summary/status")
    target.write_text(text, encoding="utf-8")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
