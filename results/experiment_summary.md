# Sleep-QuadNet experiment summary

## 1. Experiments defined by the authoritative strategy

P0-A matched/cross-device benchmark; P0-B Data2Vec and leave-one-encoder-out fusion ablations; P0-C subject-level bootstrap robustness; P0-D computational cost; P1-A paired device acoustics; P1-B preprocessing ablation; P1-C fold-local PCA dimension control; P1-D validation-target CORAL; and the combined-device recovery protocol where useful. The optional learnable P2 fusion was intentionally not started before the reviewer-critical experiments.

## 2. Previous work available

The original project supplied a v4 runner and configuration, but no valid results, checkpoints, feature caches, logs, extracted labelled clips, or SLURM scripts were present. Its supplied cross-device mode was not patient-disjoint.

## 3. Newly implemented

Annotation-driven paired window materialization, label-free device-time alignment, fixed shared five-fold subject splits, independent leakage/window audits, resumable feature caches, validation-only classifier tuning, all plan-required experiment runners, append-only master logging, SLURM workflows, statistical/efficiency/acoustics analyses, and vector table/figure generation.

## 4. Completion status

```json
{
  "P0_device_gap": {
    "completed_fold_runs": 331,
    "expected_fold_runs": 680
  },
  "P0_ablation": {
    "completed_fold_runs": 0,
    "expected_fold_runs": 60
  },
  "P1_preprocessing": {
    "completed_fold_runs": 0,
    "expected_fold_runs": 80
  },
  "P1_dimension_control": {
    "completed_fold_runs": 0,
    "expected_fold_runs": 30
  },
  "P1_domain_adaptation": {
    "completed_fold_runs": 0,
    "expected_fold_runs": 30
  },
  "P0_statistics": {
    "complete": false
  },
  "P0_efficiency": {
    "complete": false
  },
  "P1_device_acoustics": {
    "complete": true
  }
}
```

## 5. Failed or pending

- P0_device_gap: 331/680 fold-runs
- P0_ablation: 0/60 fold-runs
- P1_preprocessing: 0/80 fold-runs
- P1_dimension_control: 0/30 fold-runs
- P1_domain_adaptation: 0/30 fold-runs
- P0_statistics: pending
- P0_efficiency: pending

## 6. Main numerical results

- No final benchmark metric is reported because the aggregate result does not yet exist.

Dataset/evaluation facts: 50 annotation subjects; 43 with both device files; 41 with reliable synchronized regions; 19,798 paired logical windows; 39,596 device rows; five fixed patient-disjoint folds.

## 7. Generated tables

- Pending.

## 8. Generated figures

- `figures/device_band_energy.pdf`
- `figures/mean_psd_recorder_vs_smartphone.pdf`

## 9. Result locations

Structured experiment outputs are under `results/`; feature caches under `cached_features/`; model artifacts inside each experiment's `runs/` directories; Hugging Face checkpoints under `checkpoints/huggingface/`; logs under `logs/`; figures under `figures/`; immutable configuration snapshots under `results/configs/`.

## 10. Methodological or implementation issues

The nominal plan described 50 dual-device subjects, but only 43 have both recordings and only 41 have reliably alignable regions. Subjects 34 and 35 are excluded from controlled device comparisons. The original clip-generation policy was unavailable, so the replacement policy is explicit: annotation-duration positive windows plus duration-matched event/wake-free negatives. Alignment is label-free but subject-specific; uncertain recording intervals are excluded. Results must therefore be interpreted for these two evaluated devices and the 41-subject synchronized subset. No result is fabricated when a required output is absent.
