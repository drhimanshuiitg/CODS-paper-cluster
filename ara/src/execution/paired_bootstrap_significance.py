"""Core algorithm stub for the paired subject-level bootstrap significance
test used to back every claim in logic/claims.md that asserts "significant".

Reconstructed from the 4 near-identical real implementations in this project
(scripts/run_statistics.py, run_pca_fix_significance.py,
run_corroboration_significance.py, run_ablation_significance.py) -- this
stub isolates the shared statistical method; the real scripts additionally
handle result-file loading, dedup-key aggregation, and CSV persistence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class BootstrapResult:
    point_difference: float       # arm_a's point estimate minus arm_b's, no resampling
    bootstrap_mean: float
    bootstrap_std: float
    ci95_low: float
    ci95_high: float
    p_value_two_sided: float

    @property
    def significant(self) -> bool:
        """A comparison counts as significant iff the 95% CI excludes zero
        entirely -- used throughout this project's claims.md as the sole
        significance criterion (no separate p<0.05 threshold is applied on
        top of the CI check; the two are consistent by construction for a
        percentile bootstrap)."""
        return self.ci95_low > 0 or self.ci95_high < 0


def per_subject_scores(labels: NDArray[np.int8], probabilities: NDArray[np.float64], subject_ids: NDArray) -> pd.Series:
    """One score per subject (e.g. balanced_accuracy), NOT one score per
    window -- resampling over subjects (not windows) is what correctly
    accounts for within-subject correlation of window-level errors. See
    logic/concepts.md: Paired subject-level bootstrap significance."""
    from sklearn.metrics import balanced_accuracy_score

    frame = pd.DataFrame({"subject_id": subject_ids, "label": labels, "probability": probabilities})
    scores = {}
    for subject, group in frame.groupby("subject_id"):
        predictions = (group["probability"].to_numpy() >= 0.5).astype(int)
        scores[subject] = balanced_accuracy_score(group["label"].to_numpy(), predictions)
    return pd.Series(scores)


def paired_bootstrap(scores_a: pd.Series, scores_b: pd.Series, iterations: int, seed: int) -> BootstrapResult:
    """`scores_a`/`scores_b` MUST already be aligned on the same subject
    index (same test subjects, same fold) -- every real call site in this
    project enforces this via an explicit inner-join-then-length-check
    (align_pair()) before calling the equivalent of this function, raising
    rather than silently comparing mismatched subject sets."""
    common = scores_a.index.intersection(scores_b.index)
    if len(common) == 0:
        raise ValueError("No common subjects between the two arms -- cannot compute a paired comparison")
    diff = (scores_a.loc[common] - scores_b.loc[common]).to_numpy()

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(iterations, len(diff)))
    bootstrap_means = diff[draws].mean(axis=1)

    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    p_value = min(1.0, 2.0 * min(float(np.mean(bootstrap_means <= 0)), float(np.mean(bootstrap_means >= 0))))

    return BootstrapResult(
        point_difference=float(scores_a.loc[common].mean() - scores_b.loc[common].mean()),
        bootstrap_mean=float(bootstrap_means.mean()),
        bootstrap_std=float(bootstrap_means.std(ddof=1)),
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        p_value_two_sided=p_value,
    )
