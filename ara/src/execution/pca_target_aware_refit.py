"""Core algorithm stub for C05 / H01: the target-aware PCA refit fix.

Reconstructed from src/sleep_quadnet/advanced.py::run_pca_fold (the real
implementation additionally handles caching, hyperparameter tuning, result
persistence, and multi-dimension sweeps -- this stub isolates only the novel
contribution: what data the refit-stage PCA is fit on, and why."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SplitFeatures:
    """Feature matrices for one fold's train/val/test split, already
    restricted to the correct device set by the caller's split_indices()."""
    x_train_source: NDArray[np.float32]       # shape (n_train, d) -- source-device train subjects
    x_val_source: NDArray[np.float32]         # shape (n_val, d)   -- source-device val subjects
    x_val_target: NDArray[np.float32] | None  # shape (n_val_tgt, d) or None for matched-device protocols
    x_test: NDArray[np.float32]               # shape (n_test, d) -- target-device (or source, if matched) test subjects


def fit_target_aware_pca(features: SplitFeatures, n_components: int, random_state: int) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """The fix. Pre-fix, `fit_data` was `x_train_source` concat `x_val_source`
    only -- the PCA's principal axes never saw the target device's feature
    distribution, which caused a degenerate single-class collapse in
    cross-device transfer (C04). Post-fix, when x_val_target is available
    (cross-device protocols only), it is included in the fit -- mirroring
    how CORAL's covariance alignment was already scoped in the same codebase
    (see logic/concepts.md: Target-aware refit).

    Returns (z_fit, z_test, components) where z_fit/z_test are the fitted/
    transformed feature matrices ready for classifier fitting, and
    `components` is the fitted PCA basis (kept for provenance/inspection,
    not used downstream in this stub).
    """
    from sklearn.decomposition import PCA

    fit_parts = [features.x_train_source, features.x_val_source]
    if features.x_val_target is not None and len(features.x_val_target) > 0:
        fit_parts.append(features.x_val_target)  # <-- the entire fix is this one line's presence
    x_fit = np.concatenate(fit_parts, axis=0)

    max_dimension = min(n_components, x_fit.shape[0] - 1, x_fit.shape[1] - 1)
    if max_dimension != n_components:
        raise ValueError(
            f"n_components={n_components} invalid for fit data shape {x_fit.shape} "
            "(PCA target dimension must be strictly less than min(n_samples, n_features); "
            "see logic/solution/heuristics.md H04 for a real crash this guard caught)."
        )

    pca = PCA(n_components=n_components, svd_solver="randomized", iterated_power=2, random_state=random_state)
    z_fit = pca.fit_transform(x_fit).astype(np.float32)
    z_test = pca.transform(features.x_test).astype(np.float32)
    return z_fit, z_test, pca.components_.astype(np.float32)


def target_devices_for_protocol(train_devices: set[str], test_devices: set[str]) -> set[str]:
    """Empty for matched-device protocols (R_R, S_S) -- the fix is then a
    structural no-op, since there is no target-device data to add. Non-empty
    only for cross-device protocols (R_S, S_R). This is the exact boundary
    condition documented in logic/solution/constraints.md."""
    return test_devices - train_devices
