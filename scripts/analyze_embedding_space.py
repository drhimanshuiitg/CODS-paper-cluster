#!/usr/bin/env python3
"""Q1_Paper_Artifact analysis: PCA projection of already-cached HuBERT
embeddings, colored by device and by class -- lightweight (sklearn PCA on
an existing feature cache), no retraining, no GPU.

Outputs:
  Q1_Paper_Artifact/tables/embedding_pca_coords.csv  (2D PCA coords + device + label, sampled)
  Q1_Paper_Artifact/analysis/embedding_pca_summary.json (explained variance, silhouette-style separation stats)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_quadnet.io import read_csv_rows

FEATURE = "hubert"


def main() -> None:
    cache_dir = PROJECT_ROOT / "cached_features" / FEATURE / "peak"
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    sample_ids = json.loads((cache_dir / "sample_ids.json").read_text())
    manifest_rows = read_csv_rows(PROJECT_ROOT / "metadata" / "dataset_manifest_aligned.csv")
    row_by_id = {r["sample_id"]: r for r in manifest_rows}

    device = np.array([row_by_id[sid]["device"] for sid in sample_ids])
    label = np.array([int(row_by_id[sid]["label"]) for sid in sample_ids])

    # Stratified sample by (device, label) for a legible, balanced scatter plot
    rng = np.random.RandomState(42)
    per_cell = 600
    idx_selected = []
    for d in ("R", "S"):
        for l in (0, 1):
            cell_idx = np.flatnonzero((device == d) & (label == l))
            n = min(per_cell, len(cell_idx))
            idx_selected.extend(rng.choice(cell_idx, size=n, replace=False))
    idx_selected = np.array(sorted(idx_selected))
    print(f"sampled {len(idx_selected)} rows (stratified device x label) of {len(sample_ids)} total")

    x = np.asarray(features[idx_selected], dtype=np.float32)
    pca = PCA(n_components=10, random_state=42)
    z = pca.fit_transform(x)

    df = pd.DataFrame({
        "pc1": z[:, 0], "pc2": z[:, 1],
        "device": device[idx_selected], "label": label[idx_selected],
    })
    out_tables = PROJECT_ROOT / "Q1_Paper_Artifact" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tables / "embedding_pca_coords.csv", index=False)

    # Separation quality: silhouette score in the full 10-D PCA space (not just 2D, for a
    # less cherry-picked estimate), once grouped by device and once by class.
    device_codes = (device[idx_selected] == "S").astype(int)
    sil_device = float(silhouette_score(z, device_codes))
    sil_label = float(silhouette_score(z, label[idx_selected]))

    summary = {
        "feature": FEATURE,
        "n_sampled": int(len(idx_selected)),
        "pca_n_components": 10,
        "explained_variance_ratio_pc1_pc2": [float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])],
        "explained_variance_ratio_cumulative_10pc": float(pca.explained_variance_ratio_.sum()),
        "silhouette_by_device_10pc": sil_device,
        "silhouette_by_class_10pc": sil_label,
        "note": ("Silhouette score in [-1,1]; higher means tighter, better-separated clusters under that "
                 "grouping. Computed in the 10-component PCA space (not the 2D plot alone) for a less "
                 "cherry-picked estimate. A higher device-silhouette than class-silhouette would indicate "
                 "the frozen HuBERT embedding space separates recording device more cleanly than it "
                 "separates the apnea-event label -- i.e. device identity is an easier, more dominant "
                 "signal in this representation than the clinical label itself."),
    }
    out_analysis = PROJECT_ROOT / "Q1_Paper_Artifact" / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)
    (out_analysis / "embedding_pca_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
