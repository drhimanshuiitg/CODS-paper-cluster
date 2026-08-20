#!/usr/bin/env python3
"""Stage 4 / Section H: subject-disjoint GPU device probe (Recorder vs
Smartphone) on a frozen representation's cached embeddings. Reuses the
existing, trusted, GPU-hard-fail-checked TorchMLPClassifier
(src/sleep_quadnet/evaluation.py) -- not a new classifier implementation.

Must run inside a SLURM job with --gres=gpu:mig24gb:1 (this project's
GPU-only compute policy; no CPU fallback)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
sys.path.insert(0, str(PROJECT / "src"))

from sleep_quadnet.evaluation import TorchMLPClassifier  # noqa: E402
from sleep_quadnet.io import load_yaml, read_csv_rows  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

REPRESENTATIONS = [
    "hubert", "wavlm", "wavlm_large", "wav2vec2",
    "data2vec_audio", "data2vec_spectrogram", "hear",
]


def load_embeddings(representation: str, preprocessing: str = "peak"):
    cache_dir = Path(f"/scratch/pkdas/IEEE_healthcomm_workshop/cached_features/{representation}/{preprocessing}")
    feats = np.load(cache_dir / "features.npy")
    ids = json.load(open(cache_dir / "sample_ids.json"))
    return feats, ids


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = [rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("run_device_probe.py requires a GPU -- no CPU fallback by design (GPU-only policy).")

    config = load_yaml(PROJECT / "configs" / "base.yaml")
    manifest = read_csv_rows(Path(config["metadata"]["manifest"]))
    folds = read_csv_rows(Path(config["metadata"]["subject_folds"]))
    subj_fold = {r["subject_id"]: r["fold"] for r in folds}

    feats, ids = load_embeddings(args.representation)
    id_to_row = {r["sample_id"]: r for r in manifest}
    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    train_mask, test_mask = [], []
    for sid, idx in id_to_idx.items():
        row = id_to_row.get(sid)
        if row is None:
            continue
        f = subj_fold.get(row["subject_id"])
        if f == args.fold:
            test_mask.append(idx)
        else:
            train_mask.append(idx)

    train_idx = np.array(train_mask)
    test_idx = np.array(test_mask)

    X_train, y_train = feats[train_idx], np.array([0 if id_to_row[ids[i]]["device"] == "R" else 1 for i in train_idx])
    X_test, y_test = feats[test_idx], np.array([0 if id_to_row[ids[i]]["device"] == "R" else 1 for i in test_idx])
    subj_test = np.array([id_to_row[ids[i]]["subject_id"] for i in test_idx])

    mlp_cfg = config["classifiers"]["mlp"][0]
    clf = TorchMLPClassifier(
        hidden_layer_sizes=mlp_cfg["hidden_layer_sizes"], alpha=mlp_cfg["alpha"],
        max_iter=mlp_cfg["max_iter"], early_stopping=mlp_cfg["early_stopping"],
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    ba = balanced_accuracy_score(y_test, pred)
    try:
        auroc = roc_auc_score(y_test, proba)
    except ValueError:
        auroc = None

    # subject-level bootstrap CI: per-subject accuracy, then bootstrap over subjects
    per_subj_acc = {}
    for s in np.unique(subj_test):
        m = subj_test == s
        per_subj_acc[s] = float((pred[m] == y_test[m]).mean())
    subj_vals = np.array(list(per_subj_acc.values()))
    ci_lo, ci_hi = bootstrap_ci(subj_vals) if len(subj_vals) >= 3 else (None, None)

    result = {
        "experiment_id": f"device_probe_{args.representation}_fold{args.fold}",
        "representation": args.representation,
        "fold": args.fold,
        "seed": args.seed,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_subjects": len(per_subj_acc),
        "balanced_accuracy": float(ba),
        "auroc": float(auroc) if auroc is not None else None,
        "subject_level_accuracy_ci95_low": ci_lo,
        "subject_level_accuracy_ci95_high": ci_hi,
        "per_subject_accuracy": per_subj_acc,
        "status": "complete",
    }

    out_dir = PROJECT / "paired_physio_device" / "results" / "device_probe" / f"{args.representation}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "completion.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_subject_accuracy"}, indent=2))


if __name__ == "__main__":
    main()
