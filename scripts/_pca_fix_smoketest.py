#!/usr/bin/env python3
"""Smoke test: does fitting PCA on source+unlabeled-target-validation data
(mirroring CORAL's target-aware fit) fix the all-one-class collapse, vs the
current source-only PCA fit?"""
import sys
sys.path.insert(0, "/home/pkdas/IEEE_healthcomm_workshop/src")
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from sleep_quadnet.io import load_yaml, read_csv_rows
from sleep_quadnet.evaluation import (
    parse_folds, split_indices, load_representation, take_features, select_estimator,
    build_estimator, probability,
)
from sleep_quadnet.advanced import _calibration_safe_candidates

config = load_yaml(Path("/home/pkdas/IEEE_healthcomm_workshop/configs/base.yaml"))
config["project_root"] = "/home/pkdas/IEEE_healthcomm_workshop"
manifest_path = Path(config["metadata"]["manifest"])
fold_path = Path(config["metadata"]["subject_folds"])
cache_root = Path("/home/pkdas/IEEE_healthcomm_workshop/cached_features")

rows = read_csv_rows(manifest_path)
folds = parse_folds(fold_path)
protocol, fold, dimension, classifier = "R_S", 0, 384, "svm_rbf"

train_idx, val_idx, test_idx, subject_sets = split_indices(rows, folds, fold, protocol)
val_subjects = set(subject_sets["val_subjects"])
target_device = "S"  # R_S: source=R, target=S
target_val_idx = np.asarray(
    [i for i, row in enumerate(rows) if row["subject_id"] in val_subjects and row["device"] == target_device],
    dtype=np.int64,
)
print(f"train_idx={len(train_idx)} val_idx={len(val_idx)} target_val_idx={len(target_val_idx)} test_idx={len(test_idx)}")

arrays, native_dim = load_representation(cache_root, "full_fusion", config)
x_train = take_features(arrays, train_idx)
x_val = take_features(arrays, val_idx)
x_target_val = take_features(arrays, target_val_idx)
x_test = take_features(arrays, test_idx)
y_train = np.asarray([int(rows[i]["label"]) for i in train_idx], dtype=np.int8)
y_val = np.asarray([int(rows[i]["label"]) for i in val_idx], dtype=np.int8)
y_test = np.asarray([int(rows[i]["label"]) for i in test_idx], dtype=np.int8)

candidates = _calibration_safe_candidates(classifier, config["classifiers"][classifier])

def run(label, pca_fit_source):
    pca = PCA(n_components=dimension, svd_solver="randomized", iterated_power=2, random_state=config["seed"] + fold)
    pca.fit(pca_fit_source)
    z_train = pca.transform(x_train)[:, :dimension]
    z_val = pca.transform(x_val)[:, :dimension]
    z_test = pca.transform(x_test)[:, :dimension]
    selected, tuning = select_estimator(classifier, candidates, config["seed"] + fold * 100, z_train, y_train, z_val, y_val)
    estimator = build_estimator(classifier, selected, config["seed"] + fold * 1000)
    x_fit = np.concatenate([z_train, z_val])
    y_fit = np.concatenate([y_train, y_val])
    estimator.fit(x_fit, y_fit)
    probs = probability(estimator, z_test)
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    preds = (probs >= 0.5).astype(int)
    print(f"\n=== {label} ===")
    print(f"  prob range: [{probs.min():.4f}, {probs.max():.4f}]  mean={probs.mean():.4f}")
    print(f"  pred distribution: {np.bincount(preds)}")
    print(f"  balanced_accuracy={balanced_accuracy_score(y_test, preds):.4f}  roc_auc={roc_auc_score(y_test, probs):.4f}")

run("CURRENT (source-only PCA fit)", np.concatenate([x_train, x_val]))
run("TARGET-AWARE (source train+val + unlabeled target-val PCA fit)", np.concatenate([x_train, x_val, x_target_val]))
