#!/usr/bin/env python3
"""Recovery path for jobs killed by SLURM TIMEOUT before writing a
completion.json: loads an already-saved best_checkpoint.pt (saved every
time val_BA improved during the original run -- see run_pairphysnet_training.py)
and runs ONLY the final test-set evaluation, writing the same completion.json
schema the original run would have produced. Does not retrain. GPU-required."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "paired_physio_device"))
sys.path.insert(0, str(PROJECT / "paired_physio_device" / "scripts"))

from models.pairphysnet import PairPhysNet, PairPhysNetConfig  # noqa: E402
from paired_dataset import PairedEventDataset, subjects_for_fold  # noqa: E402
from run_pairphysnet_training import evaluate, collate, VARIANT_LAMBDAS  # noqa: E402
from sleep_quadnet.io import load_yaml, config_hash  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="e.g. A1_fold0_seed42_e6699f24d1")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("recover_from_checkpoint.py requires a GPU -- no CPU fallback by design.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = PROJECT / "paired_physio_device" / "checkpoints" / args.checkpoint_dir
    run_config = json.load(open(ckpt_dir / "config.json"))
    print(f"recovering {args.checkpoint_dir}: {run_config}")

    config = load_yaml(PROJECT / "configs" / "base.yaml")
    fold_path = Path(config["metadata"]["subject_folds"])
    _, test_subjects = subjects_for_fold(fold_path, run_config["fold"])
    test_ds = PairedEventDataset(test_subjects, config=config)
    print(f"n_test_pairs={len(test_ds)}")
    test_ds.preload_all(max_workers=8)

    def _collate(batch):
        return collate(batch, run_config["max_seconds"], config["audio"]["target_sample_rate"])
    test_loader = DataLoader(test_ds, batch_size=run_config["batch_size"], shuffle=False,
                              collate_fn=_collate, num_workers=6, pin_memory=True)

    lambdas = {k: run_config[k] for k in ("lambda_pair", "lambda_adv", "lambda_dis")}
    model_cfg = PairPhysNetConfig(
        backbone_name=run_config["backbone_name"], n_unfrozen_layers=run_config["n_unfrozen_layers"],
        pooling_mode=run_config["pooling_mode"], projection_dim=256, **lambdas,
    )
    model = PairPhysNet(model_cfg, local_files_only=True).to(device)
    state = torch.load(ckpt_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)

    test_ba, test_auroc, test_pred, test_true, test_subj = evaluate(model, test_loader, device, grl_lambda=1.0, amp=True)

    run_id = ckpt_dir.name
    result = {
        "experiment_id": run_id, "variant": run_config["variant"], "fold": run_config["fold"],
        "seed": run_config["seed"], "task": "event_classification", "train_device": "R+S_paired",
        "test_device": "R_only_at_test", "model_variant": f"PairPhysNet_{run_config['variant']}",
        "backbone": run_config["backbone_name"], "pooling": run_config["pooling_mode"],
        "window_sec": run_config["max_seconds"], **lambdas,
        "n_test_subjects": len(test_subjects), "n_test_samples": len(test_ds),
        "test_BA": test_ba, "test_AUROC": test_auroc,
        "config_hash": config_hash(run_config), "status": "complete",
        "recovered_from_timeout": True,
        "recovery_note": "Original job hit SLURM TIME LIMIT before writing completion.json; "
                          "this result evaluates the already-saved best-validation checkpoint "
                          "(saved during the original run whenever val_BA improved), not a retrain.",
    }
    results_dir = PROJECT / "paired_physio_device" / "results" / "event" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "completion.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
