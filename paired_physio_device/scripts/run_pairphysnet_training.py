#!/usr/bin/env python3
"""Stage 6 (Section E): train one PairPhysNet ablation variant (A1-A5) for
one held-out fold. GPU-required, must run inside a SLURM job.

Variant -> lambda mapping (Section E):
  A1  CE only                          lambda_pair=0   lambda_adv=0   lambda_dis=0
  A2  CE + paired contrastive          lambda_pair=1.0 lambda_adv=0   lambda_dis=0
  A3  CE + device adversarial          lambda_pair=0   lambda_adv=0.5 lambda_dis=0
  A4  CE + pair + adversarial          lambda_pair=1.0 lambda_adv=0.5 lambda_dis=0
  A5  full PairPhysNet                 lambda_pair=1.0 lambda_adv=0.5 lambda_dis=0.1 (+ SpO2 aux when available)

Lambda values above are the DEFAULT starting point, not yet validation-tuned
(Section E: 'Tune lambda values using validation subjects only. Do not tune
using test performance.') -- a small validation-only grid search is a
required follow-up before A2-A5 results are treated as final; this script
supports overriding lambdas via CLI for exactly that sweep, and always
records the actual values used in the run's config JSON (Section E:
'Record every chosen value in config files.').

A0 (frozen baseline) is NOT run by this script -- it already exists in
results/P0_device_gap (main benchmark), reused as-is per the master prompt's
Role instruction to preserve, not duplicate, trusted baselines."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "paired_physio_device"))
sys.path.insert(0, str(PROJECT / "paired_physio_device" / "scripts"))

from models.pairphysnet import PairPhysNet, PairPhysNetConfig  # noqa: E402
from paired_dataset import PairedEventDataset, subjects_for_fold  # noqa: E402
from sleep_quadnet.io import load_yaml, config_hash  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

VARIANT_LAMBDAS = {
    "A1": dict(lambda_pair=0.0, lambda_adv=0.0, lambda_dis=0.0),
    "A2": dict(lambda_pair=1.0, lambda_adv=0.0, lambda_dis=0.0),
    "A3": dict(lambda_pair=0.0, lambda_adv=0.5, lambda_dis=0.0),
    "A4": dict(lambda_pair=1.0, lambda_adv=0.5, lambda_dis=0.0),
    "A5": dict(lambda_pair=1.0, lambda_adv=0.5, lambda_dis=0.1),
}


def pad_batch(arrays, max_len):
    out = np.zeros((len(arrays), max_len), dtype=np.float32)
    for i, a in enumerate(arrays):
        n = min(len(a), max_len)
        out[i, :n] = a[:n]
    return torch.from_numpy(out)


def collate(batch, max_seconds: float, sample_rate: int):
    max_len = int(max_seconds * sample_rate)
    x_r = pad_batch([b["audio_R"] for b in batch], max_len)
    x_s = pad_batch([b["audio_S"] for b in batch], max_len)
    y_event = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    subjects = [b["subject_id"] for b in batch]
    return x_r, x_s, y_event, subjects


def evaluate(model, loader, device, grl_lambda: float, amp: bool = True):
    model.eval()
    all_pred, all_true, all_prob, all_subj = [], [], [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
        for x_r, x_s, y_event, subjects in loader:
            x_r = x_r.to(device, non_blocking=True)
            out = model(x_r, x_s=None, grl_lambda=grl_lambda)
            logits = out["event_logits_R"]
            prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            all_pred.extend(pred.tolist())
            all_true.extend(y_event.numpy().tolist())
            all_prob.extend(prob.tolist())
            all_subj.extend(subjects)
    ba = balanced_accuracy_score(all_true, all_pred)
    try:
        auroc = roc_auc_score(all_true, all_prob)
    except ValueError:
        auroc = None
    return ba, auroc, all_pred, all_true, all_subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_LAMBDAS.keys()))
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=2, help="effective batch = batch_size * grad_accum_steps")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_seconds", type=float, default=20.0)
    ap.add_argument("--n_unfrozen_layers", type=int, default=4)
    ap.add_argument("--pooling_mode", default="mean_std", choices=["mean", "mean_std", "attentive_stat"])
    ap.add_argument("--lambda_pair", type=float, default=None, help="override the variant default")
    ap.add_argument("--lambda_adv", type=float, default=None, help="override the variant default")
    ap.add_argument("--lambda_dis", type=float, default=None, help="override the variant default")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("run_pairphysnet_training.py requires a GPU -- no CPU fallback by design.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Pure speed knobs -- no change to what is computed or the training
    # semantics. cudnn.benchmark is safe here specifically because every
    # batch is padded to the same fixed length (args.max_seconds), so the
    # input shape cudnn autotunes for never changes mid-run. TF32 is already
    # implicitly consistent with AMP's own fp16/bf16 precision-for-speed
    # tradeoff (already in use via --amp), not a new kind of approximation.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_yaml(PROJECT / "configs" / "base.yaml")
    fold_path = Path(config["metadata"]["subject_folds"])
    train_subjects, test_subjects = subjects_for_fold(fold_path, args.fold)
    # validation subjects: the next fold in rotation, held out of training
    # (matches this project's existing fold-rotation convention, never the
    # test fold itself -- Section: 'No test labels or test-device statistics
    # may be used in model selection')
    import csv as _csv
    val_fold = str((int(args.fold) + 1) % 5)
    val_fold_subjects = {r["subject_id"] for r in _csv.DictReader(open(fold_path)) if r["fold"] == val_fold}
    val_subjects = train_subjects & val_fold_subjects  # earmarked for validation, disjoint from test by construction
    train_subjects = train_subjects - val_subjects

    print(f"variant={args.variant} fold={args.fold} "
          f"n_train_subj={len(train_subjects)} n_val_subj={len(val_subjects)} n_test_subj={len(test_subjects)}")

    train_ds = PairedEventDataset(train_subjects, config=config)
    val_ds = PairedEventDataset(val_subjects, config=config)
    test_ds = PairedEventDataset(test_subjects, config=config)
    print(f"n_train_pairs={len(train_ds)} n_val_pairs={len(val_ds)} n_test_pairs={len(test_ds)}")

    # RAM-exploiting speedup: decode+resample every item ONCE now, in the
    # main process, before the DataLoader's persistent workers fork (see
    # PairedEventDataset.preload_all's docstring for why this ordering
    # matters -- fork-based workers inherit the populated cache via
    # copy-on-write, for free). Without this, the same CPU-bound librosa
    # resampling work happens redundantly on every one of the 15 epochs.
    print("preloading audio into memory (one-time cost, amortized over 15 epochs)...", flush=True)
    train_ds.preload_all(max_workers=8)
    val_ds.preload_all(max_workers=8)
    test_ds.preload_all(max_workers=8)

    def _collate(batch):
        return collate(batch, args.max_seconds, config["audio"]["target_sample_rate"])

    # num_workers matched to --cpus-per-task (4, see the sbatch script) --
    # was hardcoded to 2, leaving half the allocated CPUs idle during the
    # CPU-bound audio load/resample step (src/sleep_quadnet/io.py's librosa
    # resampling, unchanged). pin_memory + non_blocking .to(device) overlaps
    # host->GPU copy with compute. persistent_workers avoids respawning the
    # worker pool at the start of every one of the 15 epochs. None of this
    # changes which samples are loaded or what is computed on them.
    loader_kwargs = dict(num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate, **loader_kwargs)

    lambdas = dict(VARIANT_LAMBDAS[args.variant])
    if args.lambda_pair is not None:
        lambdas["lambda_pair"] = args.lambda_pair
    if args.lambda_adv is not None:
        lambdas["lambda_adv"] = args.lambda_adv
    if args.lambda_dis is not None:
        lambdas["lambda_dis"] = args.lambda_dis

    model_cfg = PairPhysNetConfig(
        backbone_name="microsoft/wavlm-large", n_unfrozen_layers=args.n_unfrozen_layers,
        pooling_mode=args.pooling_mode, projection_dim=256, **lambdas,
    )
    model = PairPhysNet(model_cfg, local_files_only=True).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    run_config = {
        "variant": args.variant, "fold": args.fold, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "max_seconds": args.max_seconds,
        "n_unfrozen_layers": args.n_unfrozen_layers, "pooling_mode": args.pooling_mode,
        **lambdas, "backbone_name": model_cfg.backbone_name,
    }
    run_id = f"{args.variant}_fold{args.fold}_seed{args.seed}_{config_hash(run_config)[:10]}"
    out_dir = PROJECT / "paired_physio_device" / "checkpoints" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_val_ba = -1.0
    best_state = None
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()
        # file=sys.stderr (tqdm's default) and a "train e<N>" label, both
        # deliberately -- watch.sh greps the .out log for lines matching
        # '^epoch [0-9]+:' to detect a completed epoch's summary line
        # (printed once, below). tqdm's own per-step bar text would collide
        # with that same pattern (and appear far more often) if it shared
        # stdout or started with "epoch N:", corrupting the dashboard.
        pbar = tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"train e{epoch}",
            file=sys.stderr, mininterval=5.0, dynamic_ncols=True,
        )
        for step, (x_r, x_s, y_event, _subjects) in pbar:
            x_r = x_r.to(device, non_blocking=True)
            x_s = x_s.to(device, non_blocking=True)
            y_event = y_event.to(device, non_blocking=True)
            y_device_r = torch.zeros(len(y_event), dtype=torch.long, device=device)
            y_device_s = torch.ones(len(y_event), dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                out = model(x_r, x_s, grl_lambda=1.0)
                losses = model.compute_losses(out, y_event, y_device_r, y_device_s)
                loss = losses["total"] / args.grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_losses.append(losses["total"].item())
            if step % 10 == 0:
                pbar.set_postfix(loss=f"{np.mean(epoch_losses[-50:]):.4f}")

        val_ba, val_auroc, *_ = evaluate(model, val_loader, device, grl_lambda=1.0, amp=args.amp)
        print(f"epoch {epoch}: train_loss={np.mean(epoch_losses):.4f} val_BA={val_ba:.4f} val_AUROC={val_auroc}")
        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best_checkpoint.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_ba, test_auroc, test_pred, test_true, test_subj = evaluate(model, test_loader, device, grl_lambda=1.0, amp=args.amp)
    runtime_sec = time.time() - start_time

    result = {
        "experiment_id": run_id, "variant": args.variant, "fold": args.fold, "seed": args.seed,
        "task": "event_classification", "train_device": "R+S_paired", "test_device": "R_only_at_test",
        "model_variant": f"PairPhysNet_{args.variant}", "backbone": model_cfg.backbone_name,
        "pooling": args.pooling_mode, "window_sec": args.max_seconds,
        **{k: v for k, v in lambdas.items()},
        "n_train_subjects": len(train_subjects), "n_val_subjects": len(val_subjects), "n_test_subjects": len(test_subjects),
        "n_train_samples": len(train_ds), "n_test_samples": len(test_ds),
        "best_val_BA": best_val_ba, "test_BA": test_ba, "test_AUROC": test_auroc,
        "runtime_sec": runtime_sec, "config_hash": config_hash(run_config), "status": "complete",
    }
    results_dir = PROJECT / "paired_physio_device" / "results" / "event" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "completion.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
