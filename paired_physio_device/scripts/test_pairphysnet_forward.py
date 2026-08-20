#!/usr/bin/env python3
"""Stage 5 unit test: real forward+backward pass of PairPhysNet on a small
batch of real paired audio (not synthetic dummy tensors), verifying shapes,
finite losses, and that gradients actually flow into the unfrozen encoder
layers, both projection heads, and every loss-contributing component.
GPU-required (SharedEncoder loads a real HF model) -- must run in a SLURM job."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "paired_physio_device"))
sys.path.insert(0, str(PROJECT / "paired_physio_device" / "scripts"))

from models.pairphysnet import PairPhysNet, PairPhysNetConfig  # noqa: E402
from paired_dataset import PairedEventDataset, subjects_for_fold  # noqa: E402
from sleep_quadnet.io import load_yaml  # noqa: E402


def pad_batch(arrays: list[np.ndarray], max_len: int | None = None) -> torch.Tensor:
    max_len = max_len or max(len(a) for a in arrays)
    out = np.zeros((len(arrays), max_len), dtype=np.float32)
    for i, a in enumerate(arrays):
        n = min(len(a), max_len)
        out[i, :n] = a[:n]
    return torch.from_numpy(out)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("PairPhysNet unit test requires a GPU -- no CPU fallback by design.")
    device = torch.device("cuda")

    config = load_yaml(PROJECT / "configs" / "base.yaml")
    _, test_subjects = subjects_for_fold(Path(config["metadata"]["subject_folds"]), "0")
    ds = PairedEventDataset(test_subjects, config=config)

    batch_size = 4
    items = [ds[i] for i in range(batch_size)]
    max_len = min(16000 * 20, max(max(len(it["audio_R"]), len(it["audio_S"])) for it in items))
    x_r = pad_batch([it["audio_R"] for it in items], max_len).to(device)
    x_s = pad_batch([it["audio_S"] for it in items], max_len).to(device)
    y_event = torch.tensor([it["label"] for it in items], dtype=torch.long, device=device)
    y_device_r = torch.zeros(batch_size, dtype=torch.long, device=device)  # R = 0
    y_device_s = torch.ones(batch_size, dtype=torch.long, device=device)   # S = 1

    print(f"batch: x_r={x_r.shape} x_s={x_s.shape} labels={y_event.tolist()}")

    cfg = PairPhysNetConfig(
        backbone_name="microsoft/wavlm-large", n_unfrozen_layers=2,  # small for a fast unit test
        pooling_mode="mean_std", projection_dim=256,
        lambda_pair=1.0, lambda_adv=0.5, lambda_dis=0.1, lambda_spo2=0.0,
    )
    model = PairPhysNet(cfg, local_files_only=True).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_total:,} total, {n_trainable:,} trainable "
          f"({100*n_trainable/n_total:.2f}%)")

    out = model(x_r, x_s, grl_lambda=1.0)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all(), f"non-finite values in output {k}"
    print("forward pass OK, all outputs finite. shapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"  {k}: {{ {', '.join(f'{kk}: {tuple(vv.shape)}' for kk, vv in v.items())} }}")

    losses = model.compute_losses(out, y_event, y_device_r, y_device_s)
    print("losses:", {k: round(v.item(), 4) for k, v in losses.items()})
    for k, v in losses.items():
        assert torch.isfinite(v), f"non-finite loss: {k}"

    model.zero_grad()
    losses["total"].backward()

    # verify gradients actually flow into every intended component
    checks = {
        "encoder_unfrozen_layer": any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for n, p in model.encoder.named_parameters() if p.requires_grad
        ),
        "phys_projector": any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.phys_projector.parameters()),
        "device_projector": any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.device_projector.parameters()),
        "event_classifier": any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.event_classifier.parameters()),
        "device_classifier_on_d": any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.device_classifier_on_d.parameters()),
        "device_classifier_on_c": any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.device_classifier_on_c.parameters()),
    }
    print("gradient-flow checks:", checks)
    assert all(checks.values()), f"gradient did not reach some component: {checks}"

    print("\nPairPhysNet unit test PASSED: real forward+backward on real paired audio, "
          "finite losses, gradients reach every intended component.")


if __name__ == "__main__":
    main()
