#!/usr/bin/env python3
"""One-off benchmark: gradient checkpointing (recompute-on-backward, current
production approach) vs. torch.autograd.graph.save_on_cpu (offload-to-CPU-RAM
approach) for PairPhysNet's forward+backward step. Both save GPU VRAM the
same way (frozen-trunk activations aren't kept on GPU); they differ only in
HOW the saved-for-backward activations come back: recomputed (checkpoint) vs.
transferred from pinned CPU RAM (save_on_cpu). Loads the model twice
sequentially (not simultaneously) to avoid doubling VRAM use during the test.
GPU-required, must run inside a SLURM job on a freshly-freed slot -- not
alongside the other 3 production jobs still training."""
from __future__ import annotations

import sys
import time
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

N_WARMUP = 3
N_TIMED = 15
BATCH_SIZE = 8
MAX_SECONDS = 20.0


def pad_batch(arrays, max_len):
    out = np.zeros((len(arrays), max_len), dtype=np.float32)
    for i, a in enumerate(arrays):
        n = min(len(a), max_len)
        out[i, :n] = a[:n]
    return torch.from_numpy(out)


def build_fixed_batch(ds, sample_rate):
    max_len = int(MAX_SECONDS * sample_rate)
    items = [ds[i] for i in range(BATCH_SIZE)]
    x_r = pad_batch([it["audio_R"] for it in items], max_len)
    x_s = pad_batch([it["audio_S"] for it in items], max_len)
    y_event = torch.tensor([it["label"] for it in items], dtype=torch.long)
    return x_r, x_s, y_event


def run_variant(name, use_checkpointing, use_save_on_cpu, x_r, x_s, y_event, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = PairPhysNetConfig(
        backbone_name="microsoft/wavlm-large", n_unfrozen_layers=2,
        pooling_mode="mean_std", projection_dim=256,
        lambda_pair=1.0, lambda_adv=0.5, lambda_dis=0.1,
    )
    model = PairPhysNet(cfg, local_files_only=True).to(device)
    if not use_checkpointing:
        # undo the unconditional gradient_checkpointing_enable() in
        # SharedEncoder.__init__ for this variant
        model.encoder.model.gradient_checkpointing_disable()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    y_device_r = torch.zeros(len(y_event), dtype=torch.long, device=device)
    y_device_s = torch.ones(len(y_event), dtype=torch.long, device=device)

    def step():
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=True):
            if use_save_on_cpu:
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    out = model(x_r, x_s, grl_lambda=1.0)
                    losses = model.compute_losses(out, y_event, y_device_r, y_device_s)
            else:
                out = model(x_r, x_s, grl_lambda=1.0)
                losses = model.compute_losses(out, y_event, y_device_r, y_device_s)
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        return losses["total"].item()

    try:
        for _ in range(N_WARMUP):
            step()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(N_TIMED):
            loss_val = step()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"{name}: {elapsed/N_TIMED:.3f} s/step  (last loss={loss_val:.4f}, "
              f"peak_vram={peak_vram:.2f} GB, {N_TIMED} steps timed after {N_WARMUP} warmup)")
        result = {"name": name, "sec_per_step": elapsed / N_TIMED, "peak_vram_gb": peak_vram, "status": "ok"}
    except torch.cuda.OutOfMemoryError as e:
        print(f"{name}: OOM -- {e}")
        result = {"name": name, "status": "OOM"}
    finally:
        del model, optimizer
        torch.cuda.empty_cache()

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires a GPU -- no CPU fallback by design.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_yaml(PROJECT / "configs" / "base.yaml")
    _, test_subjects = subjects_for_fold(Path(config["metadata"]["subject_folds"]), "0")
    ds = PairedEventDataset(test_subjects, config=config)
    ds.preload_all(max_workers=8)

    x_r, x_s, y_event = build_fixed_batch(ds, config["audio"]["target_sample_rate"])
    x_r, x_s, y_event = x_r.to(device), x_s.to(device), y_event.to(device)

    print(f"Benchmark: batch_size={BATCH_SIZE}, max_seconds={MAX_SECONDS}, "
          f"{N_WARMUP} warmup + {N_TIMED} timed steps per variant\n")

    results = []
    results.append(run_variant("A) gradient_checkpointing (current production)", True, False, x_r, x_s, y_event, device))
    # B) save_on_cpu removed -- confirmed incompatible with WavLM's attention
    # kernel (RuntimeError: attn_bias stride misalignment), not a viable path.
    results.append(run_variant("C) no checkpointing (does it still fit? how much faster?)", False, False, x_r, x_s, y_event, device))

    print("\n=== Summary ===")
    baseline = next((r for r in results if r["name"].startswith("A)") and r["status"] == "ok"), None)
    for r in results:
        if r["status"] != "ok":
            print(f"{r['name']}: {r['status']}")
            continue
        speedup = f" ({baseline['sec_per_step']/r['sec_per_step']:.2f}x vs A)" if baseline else ""
        print(f"{r['name']}: {r['sec_per_step']:.3f} s/step, {r['peak_vram_gb']:.2f} GB peak{speedup}")


if __name__ == "__main__":
    main()
