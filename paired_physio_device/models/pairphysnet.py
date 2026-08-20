#!/usr/bin/env python3
"""PairPhysNet (Section C). A shared-encoder, paired physiology-guided,
device-disentangled model for respiratory-event classification.

Temporary research name only (per the master prompt: 'Do not use the final
name in the paper until the complete results support the method'). Sleep-
QuadNet / full_fusion is NOT touched by this file -- it remains the
untouched A0 baseline, imported nowhere here.

Design choices made explicit here rather than left implicit:
  - Backbone: microsoft/wavlm-large by default (validation-selected best
    single encoder from the prior benchmark, ara/logic/claims.md F05).
    Configurable via `backbone_name`.
  - Fine-tuning depth: only the last `n_unfrozen_layers` transformer layers
    (default 4) are unfrozen; everything below stays frozen. Full
    fine-tuning of a 300M+ param model on ~40k windows with a single 24GB
    MIG slice is VRAM/overfitting-risky and not attempted by default -- this
    is a stated, deliberate choice, not a silent limitation.
  - Chunking: windows longer than `max_chunk_seconds` (20.0s, matching
    configs/base.yaml's existing ssl_max_chunk_seconds convention) are split
    into non-overlapping chunks, encoded independently, and their frame-level
    hidden states concatenated along time before pooling -- mirrors the
    existing trusted feature-extraction convention in
    src/sleep_quadnet/features.py rather than inventing a new one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Gradient reversal (Section C4)
# ---------------------------------------------------------------------------

class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambd)


# ---------------------------------------------------------------------------
# Shared encoder (Section C1)
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """Wraps a frozen/partially-fine-tuned HF SSL speech model. Returns
    frame-level hidden states (B, T, D), not pre-pooled -- pooling is a
    separate, configurable module (Section C1 point 'Implement a
    configurable pooling module')."""

    def __init__(self, backbone_name: str = "microsoft/wavlm-large",
                 n_unfrozen_layers: int = 4, max_chunk_seconds: float = 20.0,
                 sample_rate: int = 16000, local_files_only: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(backbone_name, local_files_only=local_files_only)
        self.hidden_size = self.model.config.hidden_size
        self.max_chunk_samples = int(max_chunk_seconds * sample_rate)
        self.backbone_name = backbone_name

        # Memory fix (2026-08-20, found via a real CUDA OOM on the first
        # training smoke test at batch_size=8 on a 24GB MIG slice -- a
        # 315M-param transformer's forward activations through all 24 layers
        # dominate memory even with most layers frozen, since autograd must
        # retain them to backprop into the unfrozen top layers). Gradient
        # checkpointing trades recompute for memory and is the first-line
        # fix per the Four-GPU skill's OOM failure policy ("activate
        # gradient checkpointing") before resorting to a smaller backbone.
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                # use_reentrant=False avoids the classic "checkpointed segment
                # has no input with requires_grad=True" failure mode that
                # bites exactly this frozen-trunk/unfrozen-top setup under
                # the legacy reentrant checkpoint implementation.
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

        for p in self.model.parameters():
            p.requires_grad = False
        encoder_layers = self._find_transformer_layers()
        if n_unfrozen_layers > 0 and encoder_layers is not None:
            for layer in encoder_layers[-n_unfrozen_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.n_unfrozen_layers = n_unfrozen_layers

    def _find_transformer_layers(self):
        # WavLM/Wav2Vec2/HuBERT (transformers) all expose .encoder.layers
        enc = getattr(self.model, "encoder", None)
        if enc is not None and hasattr(enc, "layers"):
            return enc.layers
        return None

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, n_samples) raw waveform, already resampled to the target
        rate. Chunks internally if longer than max_chunk_samples; returns
        concatenated frame-level hidden states (B, T_total, D)."""
        b, n = x.shape
        if n <= self.max_chunk_samples:
            out = self.model(x, attention_mask=attention_mask)
            return out.last_hidden_state

        chunks = []
        for start in range(0, n, self.max_chunk_samples):
            chunk = x[:, start:start + self.max_chunk_samples]
            if chunk.shape[1] < 400:  # shorter than one conv-front-end receptive field; skip
                continue
            out = self.model(chunk)
            chunks.append(out.last_hidden_state)
        return torch.cat(chunks, dim=1)


# ---------------------------------------------------------------------------
# Pooling (Section C1: mean / mean+std / attentive statistical)
# ---------------------------------------------------------------------------

class AttentiveStatPooling(nn.Module):
    def __init__(self, in_dim: int, attn_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D)
        w = torch.softmax(self.attn(h), dim=1)  # (B, T, 1)
        mean = torch.sum(w * h, dim=1)
        var = torch.sum(w * (h - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var.clamp_min(1e-8))
        return torch.cat([mean, std], dim=-1)


class Pooling(nn.Module):
    def __init__(self, in_dim: int, mode: str = "mean"):
        super().__init__()
        assert mode in {"mean", "mean_std", "attentive_stat"}
        self.mode = mode
        if mode == "attentive_stat":
            self.attentive = AttentiveStatPooling(in_dim)
        self.out_dim = in_dim if mode == "mean" else in_dim * 2

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return h.mean(dim=1)
        if self.mode == "mean_std":
            mean = h.mean(dim=1)
            std = h.std(dim=1)
            return torch.cat([mean, std], dim=-1)
        return self.attentive(h)


# ---------------------------------------------------------------------------
# Projection heads (Sections C2/C3)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int | None = None, l2_normalize: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or max(out_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z


class SmallClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpO2AuxHeads(nn.Module):
    """Section D2 auxiliary heads, targets empirically grounded in
    results/physiology/spo2_event_timing_audit.json (median delay-to-nadir
    41-53s by subtype; 150s search window covers >99% of events)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.desat_prob_head = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.amplitude_head = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.delay_to_nadir_head = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        # event-associated desaturation area: NOT YET COMPUTED as a target
        # (needs trapezoidal-integration ground truth, not yet built --
        # tracked as an open item; head is present but unused until the
        # ground-truth column exists, to avoid training against a fabricated target).

    def forward(self, c: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "desat_prob_logit": self.desat_prob_head(c).squeeze(-1),
            "amplitude_pred": self.amplitude_head(c).squeeze(-1),
            "delay_to_nadir_pred": self.delay_to_nadir_head(c).squeeze(-1),
        }


# ---------------------------------------------------------------------------
# Losses (Sections C5, C6, C4)
# ---------------------------------------------------------------------------

def disentanglement_loss(c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Normalized cross-covariance penalty between physiology-content and
    device-style representations (Section C5: 'Do not use a numerically
    unstable raw dot-product penalty without normalization')."""
    c = (c - c.mean(dim=0, keepdim=True)) / (c.std(dim=0, keepdim=True) + 1e-6)
    d = (d - d.mean(dim=0, keepdim=True)) / (d.std(dim=0, keepdim=True) + 1e-6)
    b = c.shape[0]
    cross_cov = (c.T @ d) / max(b - 1, 1)  # (dim_c, dim_d)
    return (cross_cov ** 2).mean()


def paired_nt_xent_loss(c_r: torch.Tensor, c_s: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Standard NT-Xent/InfoNCE on same-event (R, S) pairs (Section C6).
    c_r, c_s already L2-normalized (ProjectionHead(l2_normalize=True)).
    Positives: c_r[i] <-> c_s[i] (same paired_event_id, by dataloader
    construction). Negatives: all other events in the batch, both directions."""
    b = c_r.shape[0]
    z = torch.cat([c_r, c_s], dim=0)  # (2B, D)
    sim = (z @ z.T) / temperature  # (2B, 2B)
    mask_self = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, float("-inf"))
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@dataclass
class PairPhysNetConfig:
    backbone_name: str = "microsoft/wavlm-large"
    n_unfrozen_layers: int = 4
    pooling_mode: str = "mean_std"
    projection_dim: int = 256
    event_n_classes: int = 2  # {event, non-event}; subtype variant uses 3 {normal-proxy, hypo, osa}
    contrastive_temperature: float = 0.1
    lambda_pair: float = 0.0
    lambda_adv: float = 0.0
    lambda_dis: float = 0.0
    lambda_spo2: float = 0.0
    max_chunk_seconds: float = 20.0
    sample_rate: int = 16000


class PairPhysNet(nn.Module):
    def __init__(self, cfg: PairPhysNetConfig, local_files_only: bool = False):
        super().__init__()
        self.cfg = cfg
        self.encoder = SharedEncoder(
            backbone_name=cfg.backbone_name, n_unfrozen_layers=cfg.n_unfrozen_layers,
            max_chunk_seconds=cfg.max_chunk_seconds, sample_rate=cfg.sample_rate,
            local_files_only=local_files_only,
        )
        self.pooling = Pooling(self.encoder.hidden_size, mode=cfg.pooling_mode)
        pooled_dim = self.pooling.out_dim

        self.phys_projector = ProjectionHead(pooled_dim, cfg.projection_dim, l2_normalize=True)
        self.device_projector = ProjectionHead(pooled_dim, cfg.projection_dim, l2_normalize=True)

        self.event_classifier = SmallClassifier(cfg.projection_dim, cfg.event_n_classes)
        self.device_classifier_on_d = SmallClassifier(cfg.projection_dim, 2)      # verifies d retains device info
        self.device_classifier_on_c = SmallClassifier(cfg.projection_dim, 2)      # adversarial target (via GRL)
        self.spo2_heads = SpO2AuxHeads(cfg.projection_dim)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)          # (B, T, D_enc)
        pooled = self.pooling(h)     # (B, D_pooled)
        c = self.phys_projector(pooled)
        d = self.device_projector(pooled)
        return {"pooled": pooled, "c": c, "d": d}

    def forward(self, x_r: torch.Tensor, x_s: torch.Tensor | None = None, grl_lambda: float = 1.0) -> dict:
        out_r = self.encode(x_r)
        result = {
            "event_logits_R": self.event_classifier(out_r["c"]),
            "device_logits_on_d_R": self.device_classifier_on_d(out_r["d"]),
            "device_logits_on_c_R": self.device_classifier_on_c(grad_reverse(out_r["c"], grl_lambda)),
            "spo2_R": self.spo2_heads(out_r["c"]),
            "c_R": out_r["c"], "d_R": out_r["d"],
        }
        if x_s is not None:
            out_s = self.encode(x_s)
            result.update({
                "event_logits_S": self.event_classifier(out_s["c"]),
                "device_logits_on_d_S": self.device_classifier_on_d(out_s["d"]),
                "device_logits_on_c_S": self.device_classifier_on_c(grad_reverse(out_s["c"], grl_lambda)),
                "spo2_S": self.spo2_heads(out_s["c"]),
                "c_S": out_s["c"], "d_S": out_s["d"],
            })
        return result

    def compute_losses(self, batch_out: dict, y_event: torch.Tensor,
                        y_device_r: torch.Tensor, y_device_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        losses = {}
        losses["event"] = F.cross_entropy(batch_out["event_logits_R"], y_event)
        losses["device_on_d"] = F.cross_entropy(batch_out["device_logits_on_d_R"], y_device_r)

        if self.cfg.lambda_adv > 0:
            losses["adv_device_on_c"] = F.cross_entropy(batch_out["device_logits_on_c_R"], y_device_r)
        if "c_S" in batch_out:
            if self.cfg.lambda_adv > 0:
                losses["adv_device_on_c"] = losses.get("adv_device_on_c", 0.0) + \
                    F.cross_entropy(batch_out["device_logits_on_c_S"], y_device_s)
            if self.cfg.lambda_pair > 0:
                losses["pair"] = paired_nt_xent_loss(batch_out["c_R"], batch_out["c_S"], self.cfg.contrastive_temperature)
            if self.cfg.lambda_dis > 0:
                losses["disentangle"] = disentanglement_loss(batch_out["c_R"], batch_out["d_R"]) + \
                    disentanglement_loss(batch_out["c_S"], batch_out["d_S"])

        total = losses["event"] + losses["device_on_d"]
        if "adv_device_on_c" in losses:
            total = total + self.cfg.lambda_adv * losses["adv_device_on_c"]
        if "pair" in losses:
            total = total + self.cfg.lambda_pair * losses["pair"]
        if "disentangle" in losses:
            total = total + self.cfg.lambda_dis * losses["disentangle"]
        losses["total"] = total
        return losses
