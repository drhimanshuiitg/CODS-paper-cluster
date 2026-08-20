"""Resumable handcrafted and SSL feature extraction."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Sequence

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import signal
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    Data2VecAudioModel,
    Data2VecVisionModel,
    HubertModel,
    Wav2Vec2Model,
    WavLMModel,
)
from tqdm import tqdm

from .io import config_hash, file_sha256, load_manifest_window, read_csv_rows


FEATURE_DIMENSIONS = {
    "classical": 52,
    "wav2vec2": 768,
    "hubert": 768,
    "wavlm": 768,
    "data2vec_audio": 768,
    "data2vec_spectrogram": 768,
    # SOTA-upgrade candidate (2026-08-19): same transformers/PyTorch pattern
    # as the base encoders above -- no new dependency, no license gate --
    # unlike HeAR/OPERA (TF-Keras, gated "Health AI Developer Foundations"
    # terms, different 2s/16kHz native chunking) which need a deliberate,
    # separately-scoped integration decision. wavlm-large is the cheapest
    # real SOTA test available before that heavier lift.
    "wavlm_large": 1024,
    # Per-subject clinical severity prior (2026-08-19), not derived from
    # audio at all -- Oxygen Desaturation Index and Hypoxic Burden, computed
    # once per subject from the raw SpO2 channel (metadata/odi_hypoxic_burden.csv,
    # see scripts/compute_odi_hypoxic_burden.py), validated against the PSG-
    # annotated OSA+hypopnea event counts (Pearson r=0.83 for ODI, r=0.61 for
    # hypoxic burden). Identical value for every window from a given subject
    # -- cannot discriminate *within* a subject's windows, only contributes a
    # subject-level severity prior the classifier may learn to weight.
    "odi_hb": 2,
    # HeAR (google/hear, gated HF license) -- MAE ViT-L health-acoustic
    # foundation model, ~174k hours pretraining. Unlike the transformers-
    # based encoders above, it only runs in an isolated TF-Keras venv
    # (huggingface_hub.from_pretrained_keras, pinned to huggingface_hub==0.26.2
    # since 1.x dropped that mixin) and only accepts fixed 2.0s/16kHz/32000-
    # sample clips -- extracted via the subprocess-bridge pattern in
    # scripts/extract_hear_features.py (mirrors gpu_classifier_test/
    # cuml_worker.py), NOT through extract_feature_cache()/_load_model()
    # below. Registered here so FEATURE_DIMENSIONS/cache_paths/_open_cache
    # stay the single source of truth the rest of the pipeline reads from.
    "hear": 512,
}

# Features with no model inference to accelerate -- the only two exemptions
# from the project's GPU-only, no-CPU-fallback policy (see GPU_INSTRUCTIONS.md
# and CODEBASE_ARCHITECTURE_AND_EXPERIMENT_LEARNING_GUIDE.md §26). Hoisted to
# module level (2026-08-19) so every caller shares one definition instead of
# each redeclaring its own copy -- a second, incomplete copy in
# benchmark_efficiency.py (missing "odi_hb") is exactly the kind of drift
# this is meant to prevent.
NO_GPU_NEEDED = {"classical", "odi_hb"}

MODEL_SPECS = {
    "wav2vec2": ("facebook/wav2vec2-base", AutoFeatureExtractor, Wav2Vec2Model, "audio"),
    "hubert": ("facebook/hubert-base-ls960", AutoFeatureExtractor, HubertModel, "audio"),
    "wavlm": ("microsoft/wavlm-base", AutoFeatureExtractor, WavLMModel, "audio"),
    "data2vec_audio": ("facebook/data2vec-audio-base-960h", AutoFeatureExtractor, Data2VecAudioModel, "audio"),
    "data2vec_spectrogram": ("facebook/data2vec-vision-base", AutoImageProcessor, Data2VecVisionModel, "vision"),
    "wavlm_large": ("microsoft/wavlm-large", AutoFeatureExtractor, WavLMModel, "audio"),
}


def classical_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(audio) < 512:
        audio = np.pad(audio, (0, 512 - len(audio)))
    n_fft = 400
    hop = 160
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft, hop_length=hop)[0]
    spectrum = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
    descriptors = [
        rms,
        zcr,
        librosa.feature.spectral_centroid(S=spectrum, sr=sample_rate)[0],
        librosa.feature.spectral_bandwidth(S=spectrum, sr=sample_rate)[0],
        librosa.feature.spectral_rolloff(S=spectrum, sr=sample_rate)[0],
        librosa.feature.spectral_flatness(S=spectrum)[0],
    ]
    vector: list[float] = []
    for values in descriptors:
        vector.extend((float(np.mean(values)), float(np.std(values))))
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    for values in mfcc:
        vector.extend((float(np.mean(values)), float(np.std(values))))
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (52,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid classical feature vector: {result.shape}")
    return result


def mel_image(audio: np.ndarray, sample_rate: int, config: dict) -> Image.Image:
    spec = config["spectrogram"]
    power = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=int(spec["n_fft"]),
        hop_length=int(spec["hop_length"]),
        n_mels=int(spec["n_mels"]),
        fmax=min(float(spec["fmax"]), sample_rate / 2),
        power=2.0,
    )
    db = librosa.power_to_db(power, ref=np.max)
    normalized = (db - db.min()) / (db.max() - db.min() + 1e-8)
    rgb = (plt.get_cmap(spec.get("cmap", "magma"))(normalized)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def masked_temporal_mean(hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    if attention_mask.shape[1] != hidden.shape[1]:
        attention_mask = torch.nn.functional.interpolate(
            attention_mask[:, None, :].float(), size=hidden.shape[1], mode="nearest"
        )[:, 0, :]
    weights = attention_mask.to(hidden.device, dtype=hidden.dtype)
    return (hidden * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


def cache_paths(cache_root: Path, feature: str, preprocessing: str) -> dict[str, Path]:
    root = cache_root / feature / preprocessing
    return {
        "root": root,
        "features": root / "features.npy",
        "complete": root / "complete.npy",
        "sample_ids": root / "sample_ids.json",
        "metadata": root / "metadata.json",
        "failures": root / "failures.jsonl",
    }


def _open_cache(paths: dict[str, Path], rows: Sequence[dict], feature: str, preprocessing: str, signature: str):
    root = paths["root"]
    root.mkdir(parents=True, exist_ok=True)
    dimension = FEATURE_DIMENSIONS[feature]
    sample_ids = [row["sample_id"] for row in rows]
    if paths["features"].exists():
        feature_array = np.load(paths["features"], mmap_mode="r+")
        complete = np.load(paths["complete"], mmap_mode="r+")
        cached_ids = json.loads(paths["sample_ids"].read_text(encoding="utf-8"))
        cached_meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if feature_array.shape != (len(rows), dimension) or complete.shape != (len(rows),):
            raise ValueError(f"Cache shape mismatch in {root}")
        if cached_ids != sample_ids or cached_meta.get("signature") != signature:
            raise ValueError(f"Cache signature/order mismatch in {root}")
    else:
        feature_array = np.lib.format.open_memmap(
            paths["features"], mode="w+", dtype="float32", shape=(len(rows), dimension)
        )
        feature_array[:] = np.nan
        feature_array.flush()
        complete = np.lib.format.open_memmap(paths["complete"], mode="w+", dtype="bool", shape=(len(rows),))
        complete[:] = False
        complete.flush()
        paths["sample_ids"].write_text(json.dumps(sample_ids), encoding="utf-8")
        paths["metadata"].write_text(
            json.dumps(
                {
                    "feature": feature,
                    "preprocessing": preprocessing,
                    "dimension": dimension,
                    "rows": len(rows),
                    "signature": signature,
                    "status": "running",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return feature_array, complete


def _load_odi_hb_lookup(config: dict) -> dict[str, np.ndarray]:
    path = Path(config["project_root"]) / "metadata" / "odi_hypoxic_burden.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run scripts/compute_odi_hypoxic_burden.py first"
        )
    lookup: dict[str, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        import csv as _csv

        for row in _csv.DictReader(handle):
            lookup[row["subject_id"]] = np.asarray(
                [float(row["odi"]), float(row["hypoxic_burden"])], dtype=np.float32
            )
    return lookup


def _load_model(feature: str, device: torch.device, local_files_only: bool):
    model_id, processor_class, model_class, kind = MODEL_SPECS[feature]
    processor = processor_class.from_pretrained(model_id, local_files_only=local_files_only)
    model = model_class.from_pretrained(model_id, local_files_only=local_files_only).to(device).eval()
    return model_id, processor, model, kind


def _audio_ssl_vector(
    audio: np.ndarray,
    sample_rate: int,
    processor,
    model,
    device: torch.device,
    max_chunk_seconds: float,
) -> np.ndarray:
    """Pool long events safely, retaining all audio rather than truncating it."""
    chunk_samples = max(1, int(round(max_chunk_seconds * sample_rate)))
    chunks = [audio[start : start + chunk_samples] for start in range(0, len(audio), chunk_samples)]
    minimum_samples = 400
    if len(chunks) > 1 and len(chunks[-1]) < minimum_samples:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()
    elif len(chunks) == 1 and len(chunks[0]) < minimum_samples:
        chunks[0] = np.pad(chunks[0], (0, minimum_samples - len(chunks[0])))
    pooled_chunks: list[torch.Tensor] = []
    weights: list[int] = []
    for chunk in chunks:
        inputs = processor(
            [chunk], sampling_rate=sample_rate, return_tensors="pt", padding=True, return_attention_mask=True
        )
        input_values = inputs.input_values.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model(input_values, attention_mask=attention_mask)
        pooled_chunks.append(masked_temporal_mean(output.last_hidden_state, attention_mask)[0].float().cpu())
        weights.append(int(output.last_hidden_state.shape[1]))
    stacked = torch.stack(pooled_chunks)
    weight_tensor = torch.tensor(weights, dtype=stacked.dtype)[:, None]
    return ((stacked * weight_tensor).sum(dim=0) / weight_tensor.sum()).numpy()


def extract_feature_cache(
    *,
    manifest_path: Path,
    cache_root: Path,
    config: dict,
    feature: str,
    preprocessing: str | None = None,
    local_files_only: bool = False,
    flush_every: int = 25,
) -> dict:
    if feature not in FEATURE_DIMENSIONS:
        raise ValueError(f"Unknown feature: {feature}")
    rows = read_csv_rows(manifest_path)
    if preprocessing is None:
        preprocessing = "peak_filter" if feature == "classical" else "peak"
    signature = config_hash(
        config,
        feature,
        preprocessing,
        file_sha256(manifest_path),
        os.environ.get("TRANSFORMERS_CACHE", ""),
    )
    paths = cache_paths(cache_root, feature, preprocessing)
    # One writer per feature/preprocessing cache. This permits independent
    # encoders to run in parallel while making retries and overlapping SLURM
    # workers safe: a later worker waits, reopens the bitmap, and reuses all
    # rows completed by the first worker.
    paths["root"].mkdir(parents=True, exist_ok=True)
    lock_handle = (paths["root"] / ".extract.lock").open("a+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    feature_array, complete = _open_cache(paths, rows, feature, preprocessing, signature)
    pending = np.flatnonzero(~np.asarray(complete))
    started = time.perf_counter()
    failures: list[dict] = []
    processed = 0
    # odi_hb is a per-subject lookup, not derived from audio at all -- exempt
    # from the GPU requirement the same way classical is, and skip model
    # loading entirely (no encoder involved).
    device = torch.device("cuda" if feature not in NO_GPU_NEEDED and torch.cuda.is_available() else "cpu")
    if feature not in NO_GPU_NEEDED and device.type != "cuda":
        raise RuntimeError(f"GPU is required for {feature}")
    odi_hb_lookup: dict[str, np.ndarray] | None = None
    if feature == "odi_hb":
        odi_hb_lookup = _load_odi_hb_lookup(config)
        model_id, processor, model, input_kind = "odi_hb_lookup", None, None, "lookup"
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        model_id, processor, model, input_kind = _load_model(feature, device, local_files_only)
    else:
        model_id, processor, model, input_kind = "handcrafted", None, None, "audio"
    try:
        progress = tqdm(
            pending,
            total=len(rows),
            initial=int(np.asarray(complete).sum()),
            desc=f"extract {feature}/{preprocessing}",
            unit="clip",
            dynamic_ncols=True,
        )
        for index in progress:
            row = rows[int(index)]
            try:
                if feature == "odi_hb":
                    assert odi_hb_lookup is not None
                    subject_id = row["subject_id"]
                    if subject_id not in odi_hb_lookup:
                        raise KeyError(f"No ODI/hypoxic-burden entry for subject {subject_id}")
                    vector = odi_hb_lookup[subject_id]
                    feature_array[index] = np.asarray(vector, dtype=np.float32)
                    complete[index] = True
                    processed += 1
                    if processed % flush_every == 0:
                        feature_array.flush()
                        complete.flush()
                        progress.set_postfix(completed=int(np.asarray(complete).sum()), refresh=False)
                    continue
                audio, sample_rate = load_manifest_window(row, config, preprocessing)
                if feature == "classical":
                    vector = classical_features(audio, sample_rate)
                elif input_kind == "audio":
                    vector = _audio_ssl_vector(
                        audio,
                        sample_rate,
                        processor,
                        model,
                        device,
                        float(config["audio"].get("ssl_max_chunk_seconds", 20.0)),
                    )
                else:
                    image = mel_image(audio, sample_rate, config)
                    inputs = processor(images=[image], return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        output = model(pixel_values)
                    vector = output.last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
                vector = np.asarray(vector, dtype=np.float32)
                if vector.shape != (FEATURE_DIMENSIONS[feature],) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid feature vector {vector.shape}")
                feature_array[index] = vector
                complete[index] = True
                processed += 1
                if processed % flush_every == 0:
                    feature_array.flush()
                    complete.flush()
                    progress.set_postfix(completed=int(np.asarray(complete).sum()), refresh=False)
            except Exception as error:
                failure = {"index": int(index), "sample_id": row["sample_id"], "error": repr(error)}
                failures.append(failure)
                with paths["failures"].open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure) + "\n")
                raise
    finally:
        feature_array.flush()
        complete.flush()
    elapsed = time.perf_counter() - started
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": "complete" if bool(np.asarray(complete).all()) else "partial",
            "completed_rows": int(np.asarray(complete).sum()),
            "elapsed_sec_this_run": elapsed,
            "processed_this_run": processed,
            "mean_sec_per_processed_clip": elapsed / processed if processed else 0.0,
            "model_id": model_id,
            "device": str(device),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "feature_file_bytes": paths["features"].stat().st_size,
            "failures_this_run": failures,
        }
    )
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metadata["status"] != "complete":
        raise RuntimeError(f"Feature cache incomplete: {paths['root']}")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    return metadata
