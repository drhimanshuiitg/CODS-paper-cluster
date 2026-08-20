#!/usr/bin/env python3
"""
Device-aware sleep-apnea audio benchmark runner.

What this script fixes compared with the two older scripts:
1. Scans recorder and smartphone datasets separately and keeps a `device` column.
2. Runs in-domain, cross-device, and combined-device experiments.
3. Supports the original comparison set: classical features, Wav2Vec2, HuBERT,
   WavLM, one corrected Data2Vec feature set, and SSL fusion.
4. Data2Vec is evaluated as a single representation: raw audio branch +
   Mel-spectrogram image branch concatenated internally. It is not reported as
   separate audio-only and spectrogram-only Data2Vec baselines.
5. Saves metrics, trained sklearn models, predictions, ROC/PR/confusion plots,
   combined ROC curves, dataset summaries, and LaTeX result tables.

Expected label convention:
- normal files contain one of config.labels.normal_keywords.
- apnea files contain one of config.labels.apnea_keywords.
- patient id defaults to the immediate parent folder name.

Run:
    python device_robust_sleep_apnea_experiments.py --config config.yaml
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import random
import re
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    Data2VecAudioModel,
    Data2VecVisionModel,
    HubertModel,
    Wav2Vec2Model,
    WavLMModel,
)

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: str) -> str:
    text = str(text).strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-\.]+", "", text)
    return text[:180]


def expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def json_dump(obj: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Dataset scanning
# -----------------------------------------------------------------------------


def infer_label_from_filename(filename: str, normal_keywords: Sequence[str], apnea_keywords: Sequence[str]) -> Optional[int]:
    """Return 0 for normal, 1 for apnea, None if the file does not match."""
    name = filename.lower()

    # Positive first prevents wrong handling of files such as "hypo_normalized.wav".
    if any(k.lower() in name for k in apnea_keywords):
        return 1
    if any(k.lower() in name for k in normal_keywords):
        return 0
    return None


def infer_patient_id(path: str, root_dir: str, device_name: str, patient_cfg: Dict) -> str:
    """Infer patient/subject id from regex or parent folder."""
    regex = patient_cfg.get("regex")
    if regex:
        m = re.search(regex, path)
        if m:
            return m.group(1) if m.groups() else m.group(0)

    mode = patient_cfg.get("mode", "parent")
    p = Path(path)
    if mode == "parent":
        return p.parent.name
    if mode == "grandparent":
        return p.parent.parent.name
    if mode == "filename_prefix":
        sep = patient_cfg.get("filename_separator", "_")
        return p.stem.split(sep)[0]

    # Safe fallback.
    rel = os.path.relpath(path, root_dir)
    return f"{device_name}_{Path(rel).parts[0]}"


def read_duration_seconds(path: str) -> Optional[float]:
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def scan_one_device(device_name: str, device_cfg: Dict, cfg: Dict) -> pd.DataFrame:
    root_dir = expand_path(device_cfg["root_dir"])
    file_glob = device_cfg.get("file_glob", "**/*.wav")
    normal_keywords = cfg["labels"].get("normal_keywords", ["normal"])
    apnea_keywords = cfg["labels"].get("apnea_keywords", ["osa", "csa", "msa", "mixed", "hypo", "apnea"])
    patient_cfg = device_cfg.get("patient_id", cfg.get("patient_id", {"mode": "parent"}))

    files = sorted(Path(root_dir).glob(file_glob))
    rows = []
    skipped = 0
    for p in files:
        if not p.is_file():
            continue
        label = infer_label_from_filename(p.name, normal_keywords, apnea_keywords)
        if label is None:
            skipped += 1
            continue
        patient_id = infer_patient_id(str(p), root_dir, device_name, patient_cfg)
        rows.append({
            "file_path": str(p),
            "file_name": p.name,
            "device": device_name,
            "label": int(label),
            "patient_id": str(patient_id),
            "duration_sec": read_duration_seconds(str(p)) if cfg.get("dataset", {}).get("compute_duration", False) else np.nan,
        })

    df = pd.DataFrame(rows)
    max_files = device_cfg.get("max_files")
    if max_files and len(df) > max_files:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_files // 2), random_state=cfg["seed"])
        ).reset_index(drop=True)

    print(f"[SCAN] {device_name}: root={root_dir}")
    print(f"       matched={len(df)} labelled wav files | skipped_unlabelled={skipped}")
    if len(df):
        print(df.groupby(["device", "label"]).size().rename("count"))
    return df


def scan_all_devices(cfg: Dict) -> pd.DataFrame:
    frames = []
    for device_name, device_cfg in cfg["datasets"].get("devices", {}).items():
        root_dir = expand_path(device_cfg["root_dir"])
        if not os.path.exists(root_dir):
            print(f"[WARNING] Dataset path not found for device '{device_name}': {root_dir}")
            continue
        df = scan_one_device(device_name, device_cfg, cfg)
        if len(df):
            frames.append(df)

    if not frames:
        raise RuntimeError("No labelled audio files found. Please fix datasets.devices.*.root_dir in config.yaml")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["row_id"] = np.arange(len(all_df))
    all_df["global_group"] = all_df["patient_id"].astype(str)
    return all_df


# -----------------------------------------------------------------------------
# Audio preprocessing and feature extraction
# -----------------------------------------------------------------------------


def load_audio_mono(path: str, target_sr: int, max_seconds: Optional[float] = None, normalize: bool = True) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if max_seconds is not None and max_seconds > 0:
        max_len = int(max_seconds * sr)
        if len(y) > max_len:
            y = y[:max_len]

    if normalize:
        peak = np.max(np.abs(y)) + 1e-8
        y = y / peak

    return y.astype(np.float32), sr


def extract_classical_features(y: np.ndarray, sr: int) -> np.ndarray:
    """52-D classical acoustic feature vector: RMS/ZCR/spectral stats/MFCC stats."""
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))

    nyq = 0.5 * sr
    low = max(20.0 / nyq, 1e-5)
    high = min(4000.0 / nyq, 0.99)
    if low < high:
        b, a = signal.butter(4, [low, high], btype="band")
        try:
            y_filtered = signal.filtfilt(b, a, y)
        except Exception:
            y_filtered = y
    else:
        y_filtered = y

    n_fft = 400
    hop_length = 160

    rms = librosa.feature.rms(y=y_filtered, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_filtered, frame_length=n_fft, hop_length=hop_length)[0]
    S = np.abs(librosa.stft(y_filtered, n_fft=n_fft, hop_length=hop_length))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)

    vec = [
        np.mean(rms), np.std(rms),
        np.mean(zcr), np.std(zcr),
        np.mean(centroid), np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff), np.std(rolloff),
        np.mean(flatness), np.std(flatness),
    ]
    for i in range(20):
        vec.extend([np.mean(mfcc[i]), np.std(mfcc[i])])

    return np.nan_to_num(np.asarray(vec, dtype=np.float32))


def generate_mel_spectrogram_image(y: np.ndarray, sr: int, cfg: Dict) -> Image.Image:
    """Convert waveform into RGB Mel-spectrogram image for Data2VecVisionModel."""
    spec_cfg = cfg.get("spectrogram", {})
    n_mels = int(spec_cfg.get("n_mels", 128))
    n_fft = int(spec_cfg.get("n_fft", 1024))
    hop_length = int(spec_cfg.get("hop_length", 256))
    fmax = spec_cfg.get("fmax", sr / 2)
    cmap = spec_cfg.get("cmap", "magma")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=min(float(fmax), sr / 2),
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    cmap_fn = plt.get_cmap(cmap)
    rgb = (cmap_fn(S_norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


@dataclass
class HFSpec:
    feature_key: str
    model_id: str
    model_class: object
    processor_id: str
    processor_class: object
    input_kind: str  # "audio" or "vision"


HF_FEATURES = {
    "wav2vec2": HFSpec("wav2vec2", "facebook/wav2vec2-base", Wav2Vec2Model, "facebook/wav2vec2-base", AutoProcessor, "audio"),
    "hubert": HFSpec("hubert", "facebook/hubert-base-ls960", HubertModel, "facebook/hubert-base-ls960", AutoProcessor, "audio"),
    "wavlm": HFSpec("wavlm", "microsoft/wavlm-base", WavLMModel, "microsoft/wavlm-base", AutoProcessor, "audio"),
}


DATA2VEC_AUDIO_MODEL_ID = "facebook/data2vec-audio-base-960h"
DATA2VEC_VISION_MODEL_ID = "facebook/data2vec-vision-base"


BASE_REPRESENTATION_DEPENDENCIES = {
    # Original benchmark-style single representations.
    "classical": ["classical"],
    "wav2vec2": ["wav2vec2"],
    "hubert": ["hubert"],
    "wavlm": ["wavlm"],

    # SSL fusion baseline: all four self-supervised representations together.
    # Important: classical handcrafted features are intentionally NOT included here.
    # Data2Vec is already one fused audio+spectrogram vector.
    "ssl_fusion": ["wav2vec2", "hubert", "wavlm", "data2vec"],

    # Corrected proposed Data2Vec representation.
    # Internally: raw waveform -> Data2VecAudioModel and same waveform -> Mel-spectrogram
    # image -> Data2VecVisionModel; the two embeddings are concatenated and reported as
    # ONE Data2Vec feature set, not as separate audio-only/spectrogram-only results.
    "data2vec": ["data2vec"],
}


def expand_representation_dependencies(cfg: Dict) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Keep the representation grid intentionally small and close to the original code.

    No automatic pair/triple/quad fusion generation is performed here. This prevents
    the experiment from becoming unnecessarily heavy and keeps the paper comparison clean.
    """
    rep_deps: Dict[str, List[str]] = {k: list(v) for k, v in BASE_REPRESENTATION_DEPENDENCIES.items()}

    selected: List[str] = []
    for rep in cfg.get("run", {}).get("representations", []):
        if rep == "all_named":
            for name in BASE_REPRESENTATION_DEPENDENCIES:
                if name not in selected:
                    selected.append(name)
        else:
            if rep not in rep_deps:
                raise ValueError(
                    f"Unknown representation '{rep}'. Valid named reps: {sorted(rep_deps)}. "
                    "This v4 runner intentionally does not auto-generate heavy fusion combinations."
                )
            if rep not in selected:
                selected.append(rep)

    valid_base = {"classical", "data2vec", *HF_FEATURES.keys()}
    for rep in selected:
        for dep in rep_deps[rep]:
            if dep not in valid_base:
                raise ValueError(f"Representation '{rep}' depends on unsupported base feature '{dep}'")

    if not selected:
        raise ValueError("No representations selected. Add run.representations in config.yaml")
    return selected, rep_deps


def representation_dimensions(feature_matrices: Dict[str, np.ndarray], rep_deps: Dict[str, List[str]]) -> Dict[str, int]:
    dims = {}
    for rep, deps in rep_deps.items():
        if all(d in feature_matrices for d in deps):
            dims[rep] = int(sum(feature_matrices[d].shape[1] for d in deps))
    return dims


def dataset_cache_signature(df: pd.DataFrame, feature_key: str, cfg: Dict) -> str:
    """Signature for matrix-level cache: order, file identity, and feature settings."""
    rows = []
    for path in df["file_path"].tolist():
        try:
            st = os.stat(path)
            rows.append(f"{path}|{st.st_size}|{int(st.st_mtime)}")
        except Exception:
            rows.append(str(path))
    payload = {
        "feature_key": feature_key,
        "feature_hash": feature_config_hash(feature_key, cfg),
        "rows": rows,
    }
    return sha1_text(json.dumps(payload, sort_keys=True))


def matrix_cache_path(cache_root: Path, feature_key: str, signature: str) -> Path:
    return cache_root / "matrices" / f"{safe_name(feature_key)}__{signature}.npy"

def cache_file_path(cache_root: Path, feature_key: str, file_path: str, cfg_hash: str) -> Path:
    try:
        stat = os.stat(file_path)
        stamp = f"{file_path}|{stat.st_size}|{int(stat.st_mtime)}|{feature_key}|{cfg_hash}"
    except Exception:
        stamp = f"{file_path}|{feature_key}|{cfg_hash}"
    return cache_root / feature_key / f"{sha1_text(stamp)}.npy"


def feature_config_hash(feature_key: str, cfg: Dict) -> str:
    if feature_key in HF_FEATURES:
        model_id = HF_FEATURES[feature_key].model_id
    elif feature_key == "data2vec":
        model_id = f"{DATA2VEC_AUDIO_MODEL_ID}+{DATA2VEC_VISION_MODEL_ID}"
    else:
        model_id = "none"
    relevant = {
        "feature_key": feature_key,
        "audio": cfg.get("audio", {}),
        "spectrogram": cfg.get("spectrogram", {}),
        "hf_model": model_id,
    }
    return sha1_text(json.dumps(relevant, sort_keys=True))


def extract_classical_matrix(df: pd.DataFrame, cfg: Dict, cache_root: Path) -> np.ndarray:
    target_sr = int(cfg.get("audio", {}).get("target_sr", 16000))
    max_seconds = cfg.get("audio", {}).get("max_seconds")
    normalize = bool(cfg.get("audio", {}).get("normalize", True))
    cfg_hash = feature_config_hash("classical", cfg)

    vectors = []
    ensure_dir(cache_root / "classical")
    for path in tqdm(df["file_path"].tolist(), desc="Extracting classical features"):
        cpath = cache_file_path(cache_root, "classical", path, cfg_hash)
        if cfg.get("cache", {}).get("enabled", True) and cpath.exists():
            vectors.append(np.load(cpath))
            continue
        try:
            y, sr = load_audio_mono(path, target_sr, max_seconds=max_seconds, normalize=normalize)
            vec = extract_classical_features(y, sr)
            np.save(cpath, vec)
            vectors.append(vec)
        except Exception as e:
            print(f"[SKIP] classical failed for {path}: {e}")
            vectors.append(np.full(52, np.nan, dtype=np.float32))
    return np.nan_to_num(np.vstack(vectors).astype(np.float32))


def mean_pool_last_hidden(output) -> np.ndarray:
    return output.last_hidden_state.mean(dim=1).detach().float().cpu().numpy()


def extract_hf_matrix(df: pd.DataFrame, feature_key: str, cfg: Dict, cache_root: Path, device: torch.device) -> np.ndarray:
    spec = HF_FEATURES[feature_key]
    hf_cfg = cfg.get("hf", {})
    target_sr = int(cfg.get("audio", {}).get("target_sr", 16000))
    max_seconds = cfg.get("audio", {}).get("max_seconds")
    normalize = bool(cfg.get("audio", {}).get("normalize", True))
    batch_size = int(hf_cfg.get("batch_size", 4))
    local_files_only = bool(hf_cfg.get("local_files_only", False))
    use_amp = bool(hf_cfg.get("use_amp", True)) and torch.cuda.is_available()
    use_dataparallel = bool(hf_cfg.get("data_parallel", True)) and torch.cuda.device_count() > 1
    cfg_hash = feature_config_hash(feature_key, cfg)

    ensure_dir(cache_root / feature_key)
    all_paths = df["file_path"].tolist()
    cache_paths = [cache_file_path(cache_root, feature_key, p, cfg_hash) for p in all_paths]

    missing_positions = [i for i, cp in enumerate(cache_paths) if not (cfg.get("cache", {}).get("enabled", True) and cp.exists())]
    print(f"[FEATURE] {feature_key}: total={len(all_paths)} | missing_cache={len(missing_positions)}")

    if missing_positions:
        processor = spec.processor_class.from_pretrained(spec.processor_id, local_files_only=local_files_only)
        model = spec.model_class.from_pretrained(spec.model_id, local_files_only=local_files_only).to(device).eval()
        if use_dataparallel:
            print(f"[GPU] Using DataParallel for {feature_key} on {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

        with torch.no_grad():
            for start in tqdm(range(0, len(missing_positions), batch_size), desc=f"Extracting {feature_key}"):
                positions = missing_positions[start:start + batch_size]
                batch_paths = [all_paths[i] for i in positions]
                valid_positions = []
                audio_batch = []
                image_batch = []

                for pos, path in zip(positions, batch_paths):
                    try:
                        y, sr = load_audio_mono(path, target_sr, max_seconds=max_seconds, normalize=normalize)
                        if spec.input_kind == "audio":
                            audio_batch.append(y)
                        else:
                            image_batch.append(generate_mel_spectrogram_image(y, sr, cfg))
                        valid_positions.append(pos)
                    except Exception as e:
                        print(f"[SKIP] load failed for {path}: {e}")

                if not valid_positions:
                    continue

                try:
                    if spec.input_kind == "audio":
                        inputs = processor(audio_batch, sampling_rate=target_sr, return_tensors="pt", padding=True)
                        input_values = inputs.input_values.to(device)
                        with amp_ctx:
                            out = model(input_values)
                        arr = mean_pool_last_hidden(out)
                        del input_values, out
                    else:
                        inputs = processor(images=image_batch, return_tensors="pt")
                        pixel_values = inputs.pixel_values.to(device)
                        with amp_ctx:
                            out = model(pixel_values)
                        arr = mean_pool_last_hidden(out)
                        del pixel_values, out

                    for pos, vec in zip(valid_positions, arr):
                        np.save(cache_paths[pos], vec.astype(np.float32))
                except Exception as e:
                    print(f"[ERROR] batch failed for {feature_key}: {e}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vectors = []
    failed = 0
    for cp in cache_paths:
        if cp.exists():
            vectors.append(np.load(cp).astype(np.float32))
        else:
            failed += 1
            vectors.append(None)

    if failed:
        # Infer dimension from first valid vector.
        first = next((v for v in vectors if v is not None), None)
        if first is None:
            raise RuntimeError(f"All feature extractions failed for {feature_key}")
        vectors = [v if v is not None else np.full_like(first, np.nan, dtype=np.float32) for v in vectors]
        print(f"[WARNING] {feature_key}: {failed} files failed and were filled with zeros after nan_to_num")

    return np.nan_to_num(np.vstack(vectors).astype(np.float32))


def extract_data2vec_matrix(df: pd.DataFrame, cfg: Dict, cache_root: Path, device: torch.device) -> np.ndarray:
    """
    Corrected Data2Vec feature extractor.

    This is the ONLY Data2Vec representation used by the benchmark. For each audio clip:
      1) raw waveform is passed to Data2VecAudioModel,
      2) the same waveform is converted to a Mel-spectrogram RGB image and passed to
         Data2VecVisionModel,
      3) both mean-pooled embeddings are concatenated and cached as one vector.

    The paper should call this "Data2Vec audio-spectrogram fusion" or simply
    "Data2Vec (audio + spectrogram)". Do not report the internal two branches as
    separate baselines unless you deliberately add that ablation later.
    """
    hf_cfg = cfg.get("hf", {})
    target_sr = int(cfg.get("audio", {}).get("target_sr", 16000))
    max_seconds = cfg.get("audio", {}).get("max_seconds")
    normalize = bool(cfg.get("audio", {}).get("normalize", True))
    batch_size = int(hf_cfg.get("batch_size", 4))
    local_files_only = bool(hf_cfg.get("local_files_only", False))
    use_amp = bool(hf_cfg.get("use_amp", True)) and torch.cuda.is_available()
    use_dataparallel = bool(hf_cfg.get("data_parallel", True)) and torch.cuda.device_count() > 1
    cfg_hash = feature_config_hash("data2vec", cfg)

    feature_key = "data2vec"
    ensure_dir(cache_root / feature_key)
    all_paths = df["file_path"].tolist()
    cache_paths = [cache_file_path(cache_root, feature_key, p, cfg_hash) for p in all_paths]

    cache_enabled = bool(cfg.get("cache", {}).get("enabled", True))
    missing_positions = [i for i, cp in enumerate(cache_paths) if not (cache_enabled and cp.exists())]
    print(f"[FEATURE] data2vec: total={len(all_paths)} | missing_cache={len(missing_positions)}")

    if missing_positions:
        audio_processor = AutoProcessor.from_pretrained(DATA2VEC_AUDIO_MODEL_ID, local_files_only=local_files_only)
        vision_processor = AutoImageProcessor.from_pretrained(DATA2VEC_VISION_MODEL_ID, local_files_only=local_files_only)
        audio_model = Data2VecAudioModel.from_pretrained(DATA2VEC_AUDIO_MODEL_ID, local_files_only=local_files_only).to(device).eval()
        vision_model = Data2VecVisionModel.from_pretrained(DATA2VEC_VISION_MODEL_ID, local_files_only=local_files_only).to(device).eval()

        if use_dataparallel:
            print(f"[GPU] Using DataParallel for data2vec on {torch.cuda.device_count()} GPUs")
            audio_model = torch.nn.DataParallel(audio_model)
            vision_model = torch.nn.DataParallel(vision_model)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

        with torch.no_grad():
            for start in tqdm(range(0, len(missing_positions), batch_size), desc="Extracting data2vec(audio+spectrogram)"):
                positions = missing_positions[start:start + batch_size]
                valid_positions = []
                audio_batch = []
                image_batch = []

                for pos in positions:
                    path = all_paths[pos]
                    try:
                        y, sr = load_audio_mono(path, target_sr, max_seconds=max_seconds, normalize=normalize)
                        audio_batch.append(y)
                        image_batch.append(generate_mel_spectrogram_image(y, sr, cfg))
                        valid_positions.append(pos)
                    except Exception as e:
                        print(f"[SKIP] data2vec load failed for {path}: {e}")

                if not valid_positions:
                    continue

                try:
                    audio_inputs = audio_processor(
                        audio_batch,
                        sampling_rate=target_sr,
                        return_tensors="pt",
                        padding=True,
                    ).input_values.to(device)
                    vision_inputs = vision_processor(images=image_batch, return_tensors="pt").pixel_values.to(device)

                    with amp_ctx:
                        audio_out = audio_model(audio_inputs)
                        vision_out = vision_model(vision_inputs)

                    audio_vec = mean_pool_last_hidden(audio_out)
                    vision_vec = mean_pool_last_hidden(vision_out)
                    fused = np.concatenate([audio_vec, vision_vec], axis=1).astype(np.float32)

                    for pos, vec in zip(valid_positions, fused):
                        np.save(cache_paths[pos], vec)

                    del audio_inputs, vision_inputs, audio_out, vision_out, audio_vec, vision_vec, fused
                except Exception as e:
                    print(f"[ERROR] data2vec batch failed: {e}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del audio_model, vision_model, audio_processor, vision_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vectors = []
    failed = 0
    for cp in cache_paths:
        if cp.exists():
            vectors.append(np.load(cp).astype(np.float32))
        else:
            failed += 1
            vectors.append(None)

    if failed:
        first = next((v for v in vectors if v is not None), None)
        if first is None:
            raise RuntimeError("All Data2Vec feature extractions failed")
        vectors = [v if v is not None else np.full_like(first, np.nan, dtype=np.float32) for v in vectors]
        print(f"[WARNING] data2vec: {failed} files failed and were filled with zeros after nan_to_num")

    return np.nan_to_num(np.vstack(vectors).astype(np.float32))


def required_base_features(representations: Sequence[str], rep_deps: Dict[str, List[str]]) -> List[str]:
    required: List[str] = []
    for rep in representations:
        if rep not in rep_deps:
            raise ValueError(f"Unknown representation '{rep}'. Valid: {sorted(rep_deps)}")
        for dep in rep_deps[rep]:
            if dep not in required:
                required.append(dep)
    return required


def extract_all_required_features(
    df: pd.DataFrame,
    cfg: Dict,
    output_dir: Path,
    device: torch.device,
    representations: Sequence[str],
    rep_deps: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    required = required_base_features(representations, rep_deps)
    cache_root = ensure_dir(output_dir / "cache")
    ensure_dir(cache_root / "matrices")

    matrices: Dict[str, np.ndarray] = {}
    matrix_cache_enabled = bool(cfg.get("cache", {}).get("enabled", True)) and bool(cfg.get("cache", {}).get("matrix_cache", True))

    print(f"[FEATURE] Required base features: {required}")
    for feature_key in required:
        sig = dataset_cache_signature(df, feature_key, cfg)
        mpath = matrix_cache_path(cache_root, feature_key, sig)
        if matrix_cache_enabled and mpath.exists():
            print(f"[FEATURE] {feature_key}: loading matrix cache -> {mpath}")
            matrices[feature_key] = np.load(mpath).astype(np.float32)
        else:
            if feature_key == "classical":
                matrices[feature_key] = extract_classical_matrix(df, cfg, cache_root)
            elif feature_key == "data2vec":
                matrices[feature_key] = extract_data2vec_matrix(df, cfg, cache_root, device)
            elif feature_key in HF_FEATURES:
                matrices[feature_key] = extract_hf_matrix(df, feature_key, cfg, cache_root, device)
            else:
                raise ValueError(f"Unsupported base feature: {feature_key}")
            if matrix_cache_enabled:
                np.save(mpath, matrices[feature_key].astype(np.float32))
                print(f"[FEATURE] {feature_key}: saved matrix cache -> {mpath}")

        print(f"[FEATURE] {feature_key}: shape={matrices[feature_key].shape}")

    return matrices


def representation_matrix(
    feature_matrices: Dict[str, np.ndarray],
    representation: str,
    row_indices: Sequence[int],
    rep_deps: Dict[str, List[str]],
) -> np.ndarray:
    deps = rep_deps[representation]
    parts = [feature_matrices[d][row_indices] for d in deps]
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


# -----------------------------------------------------------------------------
# Experiment protocols
# -----------------------------------------------------------------------------


def group_split_indices(df: pd.DataFrame, test_size: float, seed: int, group_col: str = "global_group") -> Tuple[np.ndarray, np.ndarray]:
    if df[group_col].nunique() < 2:
        raise RuntimeError(f"Need at least two unique groups in {group_col} for grouped split.")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rel_train, rel_test = next(gss.split(df, df["label"], groups=df[group_col]))
    return df.iloc[rel_train]["row_id"].to_numpy(), df.iloc[rel_test]["row_id"].to_numpy()


def build_experiment(df: pd.DataFrame, exp_name: str, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    seed = int(cfg.get("seed", 42))
    test_size = float(cfg.get("split", {}).get("test_size", 0.25))
    strict_cross = bool(cfg.get("split", {}).get("strict_patient_disjoint_cross_device", False))

    devices = sorted(df["device"].unique().tolist())

    if exp_name.startswith("in_domain_"):
        dev = exp_name.replace("in_domain_", "")
        subset = df[df["device"] == dev].copy()
        if subset.empty:
            raise RuntimeError(f"No samples found for device '{dev}'")
        train_idx, test_idx = group_split_indices(subset, test_size, seed)
        meta = {"protocol": "in_domain_group_split", "train_device": dev, "test_device": dev}
        return train_idx, test_idx, meta

    if exp_name.startswith("cross_"):
        # Expected format: cross_recorder_to_smartphone
        m = re.match(r"cross_(.+)_to_(.+)", exp_name)
        if not m:
            raise ValueError("Cross-device experiment must be named cross_<source>_to_<target>")
        source, target = m.group(1), m.group(2)
        train_df = df[df["device"] == source].copy()
        test_df = df[df["device"] == target].copy()
        if train_df.empty or test_df.empty:
            raise RuntimeError(f"Missing source/target samples for {exp_name}. Available devices: {devices}")
        if strict_cross:
            train_patients = set(train_df["patient_id"].astype(str))
            before = len(test_df)
            test_df = test_df[~test_df["patient_id"].astype(str).isin(train_patients)].copy()
            print(f"[STRICT] Removed {before - len(test_df)} target samples with patients seen in source training set")
            if test_df.empty:
                raise RuntimeError(f"Strict patient-disjoint filtering made test set empty for {exp_name}")
        meta = {"protocol": "cross_device", "train_device": source, "test_device": target, "strict_patient_disjoint": strict_cross}
        return train_df["row_id"].to_numpy(), test_df["row_id"].to_numpy(), meta

    if exp_name == "combined_device_group_split":
        train_idx, test_idx = group_split_indices(df.copy(), test_size, seed)
        meta = {"protocol": "combined_device_group_split", "train_device": "recorder+smartphone", "test_device": "recorder+smartphone"}
        return train_idx, test_idx, meta

    raise ValueError(f"Unknown experiment name: {exp_name}")


# -----------------------------------------------------------------------------
# Classifiers and evaluation
# -----------------------------------------------------------------------------


def make_classifier(name: str, cfg: Dict):
    clf_cfg = cfg.get("classifiers", {}).get(name, {})
    seed = int(cfg.get("seed", 42))

    if name == "logistic_regression":
        base = LogisticRegression(
            max_iter=int(clf_cfg.get("max_iter", 2000)),
            C=float(clf_cfg.get("C", 1.0)),
            class_weight=clf_cfg.get("class_weight", "balanced"),
            solver=clf_cfg.get("solver", "lbfgs"),
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", base)])

    if name == "svm_rbf":
        base = SVC(
            kernel="rbf",
            C=float(clf_cfg.get("C", 1.0)),
            gamma=clf_cfg.get("gamma", "scale"),
            probability=bool(clf_cfg.get("probability", True)),
            class_weight=clf_cfg.get("class_weight", "balanced"),
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", base)])

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(clf_cfg.get("n_estimators", 300)),
            max_depth=clf_cfg.get("max_depth", None),
            class_weight=clf_cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=int(clf_cfg.get("n_jobs", -1)),
        )

    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Remove xgboost from run.classifiers or install xgboost.")
        return XGBClassifier(
            n_estimators=int(clf_cfg.get("n_estimators", 300)),
            max_depth=int(clf_cfg.get("max_depth", 4)),
            learning_rate=float(clf_cfg.get("learning_rate", 0.05)),
            subsample=float(clf_cfg.get("subsample", 0.9)),
            colsample_bytree=float(clf_cfg.get("colsample_bytree", 0.9)),
            eval_metric="logloss",
            random_state=seed,
            n_jobs=int(clf_cfg.get("n_jobs", -1)),
        )

    if name == "mlp":
        base = MLPClassifier(
            hidden_layer_sizes=tuple(clf_cfg.get("hidden_layer_sizes", [256, 128])),
            max_iter=int(clf_cfg.get("max_iter", 400)),
            alpha=float(clf_cfg.get("alpha", 0.0001)),
            random_state=seed,
            early_stopping=bool(clf_cfg.get("early_stopping", True)),
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", base)])

    raise ValueError(f"Unknown classifier: {name}")


def maybe_set_xgb_class_balance(model, name: str, y_train: np.ndarray):
    if name != "xgboost":
        return model
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    if pos > 0:
        model.set_params(scale_pos_weight=max(1.0, float(neg) / float(pos)))
    return model


def get_scores(model, X: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return continuous score for positive class and whether it is a probability."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1], True
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores).ravel(), False
    pred = model.predict(X)
    return np.asarray(pred).ravel(), False


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    out = {
        "n_test": int(len(y_true)),
        "support_normal": int(np.sum(y_true == 0)),
        "support_apnea": int(np.sum(y_true == 1)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp + 1e-12),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["average_precision_pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan
        out["average_precision_pr_auc"] = np.nan
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Apnea"]); ax.set_yticklabels(["Normal", "Apnea"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, path: Path, title: str) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return float(auc)


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, path: Path, title: str) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall / Sensitivity")
    ax.set_ylabel("Precision / PPV")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return float(ap)


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, path: Path, title: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed apnea fraction")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_combined_roc(items: List[Dict], path: Path, title: str) -> None:
    valid = [it for it in items if len(np.unique(it["y_true"])) == 2]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for it in valid:
        fpr, tpr, _ = roc_curve(it["y_true"], it["y_score"])
        auc = roc_auc_score(it["y_true"], it["y_score"])
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{it['representation']} + {it['classifier']} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Sensitivity")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_metric_summary(results_df: pd.DataFrame, output_dir: Path) -> None:
    if results_df.empty:
        return
    for metric in ["roc_auc", "macro_f1", "balanced_accuracy", "recall_sensitivity", "specificity"]:
        if metric not in results_df.columns:
            continue
        plot_df = results_df[results_df["scope"] == "overall"].copy()
        plot_df = plot_df.dropna(subset=[metric])
        if plot_df.empty:
            continue
        plot_df["label"] = plot_df["experiment"] + "\n" + plot_df["representation"] + "+" + plot_df["classifier"]
        plot_df = plot_df.sort_values(metric, ascending=False).head(40)
        fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(plot_df)), 6))
        ax.bar(np.arange(len(plot_df)), plot_df[metric].values)
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df["label"].tolist(), rotation=75, ha="right", fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(f"Top configurations by {metric}")
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / f"summary_top_{metric}.png", dpi=300)
        plt.close(fig)


def plot_tsne_if_enabled(X: np.ndarray, y: np.ndarray, path: Path, title: str, enabled: bool) -> None:
    if not enabled or len(X) < 10:
        return
    perplexity = min(30, max(5, (len(X) - 1) // 3))
    try:
        Z = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 6))
        for cls, label in [(0, "Normal"), (1, "Apnea")]:
            mask = y == cls
            ax.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.7, label=label)
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] t-SNE failed for {title}: {e}")



# -----------------------------------------------------------------------------
# Final comparison CSV and journal-style LaTeX table generation
# -----------------------------------------------------------------------------


def add_result_ranks(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    if df.empty:
        return df
    overall_mask = df["scope"].eq("overall") if "scope" in df.columns else pd.Series(False, index=df.index)
    for metric, rank_col in [
        ("roc_auc", "rank_auc_within_experiment"),
        ("balanced_accuracy", "rank_balacc_within_experiment"),
        ("macro_f1", "rank_macro_f1_within_experiment"),
    ]:
        df[rank_col] = np.nan
        if metric in df.columns and "experiment" in df.columns:
            df.loc[overall_mask, rank_col] = df.loc[overall_mask].groupby("experiment")[metric].rank(
                ascending=False, method="min"
            )
    return df


def save_final_comparison_csv(results_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Create one compared CSV containing overall and per-device rows with ranks."""
    ranked = add_result_ranks(results_df)
    preferred_cols = [
        "scope", "device_scope", "experiment", "protocol", "train_device", "test_device",
        "strict_patient_disjoint", "representation", "classifier", "train_samples", "test_samples",
        "n_test", "support_normal", "support_apnea", "roc_auc", "average_precision_pr_auc",
        "accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity",
        "f1", "macro_f1", "mcc", "cohen_kappa", "tn", "fp", "fn", "tp",
        "rank_auc_within_experiment", "rank_balacc_within_experiment", "rank_macro_f1_within_experiment",
    ]
    cols = [c for c in preferred_cols if c in ranked.columns] + [c for c in ranked.columns if c not in preferred_cols]
    ranked = ranked[cols]
    ranked.to_csv(output_dir / "final_comparison_all_results.csv", index=False)

    if "scope" in ranked.columns:
        overall = ranked[ranked["scope"] == "overall"].copy()
        if not overall.empty:
            overall.sort_values(["experiment", "roc_auc", "balanced_accuracy", "macro_f1"], ascending=[True, False, False, False]).to_csv(
                output_dir / "final_overall_comparison_sorted.csv", index=False
            )
            best_per_experiment = overall.sort_values(["experiment", "roc_auc", "balanced_accuracy", "macro_f1"], ascending=[True, False, False, False]).groupby("experiment", as_index=False).head(1)
            best_per_experiment.to_csv(output_dir / "best_model_per_protocol.csv", index=False)
    return ranked


def latex_escape(text: object) -> str:
    s = "" if pd.isna(text) else str(text)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def fmt_num(x: object, percent: bool = False, decimals: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return latex_escape(x)
    if np.isnan(val):
        return "--"
    if percent:
        return f"{100.0 * val:.1f}"
    return f"{val:.{decimals}f}"


def table_to_latex_tabular(df: pd.DataFrame, col_formats: Optional[Dict[str, Tuple[bool, int]]] = None) -> str:
    col_formats = col_formats or {}
    lines = []
    cols = list(df.columns)
    align = "l" + "c" * (len(cols) - 1)
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            percent, decimals = col_formats.get(c, (False, 3))
            val = row[c]
            if isinstance(val, (int, np.integer)) and not percent:
                cells.append(str(int(val)))
            elif isinstance(val, (float, np.floating)) or percent:
                cells.append(fmt_num(val, percent=percent, decimals=decimals))
            else:
                cells.append(latex_escape(val))
        lines.append(" & ".join(cells) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def wrap_latex_table(tabular: str, caption: str, label: str, note: Optional[str] = None, table_star: bool = True) -> str:
    env = "table*" if table_star else "table"
    lines = [
        f"\\begin{{{env}}}[!t]",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        tabular,
        "\\end{adjustbox}",
    ]
    if note:
        lines.append(f"\\vspace{{1mm}}\\footnotesize{{{latex_escape(note)}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines)


def compact_result_table(df: pd.DataFrame, max_rows: int = 20) -> pd.DataFrame:
    cols = [
        "experiment", "train_device", "test_device", "representation", "classifier",
        "roc_auc", "balanced_accuracy", "recall_sensitivity", "specificity", "macro_f1", "mcc",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy().head(max_rows)
    rename = {
        "experiment": "Protocol", "train_device": "Train", "test_device": "Test",
        "representation": "Representation", "classifier": "Classifier", "roc_auc": "AUC",
        "balanced_accuracy": "Bal. Acc.", "recall_sensitivity": "Sensitivity",
        "specificity": "Specificity", "macro_f1": "Macro-F1", "mcc": "MCC",
    }
    return out.rename(columns=rename)


def write_latex_tables(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Write journal-ready booktabs tables using the actual generated result CSV."""
    latex_dir = ensure_dir(output_dir / "latex_tables")
    pieces = []
    header = r"""% Auto-generated by device_robust_sleep_apnea_experiments.py
% Recommended packages:
% \usepackage{booktabs}
% \usepackage{adjustbox}
% \usepackage{multirow}
% Metrics shown as percentages unless noted otherwise; MCC is unitless.
"""
    pieces.append(header)

    # Dataset summary.
    dataset_path = output_dir / "dataset_summary.csv"
    if dataset_path.exists():
        ds = pd.read_csv(dataset_path)
        ds["label"] = ds["label"].map({0: "Normal", 1: "Apnea"}).fillna(ds["label"])
        ds = ds.rename(columns={"device": "Device", "label": "Class", "samples": "Clips", "patients": "Patients"})
        tab = table_to_latex_tabular(ds)
        tex = wrap_latex_table(tab, "Device-wise dataset summary used for robustness evaluation.", "tab:dataset_device_summary", table_star=False)
        (latex_dir / "table_dataset_summary.tex").write_text(tex, encoding="utf-8")
        pieces.append(tex)

    if results_df.empty or "scope" not in results_df.columns:
        (latex_dir / "journal_results_tables.tex").write_text("\n\n".join(pieces), encoding="utf-8")
        return

    overall = results_df[results_df["scope"] == "overall"].copy()
    if overall.empty:
        (latex_dir / "journal_results_tables.tex").write_text("\n\n".join(pieces), encoding="utf-8")
        return

    sort_cols = [c for c in ["experiment", "roc_auc", "balanced_accuracy", "macro_f1"] if c in overall.columns]
    ascending = [True, False, False, False][:len(sort_cols)]
    overall_sorted = overall.sort_values(sort_cols, ascending=ascending)

    metric_formats = {
        "AUC": (True, 1), "Bal. Acc.": (True, 1), "Sensitivity": (True, 1),
        "Specificity": (True, 1), "Macro-F1": (True, 1), "MCC": (False, 3),
    }

    # Best model for each protocol.
    best_protocol = overall_sorted.groupby("experiment", as_index=False).head(1)
    best_protocol_ltx = compact_result_table(best_protocol, max_rows=50)
    tex = wrap_latex_table(
        table_to_latex_tabular(best_protocol_ltx, metric_formats),
        "Best-performing configuration under each device-robustness protocol.",
        "tab:best_by_device_protocol",
        note="Best row selected by AUC, then balanced accuracy, then macro-F1.",
    )
    (latex_dir / "table_best_by_protocol.tex").write_text(tex, encoding="utf-8")
    pieces.append(tex)

    # Cross-device robustness table.
    cross = overall_sorted[overall_sorted.get("protocol", pd.Series(index=overall_sorted.index, dtype=str)).eq("cross_device")]
    if not cross.empty:
        cross_best = cross.groupby("experiment", as_index=False).head(1)
        cross_ltx = compact_result_table(cross_best, max_rows=20)
        tex = wrap_latex_table(
            table_to_latex_tabular(cross_ltx, metric_formats),
            "Cross-device generalization results for recorder-smartphone robustness.",
            "tab:cross_device_robustness",
            note="These rows are the strongest evidence for or against device robustness.",
        )
        (latex_dir / "table_cross_device_robustness.tex").write_text(tex, encoding="utf-8")
        pieces.append(tex)

    # Top-N full comparison.
    topn = overall.sort_values(["roc_auc", "balanced_accuracy", "macro_f1"], ascending=[False, False, False]).head(30)
    topn_ltx = compact_result_table(topn, max_rows=30)
    tex = wrap_latex_table(
        table_to_latex_tabular(topn_ltx, metric_formats),
        "Top configurations across all protocols, feature representations, and classifiers.",
        "tab:top_configurations_all",
    )
    (latex_dir / "table_top30_configurations.tex").write_text(tex, encoding="utf-8")
    pieces.append(tex)

    # Representation x protocol pivot, best AUC over classifiers.
    if {"experiment", "representation", "roc_auc"}.issubset(overall.columns):
        rep_mean = overall.groupby("representation")["roc_auc"].mean().sort_values(ascending=False).head(12).index.tolist()
        piv = overall[overall["representation"].isin(rep_mean)].groupby(["representation", "experiment"])["roc_auc"].max().reset_index()
        piv = piv.pivot(index="representation", columns="experiment", values="roc_auc").reset_index()
        piv = piv.rename(columns={"representation": "Representation"})
        pivot_formats = {c: (True, 1) for c in piv.columns if c != "Representation"}
        tex = wrap_latex_table(
            table_to_latex_tabular(piv, pivot_formats),
            "Best AUC of each representation across device protocols.",
            "tab:representation_protocol_auc",
            note="For each representation and protocol, the best classifier AUC is shown.",
        )
        (latex_dir / "table_representation_protocol_auc.tex").write_text(tex, encoding="utf-8")
        pieces.append(tex)

    # Per-device table for the best combined-device configuration.
    per_dev = results_df[results_df["scope"] == "per_device"].copy()
    if not per_dev.empty and "combined_device_group_split" in set(overall["experiment"]):
        comb_best = overall_sorted[overall_sorted["experiment"] == "combined_device_group_split"].head(1)
        if not comb_best.empty:
            r = comb_best.iloc[0]
            mask = (
                (per_dev["experiment"] == r["experiment"]) &
                (per_dev["representation"] == r["representation"]) &
                (per_dev["classifier"] == r["classifier"])
            )
            dev_rows = per_dev[mask].copy()
            if not dev_rows.empty:
                cols = ["device_scope", "n_test", "support_normal", "support_apnea", "roc_auc", "balanced_accuracy", "recall_sensitivity", "specificity", "macro_f1", "mcc"]
                dev_ltx = dev_rows[[c for c in cols if c in dev_rows.columns]].rename(columns={
                    "device_scope": "Device", "n_test": "N", "support_normal": "Normal", "support_apnea": "Apnea",
                    "roc_auc": "AUC", "balanced_accuracy": "Bal. Acc.", "recall_sensitivity": "Sensitivity",
                    "specificity": "Specificity", "macro_f1": "Macro-F1", "mcc": "MCC",
                })
                tex = wrap_latex_table(
                    table_to_latex_tabular(dev_ltx, metric_formats),
                    "Per-device breakdown for the best combined-device model.",
                    "tab:combined_training_per_device_breakdown",
                    note=f"Model: {r['representation']} + {r['classifier']}.",
                    table_star=False,
                )
                (latex_dir / "table_combined_training_per_device.tex").write_text(tex, encoding="utf-8")
                pieces.append(tex)

    # One master file that can be pasted into the paper.
    (latex_dir / "journal_results_tables.tex").write_text("\n\n".join(pieces), encoding="utf-8")
    print(f"[LATEX] Journal-style LaTeX tables written to: {latex_dir}")

# -----------------------------------------------------------------------------
# Main benchmark loop
# -----------------------------------------------------------------------------


def save_dataset_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    df.to_csv(output_dir / "dataset_manifest.csv", index=False)
    summary = df.groupby(["device", "label"]).agg(
        samples=("file_path", "count"),
        patients=("patient_id", "nunique"),
    ).reset_index()
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    print("\n[DATASET SUMMARY]")
    print(summary)


def evaluate_scope(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    result_base: Dict,
    scope: str,
    device_scope: str,
) -> Dict:
    metrics = binary_metrics(y_true, y_pred, y_score)
    row = dict(result_base)
    row.update(metrics)
    row["scope"] = scope
    row["device_scope"] = device_scope
    return row


def run_benchmark(cfg: Dict) -> None:
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(Path(expand_path(cfg.get("output_dir", "./device_robust_outputs"))))
    ensure_dir(output_dir / "models")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "figures" / "roc")
    ensure_dir(output_dir / "figures" / "pr")
    ensure_dir(output_dir / "figures" / "confusion")
    ensure_dir(output_dir / "figures" / "calibration")
    ensure_dir(output_dir / "figures" / "combined_roc")
    ensure_dir(output_dir / "figures" / "tsne")

    json_dump(cfg, output_dir / "run_config_resolved.json")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("hf", {}).get("use_gpu", True) else "cpu")
    print(f"[DEVICE] Compute device: {device}")

    df = scan_all_devices(cfg)
    save_dataset_summaries(df, output_dir)

    representations, representation_deps = expand_representation_dependencies(cfg)
    print(f"[REPRESENTATIONS] Selected {len(representations)} representations")

    feature_matrices = extract_all_required_features(df, cfg, output_dir, device, representations, representation_deps)

    rep_dims = representation_dimensions(feature_matrices, {r: representation_deps[r] for r in representations})
    pd.DataFrame([
        {"representation": r, "dependencies": "+".join(representation_deps[r]), "dimension": rep_dims.get(r, np.nan)}
        for r in representations
    ]).to_csv(output_dir / "representation_manifest.csv", index=False)

    experiments = cfg["run"].get("experiments", [])
    classifier_names = cfg["run"].get("classifiers", [])
    report_by_device = bool(cfg.get("evaluation", {}).get("report_by_device", True))
    tsne_enabled = bool(cfg.get("plots", {}).get("tsne", {}).get("enabled", False))

    all_results = []
    all_roc_items_by_exp: Dict[str, List[Dict]] = {}

    for exp_name in experiments:
        print("\n" + "=" * 90)
        print(f"[EXPERIMENT] {exp_name}")
        print("=" * 90)
        train_idx, test_idx, exp_meta = build_experiment(df, exp_name, cfg)
        train_df = df[df["row_id"].isin(train_idx)].copy()
        test_df = df[df["row_id"].isin(test_idx)].copy()
        y_train = df.loc[train_idx, "label"].to_numpy()
        y_test = df.loc[test_idx, "label"].to_numpy()

        print(f"Train samples={len(train_idx)} | Test samples={len(test_idx)}")
        print("Train distribution:")
        print(train_df.groupby(["device", "label"]).size().rename("count"))
        print("Test distribution:")
        print(test_df.groupby(["device", "label"]).size().rename("count"))

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[WARNING] Skipping {exp_name}: train or test does not contain both classes.")
            continue

        roc_items = []

        for rep in representations:
            X_train = representation_matrix(feature_matrices, rep, train_idx, representation_deps)
            X_test = representation_matrix(feature_matrices, rep, test_idx, representation_deps)

            print(f"\n[REPRESENTATION] {rep}: X_train={X_train.shape}, X_test={X_test.shape}")
            plot_tsne_if_enabled(
                X_train,
                y_train,
                output_dir / "figures" / "tsne" / f"tsne_{safe_name(exp_name)}__{safe_name(rep)}.png",
                f"t-SNE: {exp_name} | {rep}",
                tsne_enabled,
            )

            for clf_name in classifier_names:
                print(f"  -> Training {clf_name}")
                try:
                    model = make_classifier(clf_name, cfg)
                    model = maybe_set_xgb_class_balance(model, clf_name, y_train)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_score, is_proba = get_scores(model, X_test)

                    base = {
                        "experiment": exp_name,
                        "representation": rep,
                        "classifier": clf_name,
                        "train_samples": int(len(train_idx)),
                        "test_samples": int(len(test_idx)),
                        **exp_meta,
                    }
                    overall = evaluate_scope(y_test, y_pred, y_score, base, "overall", "all")
                    all_results.append(overall)

                    print(
                        f"     AUC={overall['roc_auc']:.4f} | BalAcc={overall['balanced_accuracy']:.4f} | "
                        f"F1={overall['f1']:.4f} | Sens={overall['recall_sensitivity']:.4f} | Spec={overall['specificity']:.4f}"
                    )

                    pred_df = test_df[["row_id", "file_path", "file_name", "device", "patient_id", "label"]].copy()
                    pred_df["prediction"] = y_pred
                    pred_df["score_apnea"] = y_score
                    pred_df["is_correct"] = pred_df["label"].to_numpy() == y_pred
                    pred_path = output_dir / "predictions" / f"{safe_name(exp_name)}__{safe_name(rep)}__{safe_name(clf_name)}.csv"
                    pred_df.sort_values(["is_correct", "device", "patient_id"]).to_csv(pred_path, index=False)

                    model_dir = ensure_dir(output_dir / "models" / safe_name(exp_name))
                    model_path = model_dir / f"{safe_name(rep)}__{safe_name(clf_name)}.joblib"
                    joblib.dump({
                        "model": model,
                        "representation": rep,
                        "classifier": clf_name,
                        "experiment": exp_name,
                        "dependencies": representation_deps[rep],
                        "config": cfg,
                    }, model_path)

                    stem = f"{safe_name(exp_name)}__{safe_name(rep)}__{safe_name(clf_name)}"
                    plot_confusion(y_test, y_pred, output_dir / "figures" / "confusion" / f"confusion_{stem}.png", stem)
                    plot_roc(y_test, y_score, output_dir / "figures" / "roc" / f"roc_{stem}.png", stem)
                    plot_pr(y_test, y_score, output_dir / "figures" / "pr" / f"pr_{stem}.png", stem)
                    if is_proba:
                        plot_calibration(y_test, y_score, output_dir / "figures" / "calibration" / f"calibration_{stem}.png", stem)

                    roc_items.append({
                        "representation": rep,
                        "classifier": clf_name,
                        "y_true": y_test.copy(),
                        "y_score": y_score.copy(),
                    })

                    if report_by_device:
                        test_devices = test_df["device"].to_numpy()
                        for dev in sorted(np.unique(test_devices)):
                            mask = test_devices == dev
                            if mask.sum() == 0:
                                continue
                            row = evaluate_scope(
                                y_test[mask], y_pred[mask], y_score[mask], base,
                                scope="per_device", device_scope=str(dev)
                            )
                            all_results.append(row)

                except Exception as e:
                    print(f"[ERROR] Failed {exp_name} | {rep} | {clf_name}: {e}")

        all_roc_items_by_exp[exp_name] = roc_items
        plot_combined_roc(
            roc_items,
            output_dir / "figures" / "combined_roc" / f"combined_roc_{safe_name(exp_name)}.png",
            f"Combined ROC: {exp_name}",
        )

        # Persist results after each experiment so a long Kaggle run still keeps partial results.
        pd.DataFrame(all_results).to_csv(output_dir / "all_results_partial.csv", index=False)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    compared_df = save_final_comparison_csv(results_df, output_dir)
    write_latex_tables(compared_df, output_dir)

    if not results_df.empty:
        overall = results_df[results_df["scope"] == "overall"].copy()
        overall.sort_values(["experiment", "roc_auc", "macro_f1"], ascending=[True, False, False]).to_csv(
            output_dir / "overall_results_sorted.csv", index=False
        )
        per_device = results_df[results_df["scope"] == "per_device"].copy()
        per_device.to_csv(output_dir / "per_device_results.csv", index=False)
        plot_metric_summary(results_df, output_dir)

    print("\n[DONE] Outputs saved in:", output_dir)
    print("  - all_results.csv")
    print("  - final_comparison_all_results.csv")
    print("  - final_overall_comparison_sorted.csv")
    print("  - best_model_per_protocol.csv")
    print("  - overall_results_sorted.csv")
    print("  - per_device_results.csv")
    print("  - predictions/*.csv")
    print("  - models/<experiment>/*.joblib")
    print("  - figures/roc, pr, confusion, calibration, combined_roc, summary")
    print("  - latex_tables/journal_results_tables.tex")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError("Empty config file")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
