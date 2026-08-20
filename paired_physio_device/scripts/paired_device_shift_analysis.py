#!/usr/bin/env python3
"""Stage 3 (Section G): paired same-event R/S acoustic device-shift analysis.
Uses EXACT paired recordings (same event, same subject, same night) --
scientifically stronger than a population-level marginal comparison, since
subject/session/environment are held fixed by construction and only device
identity varies. CPU-only signal analysis; login-node safe for this sample
size (deterministic stratified sample of real pairs, not the full 10,325)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import librosa
from scipy import signal as sps
from scipy.stats import mannwhitneyu, wilcoxon

PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
sys.path.insert(0, str(PROJECT / "src"))

from sleep_quadnet.io import load_yaml, read_csv_rows, load_manifest_window  # noqa: E402

OUT_DIR = PROJECT / "paired_physio_device" / "results" / "physiology"
RNG = np.random.default_rng(42)
N_PER_STRATUM = 150  # -> up to 450 pairs across {normal, hypo, osa}, matches the
                       # project's established stratified-sample-size convention


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n_gt = np.sum(a[:, None] > b[None, :])
    n_lt = np.sum(a[:, None] < b[None, :])
    return (n_gt - n_lt) / (len(a) * len(b))


def spectral_stats(y, sr):
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    return dict(rms=rms, spectral_centroid=centroid, spectral_bandwidth=bandwidth,
                spectral_rolloff=rolloff, zcr=zcr, spectral_flatness=flatness)


def paired_measures(audio_R_raw, audio_S_raw, audio_R_norm, audio_S_norm, sr):
    n = min(len(audio_R_norm), len(audio_S_norm))
    r, s = audio_R_norm[:n], audio_S_norm[:n]

    # pre/post-normalization RMS (Section G items 1-2)
    pre_rms_R = float(np.sqrt(np.mean(audio_R_raw.astype(np.float64) ** 2)))
    pre_rms_S = float(np.sqrt(np.mean(audio_S_raw.astype(np.float64) ** 2)))
    post_rms_R = float(np.sqrt(np.mean(r.astype(np.float64) ** 2)))
    post_rms_S = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))

    # PSD + frequency-wise paired difference + log-spectral distance
    f, psd_r = sps.welch(r, fs=sr, nperseg=min(1024, n))
    _, psd_s = sps.welch(s, fs=sr, nperseg=min(1024, n))
    eps = 1e-12
    log_diff = 10 * np.log10(psd_r + eps) - 10 * np.log10(psd_s + eps)
    log_spectral_distance = float(np.sqrt(np.mean(log_diff ** 2)))

    # cross-correlation lag / residual alignment error (Section G item 9-10)
    if n > 1:
        corr = sps.correlate(r - r.mean(), s - s.mean(), mode="full")
        lags = sps.correlation_lags(n, n, mode="full")
        peak_lag = int(lags[np.argmax(corr)])
        peak_lag_ms = float(peak_lag / sr * 1000)
        residual_norm_corr = float(np.max(corr) / (np.linalg.norm(r) * np.linalg.norm(s) + eps))
    else:
        peak_lag_ms, residual_norm_corr = None, None

    # magnitude-squared coherence (Section G item 11)
    try:
        f_coh, coh = sps.coherence(r, s, fs=sr, nperseg=min(1024, n))
        mean_coherence = float(np.mean(coh))
    except Exception:
        mean_coherence = None

    stats_r = spectral_stats(r, sr)
    stats_s = spectral_stats(s, sr)

    return {
        "pre_norm_rms_R": pre_rms_R, "pre_norm_rms_S": pre_rms_S,
        "post_norm_rms_R": post_rms_R, "post_norm_rms_S": post_rms_S,
        "log_spectral_distance": log_spectral_distance,
        "peak_cross_corr_lag_ms": peak_lag_ms,
        "peak_normalized_cross_correlation": residual_norm_corr,
        "mean_magnitude_squared_coherence": mean_coherence,
        **{f"{k}_R": v for k, v in stats_r.items()},
        **{f"{k}_S": v for k, v in stats_s.items()},
    }


def main():
    config = load_yaml(PROJECT / "configs" / "base.yaml")
    manifest = read_csv_rows(Path(config["metadata"]["manifest"]))

    from collections import defaultdict
    by_pair = defaultdict(dict)
    for row in manifest:
        by_pair[row["paired_positive_id"]][row["device"]] = row
    complete_pairs = {pid: devs for pid, devs in by_pair.items() if set(devs.keys()) == {"R", "S"}}

    by_type = defaultdict(list)
    for pid, devs in complete_pairs.items():
        by_type[devs["R"]["event_type"]].append(pid)

    sampled_pids = []
    for event_type, pids in by_type.items():
        pids = sorted(pids)
        idx = RNG.choice(len(pids), size=min(N_PER_STRATUM, len(pids)), replace=False)
        sampled_pids.extend([pids[i] for i in idx])

    print(f"sampling {len(sampled_pids)} pairs across strata: "
          f"{ {k: len(v) for k, v in by_type.items()} }")

    rows = []
    n_errors = 0
    for pid in sampled_pids:
        devs = complete_pairs[pid]
        r_row, s_row = devs["R"], devs["S"]
        try:
            audio_R_raw, sr = load_manifest_window(r_row, config, preprocessing="none")
            audio_S_raw, sr2 = load_manifest_window(s_row, config, preprocessing="none")
            audio_R_norm, _ = load_manifest_window(r_row, config, preprocessing="peak")
            audio_S_norm, _ = load_manifest_window(s_row, config, preprocessing="peak")
            assert sr == sr2
            m = paired_measures(audio_R_raw, audio_S_raw, audio_R_norm, audio_S_norm, sr)
            m.update({
                "paired_event_id": pid, "subject_id": r_row["subject_id"],
                "event_type": r_row["event_type"], "sleep_stage": r_row["sleep_stage"],
                "label": r_row["label"],
            })
            rows.append(m)
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  [error] pair {pid}: {e}")

    print(f"successfully processed {len(rows)}/{len(sampled_pids)} pairs ({n_errors} errors)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = OUT_DIR / "paired_device_shift_measures.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {csv_path}")

    # subject-level aggregation for inferential claims (Section G: "do not
    # base major p-value claims on treating correlated windows as independent")
    import statistics as st
    metrics_for_test = [
        "post_norm_rms", "spectral_centroid", "spectral_bandwidth",
        "spectral_rolloff", "zcr", "spectral_flatness",
    ]
    summary = {"n_pairs_sampled": len(sampled_pids), "n_pairs_processed": len(rows),
               "n_errors": n_errors, "window_level": {}, "subject_level_paired_test": {}}

    for metric in metrics_for_test:
        r_key, s_key = f"{metric}_R", f"{metric}_S"
        if metric == "post_norm_rms":
            r_vals = np.array([row["post_norm_rms_R"] for row in rows])
            s_vals = np.array([row["post_norm_rms_S"] for row in rows])
        else:
            r_vals = np.array([row[r_key] for row in rows])
            s_vals = np.array([row[s_key] for row in rows])
        u_stat, p_window = mannwhitneyu(r_vals, s_vals, alternative="two-sided")
        delta = cliffs_delta(r_vals, s_vals)
        summary["window_level"][metric] = {
            "mean_R": float(np.mean(r_vals)), "mean_S": float(np.mean(s_vals)),
            "mannwhitney_p": float(p_window), "cliffs_delta": float(delta),
            "note": "window-level (pair-level), NOT subject-level -- see subject_level_paired_test for the inferential claim",
        }

        # subject-level aggregation: mean paired difference per subject, then
        # a paired (Wilcoxon signed-rank) test across subject-level means
        by_subj = defaultdict(list)
        for row in rows:
            diff = (row[r_key] if metric != "post_norm_rms" else row["post_norm_rms_R"]) - \
                   (row[s_key] if metric != "post_norm_rms" else row["post_norm_rms_S"])
            by_subj[row["subject_id"]].append(diff)
        subj_means = {sid: float(np.mean(v)) for sid, v in by_subj.items() if len(v) >= 1}
        vals = np.array(list(subj_means.values()))
        n_subj = len(vals)
        if n_subj >= 5 and not np.allclose(vals, 0):
            try:
                w_stat, p_subj = wilcoxon(vals)
            except ValueError:
                p_subj = None
        else:
            p_subj = None
        summary["subject_level_paired_test"][metric] = {
            "n_subjects": n_subj,
            "mean_paired_diff_R_minus_S": float(np.mean(vals)) if n_subj else None,
            "wilcoxon_p_subject_level": float(p_subj) if p_subj is not None else None,
        }

    with open(OUT_DIR / "paired_device_shift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
