#!/usr/bin/env python3
"""Stage 2: the exact paired R/S event dataloader (Section C6's positive-pair
source). Reuses the existing, trusted `load_manifest_window` audio-loading
path (`src/sleep_quadnet/io.py`) rather than reimplementing WAV
reading/resampling/stitching -- per the master prompt's instruction to
reproduce trusted infrastructure, not rebuild it.

Login-node safe to *construct* and to smoke-test on a handful of items
(pure CPU audio I/O). Full-dataset iteration for real training runs inside
a SLURM job, per GPU_INSTRUCTIONS.md."""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
sys.path.insert(0, str(PROJECT / "src"))

from sleep_quadnet.io import load_yaml, read_csv_rows, load_manifest_window  # noqa: E402


class PairedEventDataset:
    """One item = one same-event (R, S) pair, with label/event_type/subject_id.

    Only complete pairs (both R and S rows present for a paired_positive_id)
    are used -- confirmed 10,325/10,325 complete by
    `validate_pairing_and_folds.py` (0 incomplete in the current manifest).
    """

    def __init__(self, subjects: set[str], config: dict | None = None,
                 preprocessing: str = "peak", manifest_path: Path | None = None):
        self.config = config or load_yaml(PROJECT / "configs" / "base.yaml")
        self.preprocessing = preprocessing
        manifest_path = manifest_path or Path(self.config["metadata"]["manifest"])
        rows = read_csv_rows(manifest_path)
        rows = [r for r in rows if r["subject_id"] in subjects]

        by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            by_pair[row["paired_positive_id"]][row["device"]] = row

        self.pairs: list[dict] = []
        for pid, devs in by_pair.items():
            if set(devs.keys()) != {"R", "S"}:
                continue
            r, s = devs["R"], devs["S"]
            assert r["subject_id"] == s["subject_id"], (
                f"pair {pid} spans two subjects -- should be impossible; "
                f"validate_pairing_and_folds.py did not catch this, investigate immediately"
            )
            self.pairs.append({
                "paired_event_id": pid,
                "subject_id": r["subject_id"],
                "label": int(r["label"]),
                "event_type": r["event_type"],
                "sleep_stage": r["sleep_stage"],
                "row_R": r,
                "row_S": s,
            })

        # In-memory decoded-audio cache. Without this, every __getitem__ call
        # re-reads and re-resamples (librosa polyphase, 8kHz->16kHz) from disk
        # on EVERY epoch for EVERY item -- pure redundant CPU work across all
        # 15 epochs, since load_manifest_window is a deterministic function of
        # (row, config, preprocessing). RAM was confirmed abundant (hundreds
        # of GB available, no per-account ceiling found) and this dataset's
        # full decoded size is a few GB at most -- an easy trade. Populate via
        # preload_all() BEFORE the DataLoader's first __iter__() call (which
        # is when persistent_workers fork): fork-based worker processes
        # inherit the already-populated cache through copy-on-write, so every
        # worker gets full-cache reads for free with no IPC or duplication
        # cost, rather than each worker building its own partial cache under
        # random per-epoch index reshuffling.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_uncached(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        item = self.pairs[idx]
        audio_R, sr_R = load_manifest_window(item["row_R"], self.config, self.preprocessing)
        audio_S, sr_S = load_manifest_window(item["row_S"], self.config, self.preprocessing)
        assert sr_R == sr_S, f"mismatched target sample rate: R={sr_R} S={sr_S}"
        return audio_R, audio_S, sr_R

    def preload_all(self, max_workers: int = 8) -> None:
        """Eagerly decode+resample every item once and cache it. Uses a
        thread pool, not a process pool: librosa/scipy/numpy's resampling and
        WAV-reading do their heavy per-sample work in C extensions that
        release the GIL, so threads give real parallelism here without the
        pickling/IPC cost of multiprocessing -- and critically, threads share
        the parent's memory directly, so the populated cache is immediately
        visible to a subsequent fork (a separate process pool's cache would
        die with the pool's worker processes instead)."""
        if len(self._cache) == len(self.pairs):
            return  # already fully cached (e.g. train/val/test share no
                     # indices, but a repeated call on the same instance should be a no-op)
        start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._load_uncached, i): i for i in range(len(self.pairs)) if i not in self._cache}
            for fut in as_completed(futures):
                i = futures[fut]
                self._cache[i] = fut.result()
        print(f"  preloaded {len(self.pairs)} pairs into memory in {time.time() - start:.1f}s "
              f"({max_workers} threads)", flush=True)

    def __getitem__(self, idx: int) -> dict:
        item = self.pairs[idx]
        if idx in self._cache:
            audio_R, audio_S, sr_R = self._cache[idx]
        else:
            audio_R, audio_S, sr_R = self._load_uncached(idx)
        return {
            "paired_event_id": item["paired_event_id"],
            "subject_id": item["subject_id"],
            "label": item["label"],
            "event_type": item["event_type"],
            "sleep_stage": item["sleep_stage"],
            "audio_R": audio_R,
            "audio_S": audio_S,
            "sample_rate": sr_R,
        }

    def subject_label_counts(self) -> dict[str, dict[int, int]]:
        counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for item in self.pairs:
            counts[item["subject_id"]][item["label"]] += 1
        return {k: dict(v) for k, v in counts.items()}


def subjects_for_fold(fold_path: Path, held_out_fold: str) -> tuple[set[str], set[str]]:
    """Returns (train_subjects, test_subjects) for a given held-out fold id,
    matching this project's existing fold-rotation convention."""
    rows = read_csv_rows(fold_path)
    test = {r["subject_id"] for r in rows if r["fold"] == held_out_fold}
    train = {r["subject_id"] for r in rows if r["fold"] != held_out_fold}
    return train, test


if __name__ == "__main__":
    # Smoke test: construct the dataset for fold 0's test subjects and load a
    # handful of real (R, S) pairs end to end, on the login node (CPU-only,
    # a few items -- explicitly allowed lightweight validation per
    # GPU_INSTRUCTIONS.md Section 1/4).
    config = load_yaml(PROJECT / "configs" / "base.yaml")
    _, test_subjects = subjects_for_fold(Path(config["metadata"]["subject_folds"]), "0")
    ds = PairedEventDataset(test_subjects, config=config)
    print(f"fold-0 test subjects: {sorted(test_subjects)}")
    print(f"n paired items: {len(ds)}")
    for i in [0, len(ds) // 2, len(ds) - 1]:
        item = ds[i]
        print(
            f"item {i}: subj={item['subject_id']} label={item['label']} "
            f"event_type={item['event_type']} sleep_stage={item['sleep_stage']} "
            f"audio_R.shape={item['audio_R'].shape} audio_S.shape={item['audio_S'].shape} "
            f"sr={item['sample_rate']} "
            f"R_finite={np.isfinite(item['audio_R']).all()} S_finite={np.isfinite(item['audio_S']).all()}"
        )
    print("Smoke test passed: paired dataset constructs and loads real audio end-to-end.")
