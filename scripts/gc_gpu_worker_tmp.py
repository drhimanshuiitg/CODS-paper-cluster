#!/usr/bin/env python3
"""One-time (repeatable) garbage collector for gpu_worker_tmp/<uuid> leftovers.

GPUSubprocessEstimator (src/sleep_quadnet/evaluation.py) writes each cuML
fit's model to gpu_worker_tmp/<uuid>/model.pkl. Tuning-candidate estimators
are now cleaned up immediately by _cleanup_estimator() (see
select_estimator()), but every directory created before that fix landed, and
any directory whose owning cache entry has since been superseded (e.g. by
the gpu_tag fix forcing PCA/CORAL to re-key), is orphaned and safe to
reclaim.

This scans every classifier.joblib / pipeline.joblib under
checkpoints/{downstream_fit_cache,pca_cache,coral_cache}, properly
joblib.load()s each one (NOT a raw byte/regex scan -- pathlib.Path pickles
its path components as separate string opcodes, not a contiguous path
string, so a regex over the raw pickle bytes will never match and would
misidentify every live reference as unreferenced), and collects every
GPUSubprocessEstimator._work_dir it finds (unwrapping sklearn Pipelines and
the {"classifier": estimator} dict shape used by pca_cache/coral_cache).
Any gpu_worker_tmp/<uuid> directory NOT in that referenced set is orphaned.

Defaults to a dry run (prints candidates + reclaimable bytes, deletes
nothing). Pass --execute to actually delete. Run again any time (e.g. after
the Issue-A gpu_tag rerun completes) to reclaim newly-orphaned entries.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib

from sleep_quadnet.evaluation import GPUSubprocessEstimator  # noqa: F401 -- must be importable for unpickling

_GPU_WORKER_TMP = Path("/scratch/pkdas/IEEE_healthcomm_workshop/gpu_worker_tmp")
_CACHE_ROOTS = [
    ("classifier.joblib", Path("/scratch/pkdas/IEEE_healthcomm_workshop/checkpoints/downstream_fit_cache")),
    ("pipeline.joblib", Path("/scratch/pkdas/IEEE_healthcomm_workshop/checkpoints/pca_cache")),
    ("pipeline.joblib", Path("/scratch/pkdas/IEEE_healthcomm_workshop/checkpoints/coral_cache")),
]


def _referenced_work_dirs(obj) -> list[Path]:
    """Find every GPUSubprocessEstimator._work_dir reachable from a loaded
    cache object: a raw estimator, an sklearn Pipeline wrapping one, or a
    {"classifier": estimator, ...} dict (pca_cache/coral_cache shape)."""
    candidates = []
    if isinstance(obj, dict) and "classifier" in obj:
        candidates.append(obj["classifier"])
    else:
        candidates.append(obj)
    found = []
    for candidate in candidates:
        target = candidate.steps[-1][1] if hasattr(candidate, "steps") else candidate
        work_dir = getattr(target, "_work_dir", None)
        if work_dir is not None:
            found.append(Path(work_dir))
    return found


def scan_referenced() -> set[Path]:
    referenced: set[Path] = set()
    scanned = 0
    failed = 0
    for filename, root in _CACHE_ROOTS:
        if not root.exists():
            continue
        for path in root.glob(f"*/{filename}"):
            scanned += 1
            try:
                obj = joblib.load(path)
            except Exception as error:  # noqa: BLE001 -- best-effort scan, report and continue
                failed += 1
                print(f"  WARN: failed to load {path}: {error}", file=sys.stderr)
                continue
            referenced.update(_referenced_work_dirs(obj))
    print(f"Scanned {scanned} cache entries ({failed} failed to load) across {[str(r) for _, r in _CACHE_ROOTS]}")
    return referenced


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually delete orphaned directories (default: dry run only)")
    args = parser.parse_args()

    if not _GPU_WORKER_TMP.exists():
        print(f"{_GPU_WORKER_TMP} does not exist -- nothing to do")
        return

    referenced = scan_referenced()
    all_dirs = sorted(p for p in _GPU_WORKER_TMP.iterdir() if p.is_dir())
    orphaned = [p for p in all_dirs if p not in referenced]

    total_size = sum(dir_size(p) for p in orphaned)
    print(f"gpu_worker_tmp: {len(all_dirs)} total dirs, {len(referenced)} referenced by a live cache entry, "
          f"{len(orphaned)} orphaned ({total_size / 1e9:.2f} GB reclaimable)")

    if not args.execute:
        print("Dry run only -- pass --execute to actually delete. Sample of orphaned dirs:")
        for p in orphaned[:10]:
            print(f"  {p}")
        if len(orphaned) > 10:
            print(f"  ... and {len(orphaned) - 10} more")
        return

    deleted = 0
    for p in orphaned:
        shutil.rmtree(p, ignore_errors=True)
        deleted += 1
    print(f"Deleted {deleted} orphaned directories, reclaimed ~{total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
