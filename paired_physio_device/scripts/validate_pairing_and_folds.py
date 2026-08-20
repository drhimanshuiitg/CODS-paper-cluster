#!/usr/bin/env python3
"""Stage 2: real validation of the R/S pairing structure and subject-fold
assignment before any model touches this data. Login-node safe (CSV parsing
only). Writes paired_physio_device/audit/pairing_fold_validation_report.json
and fails loudly (non-zero exit) on any violation of the master prompt's
non-negotiable principles 1-2 (subject-disjoint folds; same-subject R/S pairs
share a fold)."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path("/home/pkdas/IEEE_healthcomm_workshop")
OUT = PROJECT / "paired_physio_device" / "audit" / "pairing_fold_validation_report.json"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    manifest = load_csv(PROJECT / "metadata" / "dataset_manifest_aligned.csv")
    folds = load_csv(PROJECT / "metadata" / "subject_folds_5cv_aligned.csv")

    report = {"checks": [], "violations": [], "pass": True}

    def check(name, condition, detail):
        report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            report["violations"].append({"name": name, "detail": detail})
            report["pass"] = False

    # --- 1. every paired_positive_id has exactly one R and one S row -------
    by_pair = defaultdict(dict)
    for row in manifest:
        by_pair[row["paired_positive_id"]][row["device"]] = row
    n_pairs_total = len(by_pair)
    n_pairs_complete = sum(1 for devs in by_pair.values() if set(devs.keys()) == {"R", "S"})
    n_pairs_incomplete = n_pairs_total - n_pairs_complete
    check(
        "every_paired_positive_id_has_exactly_R_and_S",
        n_pairs_incomplete == 0,
        f"{n_pairs_complete}/{n_pairs_total} paired_positive_id groups have both R and S; "
        f"{n_pairs_incomplete} incomplete (single-device only -- expected for the subset of "
        f"windows near a subject's alignment-segment boundary where only one device's segment "
        f"was usable; not itself a leakage issue, but recorded here).",
    )

    # --- 2. both rows of a pair share the same subject_id -------------------
    mismatched = [
        pid for pid, devs in by_pair.items()
        if set(devs.keys()) == {"R", "S"} and devs["R"]["subject_id"] != devs["S"]["subject_id"]
    ]
    check(
        "paired_rows_share_subject_id",
        len(mismatched) == 0,
        f"{len(mismatched)} paired_positive_id groups have mismatched subject_id across R/S rows.",
    )

    # --- 3. subject -> fold assignment is a function (no subject in 2 folds) ---
    subj_to_folds = defaultdict(set)
    for row in folds:
        subj_to_folds[row["subject_id"]].add(row["fold"])
    multi_fold_subjects = {s: f for s, f in subj_to_folds.items() if len(f) > 1}
    check(
        "each_subject_assigned_to_exactly_one_fold",
        len(multi_fold_subjects) == 0,
        f"{len(multi_fold_subjects)} subjects appear in more than one fold: {multi_fold_subjects}",
    )

    # --- 4. fold-file subject set matches manifest subject set --------------
    manifest_subjects = set(row["subject_id"] for row in manifest)
    fold_subjects = set(subj_to_folds.keys())
    only_in_manifest = manifest_subjects - fold_subjects
    only_in_folds = fold_subjects - manifest_subjects
    check(
        "manifest_and_fold_subject_sets_match",
        not only_in_manifest and not only_in_folds,
        f"only_in_manifest={sorted(only_in_manifest)}, only_in_folds={sorted(only_in_folds)}",
    )

    # --- 5. same-event R/S pair never split across folds (principle 2) ------
    # (implied by checks 2+3 together, but verified directly and explicitly)
    subj_fold = {s: next(iter(fs)) for s, fs in subj_to_folds.items() if len(fs) == 1}
    split_pairs = []
    for pid, devs in by_pair.items():
        if set(devs.keys()) != {"R", "S"}:
            continue
        sid = devs["R"]["subject_id"]
        # trivially true since both rows share subject_id (check 2) and a
        # subject maps to one fold (check 3), but verified independently here
        r_fold = subj_fold.get(sid)
        s_fold = subj_fold.get(devs["S"]["subject_id"])
        if r_fold != s_fold:
            split_pairs.append(pid)
    check(
        "no_paired_event_split_across_folds",
        len(split_pairs) == 0,
        f"{len(split_pairs)} paired events would straddle two folds (none expected, given checks 2+3).",
    )

    # --- 6. label counts per fold (small-class-count sanity, Section B) -----
    label_by_fold = defaultdict(lambda: defaultdict(int))
    for row in manifest:
        f = subj_fold.get(row["subject_id"])
        label_by_fold[f][row["label"]] += 1
    report["label_counts_per_fold"] = {str(k): dict(v) for k, v in label_by_fold.items()}

    # --- 7. subjects per fold ------------------------------------------------
    subjects_per_fold = defaultdict(set)
    for s, f in subj_fold.items():
        subjects_per_fold[f].add(s)
    report["n_subjects_per_fold"] = {k: len(v) for k, v in subjects_per_fold.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "label_counts_per_fold"}, indent=2))
    if not report["pass"]:
        print("VALIDATION FAILED -- see violations above", file=sys.stderr)
        sys.exit(1)
    print("\nAll pairing/fold validation checks passed.")


if __name__ == "__main__":
    main()
