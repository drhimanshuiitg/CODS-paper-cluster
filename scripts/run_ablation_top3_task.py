#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_REPRESENTATIONS = {"hubert", "wavlm", "wav2vec2", "data2vec_audio", "data2vec_spectrogram", "data2vec_fusion", "full_fusion"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=range(3), required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    selection = json.loads((PROJECT_ROOT / "results" / "P0_ablation" / "selection.json").read_text(encoding="utf-8"))
    representation = selection["top3_representations"][args.rank]
    if representation in BASE_REPRESENTATIONS:
        print(json.dumps({"status": "reused_from_main_benchmark", "representation": representation, "classifier": args.classifier, "protocol": args.protocol}))
        return
    command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(PROJECT_ROOT / "scripts" / "run_main_benchmark.py"),
        "--results-root", str(PROJECT_ROOT / "results" / "P0_ablation"), "--representations", representation,
        "--classifiers", args.classifier, "--protocols", args.protocol,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
