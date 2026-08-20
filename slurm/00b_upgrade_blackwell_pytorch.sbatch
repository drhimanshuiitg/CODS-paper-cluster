#!/bin/bash
#SBATCH --job-name=sq_torch_fix
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:mig24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.out
#SBATCH --error=/home/pkdas/IEEE_healthcomm_workshop/logs/slurm_%x_%j.err

set -euo pipefail
PROJECT=/home/pkdas/IEEE_healthcomm_workshop
ENV_DIR="$PROJECT/.venv"
cd "$PROJECT"
"$ENV_DIR/bin/python" -m pip install --upgrade torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip freeze > "$PROJECT/environment/pip_freeze_blackwell_slurm_${SLURM_JOB_ID}.txt"
"$ENV_DIR/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
value = torch.randn(1024, device="cuda").sum().item()
record = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "cuda_arch_list": torch.cuda.get_arch_list(),
    "cuda_compute_test_sum": value,
    "blackwell_sm120_supported": "sm_120" in torch.cuda.get_arch_list(),
}
if not record["blackwell_sm120_supported"]:
    raise SystemExit(f"Installed PyTorch still lacks sm_120: {record}")
target = Path("environment") / f"blackwell_validation_slurm_{os.environ['SLURM_JOB_ID']}.json"
target.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
PY
