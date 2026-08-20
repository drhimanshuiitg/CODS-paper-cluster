#!/bin/bash
#SBATCH --job-name=sq_env
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
mkdir -p "$PROJECT/logs" "$PROJECT/environment" "$PROJECT/checkpoints/huggingface"
cd "$PROJECT"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV_DIR/bin/python" -m pip install -r requirements.txt
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip freeze > "$PROJECT/environment/pip_freeze_slurm_${SLURM_JOB_ID}.txt"
HF_HOME="$PROJECT/checkpoints/huggingface" "$ENV_DIR/bin/python" - <<'PY'
import json
import platform
from pathlib import Path
import sklearn
import torch
import transformers

record = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    "transformers": transformers.__version__,
    "scikit_learn": sklearn.__version__,
}
target = Path("environment") / "validation.json"
if target.exists():
    target = Path("environment") / f"validation_slurm_{__import__('os').environ['SLURM_JOB_ID']}.json"
target.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
if not record["cuda_available"]:
    raise SystemExit("CUDA is not available inside the requested GPU job")
print("cuda_test_sum", torch.randn(1024, device="cuda").sum().item())
PY
