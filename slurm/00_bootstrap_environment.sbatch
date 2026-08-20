#!/bin/bash
#SBATCH --job-name=sq_env
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
PROJECT=/userhome/phd/h.sharma/CODS-paper
ENV_DIR="$PROJECT/.venv"
GPU_ENV_DIR="$PROJECT/.gpu_classifier_venv"
mkdir -p "$PROJECT/logs" "$PROJECT/environment" "$PROJECT/checkpoints/huggingface"
cd "$PROJECT"

bootstrap_venv() {
  local target="$1"
  if [[ ! -x "$target/bin/python" ]]; then
    python3 -m venv --without-pip "$target"
  fi
  if ! "$target/bin/python" -m pip --version >/dev/null 2>&1; then
    local get_pip="$TMPDIR/get-pip.py"
    if [[ ! -f "$get_pip" ]]; then
      curl --fail --location --retry 3 --output "$get_pip" https://bootstrap.pypa.io/get-pip.py
    fi
    "$target/bin/python" "$get_pip"
  fi
}

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python3 -m venv --without-pip "$ENV_DIR"
fi
bootstrap_venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV_DIR/bin/python" -m pip install -r requirements.txt
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip freeze > "$PROJECT/environment/pip_freeze_slurm_${SLURM_JOB_ID}.txt"

if [[ ! -x "$GPU_ENV_DIR/bin/python" ]]; then
  python3 -m venv --without-pip "$GPU_ENV_DIR"
fi
bootstrap_venv "$GPU_ENV_DIR"
"$GPU_ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$GPU_ENV_DIR/bin/python" -m pip install --extra-index-url https://pypi.nvidia.com cuml-cu12==26.8.0
"$GPU_ENV_DIR/bin/python" -m pip check
"$GPU_ENV_DIR/bin/python" -m pip freeze > "$PROJECT/environment/pip_freeze_cuml_slurm_${SLURM_JOB_ID}.txt"
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

"$GPU_ENV_DIR/bin/python" - <<'PY'
import cupy as cp
import cuml
import numpy as np
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC

if cp.cuda.runtime.getDeviceCount() < 1:
    raise SystemExit("RAPIDS environment cannot see the allocated H100")
print("cuml", cuml.__version__)
print("cupy_device", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
print("cupy_test_sum", float(cp.arange(1024, dtype=cp.float32).sum()))
x = np.asarray([[0.0, 0.0], [0.1, 0.2], [0.9, 0.8], [1.0, 1.0]], dtype=np.float32)
y = np.asarray([0, 0, 1, 1], dtype=np.int32)
rf = RandomForestClassifier(n_estimators=4, max_depth=3, class_weight="balanced", output_type="numpy")
rf.fit(x, y)
svc = SVC(C=1.0, gamma="scale", class_weight="balanced", probability=False, output_type="numpy")
svc.fit(x, y)
print("cuml_rf_smoke", rf.predict(x).tolist())
print("cuml_svc_smoke", svc.predict(x).tolist())
PY
