#!/bin/bash
#SBATCH --job-name=sq_h100_preflight
#SBATCH --partition=gpu-H100
#SBATCH --qos=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.out
#SBATCH --error=/userhome/phd/h.sharma/CODS-paper/logs/slurm_%x_%j.err

set -euo pipefail

PROJECT="/userhome/phd/h.sharma/CODS-paper"
DATASET_ROOT="/userhome/phd/h.sharma/Sleep quad Net/Data_v5_extracted/Data"
cd "$PROJECT"

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project=$PROJECT"
echo "dataset=$DATASET_ROOT"
test -d "$DATASET_ROOT"
test -f "$DATASET_ROOT/01/01_phone.wav"

command -v nvidia-smi
nvidia-smi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader

command -v python3
python3 --version
python3 - <<'PY'
import importlib.util
import platform

print("python=", platform.python_version())
for package in ("torch", "numpy", "venv"):
    print(f"{package}_available=", importlib.util.find_spec(package) is not None)

if importlib.util.find_spec("torch") is not None:
    import torch

    print("torch=", torch.__version__)
    print("torch_cuda=", torch.version.cuda)
    print("cuda_available=", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu=", torch.cuda.get_device_name(0))
        print("vram_bytes=", torch.cuda.get_device_properties(0).total_memory)
PY

echo "finished=$(date --iso-8601=seconds)"
