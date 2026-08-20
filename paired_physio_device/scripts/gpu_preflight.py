#!/usr/bin/env python3
"""GPU preflight check for the PairPhysNet pipeline (Section P / Four-GPU skill).
Run only inside a SLURM job -- never on the login node."""
import json
import subprocess
import sys
from datetime import datetime, timezone

report = {"timestamp": datetime.now(timezone.utc).isoformat()}

try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
         "--format=csv"],
        capture_output=True, text=True, timeout=30,
    )
    report["nvidia_smi_stdout"] = smi.stdout
    report["nvidia_smi_returncode"] = smi.returncode
except Exception as e:
    report["nvidia_smi_error"] = str(e)

try:
    import torch
    report["torch_version"] = torch.__version__
    report["torch_cuda_version"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_device_count"] = torch.cuda.device_count()
    devices = []
    for i in range(torch.cuda.device_count()):
        try:
            x = torch.zeros(4, device=f"cuda:{i}")
            _ = (x + 1).sum().item()
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i, "name": props.name,
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "small_tensor_alloc_ok": True,
            })
        except Exception as e:
            devices.append({"index": i, "error": str(e)})
    report["devices"] = devices
except Exception as e:
    report["torch_error"] = str(e)

# Required GPU library import checks
libs = {}
for modname in ["torch", "torchaudio", "transformers", "xgboost"]:
    try:
        __import__(modname)
        libs[modname] = "ok"
    except Exception as e:
        libs[modname] = f"FAIL: {e}"
try:
    import cuml  # noqa
    libs["cuml"] = "ok"
except Exception as e:
    libs["cuml"] = f"not importable in this venv: {e}"
report["library_imports"] = libs

print(json.dumps(report, indent=2))
with open("paired_physio_device/logs/gpu_preflight_report.json", "w") as f:
    json.dump(report, f, indent=2)

if not report.get("cuda_available", False):
    print("HARD FAIL: CUDA not available in this job's environment.", file=sys.stderr)
    sys.exit(1)
