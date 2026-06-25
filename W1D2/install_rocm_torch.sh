#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# ROCm wheels for AMD GPU. PyTorch still reports the device as "cuda" when ROCm is active.
ROCM_TORCH_INDEX="${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.2}"
.venv/bin/pip install torch --index-url "$ROCM_TORCH_INDEX"
.venv/bin/pip install -r requirements.txt

.venv/bin/python - <<'PY'
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import subprocess
import torch

print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
print("HSA_OVERRIDE_GFX_VERSION:", os.environ.get("HSA_OVERRIDE_GFX_VERSION"))
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.arange(8, device="cuda")
    print("kernel_test:", (x + 1).detach().cpu().tolist())
else:
    print("AMD GPU not detected by PyTorch ROCm")
    print("")
    print("Diagnostics:")
    print("- groups:", subprocess.check_output(["groups"], text=True).strip())
    print("- /dev/kfd exists:", os.path.exists("/dev/kfd"))
    print("- /dev/dri exists:", os.path.exists("/dev/dri"))
    print("")
    print("If groups does not include render and video, run:")
    print("  sudo usermod -aG render,video $USER")
    print("Then log out/in or reboot before running this script again.")
PY
