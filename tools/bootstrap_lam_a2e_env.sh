#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
ENV_NAME="${ENV_NAME:-lam-a2e310}"
REPO_ROOT="${REPO_ROOT:-${WORKSPACE_ROOT}/third_party/LAM_Audio2Expression}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
cd "${REPO_ROOT}"

python --version
sh ./scripts/install/install_cu121.sh

# Official cu121 install is not sufficient on this workstation's Blackwell GPU.
set +e
python - <<'PY'
import sys
import torch

def cuda_probe() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "cuda unavailable"
    try:
        x = torch.randn(2, 2, device="cuda")
        _ = x @ x
        return True, "ok"
    except Exception as exc:  # pragma: no cover - runtime probe
        return False, str(exc)

ok, message = cuda_probe()
print(f"cuda_probe={ok} message={message}")
if not ok and "no kernel image is available" in message:
    sys.exit(42)
PY
probe_exit=$?
set -e

if [[ "${probe_exit}" == "42" ]]; then
  python -m pip install --upgrade \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
fi

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2, 2, device="cuda")
    y = x @ x
    print("cuda_op_ok", y.cpu().tolist())
    print("device_name", torch.cuda.get_device_name(0))
PY

python inference.py --help >/dev/null
echo "LAM environment ready: ${ENV_NAME}"
