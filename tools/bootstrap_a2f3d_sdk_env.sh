#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
ENV_NAME="${ENV_NAME:-a2f3d-sdk310}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$HOME/.local/nvidia/a2f3d-sdk}"
SDK_ROOT="${SDK_ROOT:-$WORKSPACE_ROOT/third_party/Audio2Face-3D-SDK}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install "numpy<1.28" soundfile pyyaml grpcio protobuf imageio-ffmpeg "opencv-python-headless<4.9"

if [[ -f "${SDK_ROOT}/deps/requirements.txt" ]]; then
  python -m pip install -r "${SDK_ROOT}/deps/requirements.txt"
  python -m pip install "numpy<1.28" "opencv-python-headless<4.9" imageio-ffmpeg
fi

ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/a2f3d-sdk-env.sh" <<EOF
export A2F3D_TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}"
export CUDA_HOME="${TOOLCHAIN_ROOT}/cuda-12.9"
export CUDA_PATH="${TOOLCHAIN_ROOT}/cuda-12.9"
export TENSORRT_ROOT_DIR="${TOOLCHAIN_ROOT}/tensorrt-10.13"
export _OLD_A2F3D_PATH="\$PATH"
export _OLD_A2F3D_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export PATH="${TOOLCHAIN_ROOT}/cuda-12.9/bin:${TOOLCHAIN_ROOT}/tensorrt-10.13/bin:\$PATH"
export LD_LIBRARY_PATH="${TOOLCHAIN_ROOT}/cuda-12.9/lib64:${TOOLCHAIN_ROOT}/tensorrt-10.13/lib:\${LD_LIBRARY_PATH:-}"
EOF

cat > "${DEACTIVATE_DIR}/a2f3d-sdk-env.sh" <<'EOF'
export PATH="${_OLD_A2F3D_PATH:-$PATH}"
export LD_LIBRARY_PATH="${_OLD_A2F3D_LD_LIBRARY_PATH:-}"
unset _OLD_A2F3D_PATH
unset _OLD_A2F3D_LD_LIBRARY_PATH
unset A2F3D_TOOLCHAIN_ROOT
unset CUDA_HOME
unset CUDA_PATH
unset TENSORRT_ROOT_DIR
EOF

source "${ACTIVATE_DIR}/a2f3d-sdk-env.sh"
python "${WORKSPACE_ROOT}/tools/verify_a2f3d_sdk_env.py" --toolchain-root "${TOOLCHAIN_ROOT}"
echo "A2F-3D SDK environment ready: ${ENV_NAME}"
