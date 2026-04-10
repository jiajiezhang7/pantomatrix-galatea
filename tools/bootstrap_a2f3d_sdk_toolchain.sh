#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PREFIX="${PREFIX:-$HOME/.local/nvidia/a2f3d-sdk}"
CUDA_ROOT="${PREFIX}/cuda-12.9"
TENSORRT_ROOT="${PREFIX}/tensorrt-10.13"

usage() {
  cat <<'EOF'
Usage:
  tools/bootstrap_a2f3d_sdk_toolchain.sh

This installs a side-by-side CUDA 12.9 and TensorRT 10.13 prefix for A2F-3D SDK
without replacing the system CUDA 13.x toolkit.

Implementation notes:
- CUDA 12.9 is installed into a dedicated prefix through `conda create -p`.
- TensorRT 10.13 development packages are downloaded from the NVIDIA Ubuntu 24.04 repo
  with `apt download` and extracted into the same side-by-side prefix with `dpkg-deb -x`.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${PREFIX}"

source "${HOME}/anaconda3/etc/profile.d/conda.sh"

if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  conda create -y -p "${CUDA_ROOT}" -c nvidia cuda-toolkit=12.9.1
fi

if [[ ! -f "${TENSORRT_ROOT}/usr/include/x86_64-linux-gnu/NvInfer.h" ]]; then
  mkdir -p /tmp/a2f-trt-debs "${TENSORRT_ROOT}"
  pushd /tmp/a2f-trt-debs >/dev/null
  apt download \
    libnvinfer10=10.13.3.9-1+cuda12.9 \
    libnvinfer-bin=10.13.3.9-1+cuda12.9 \
    libnvinfer-dev=10.13.3.9-1+cuda12.9 \
    libnvinfer-headers-dev=10.13.3.9-1+cuda12.9 \
    libnvinfer-headers-plugin-dev=10.13.3.9-1+cuda12.9 \
    libnvinfer-plugin10=10.13.3.9-1+cuda12.9 \
    libnvinfer-plugin-dev=10.13.3.9-1+cuda12.9 \
    libnvonnxparsers10=10.13.3.9-1+cuda12.9 \
    libnvonnxparsers-dev=10.13.3.9-1+cuda12.9
  for deb in ./*.deb; do
    dpkg-deb -x "${deb}" "${TENSORRT_ROOT}"
  done
  popd >/dev/null
fi

ln -sfn "${CUDA_ROOT}/lib" "${CUDA_ROOT}/lib64"
ln -sfn "${TENSORRT_ROOT}/usr/include/x86_64-linux-gnu" "${TENSORRT_ROOT}/include"
ln -sfn "${TENSORRT_ROOT}/usr/lib/x86_64-linux-gnu" "${TENSORRT_ROOT}/lib"
mkdir -p "${TENSORRT_ROOT}/bin"
if [[ -x "${TENSORRT_ROOT}/usr/src/tensorrt/bin/trtexec" ]]; then
  ln -sfn "${TENSORRT_ROOT}/usr/src/tensorrt/bin/trtexec" "${TENSORRT_ROOT}/bin/trtexec"
fi

cat > "${PREFIX}/env.sh" <<EOF
export A2F3D_TOOLCHAIN_ROOT="${PREFIX}"
export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export TENSORRT_ROOT_DIR="${TENSORRT_ROOT}"
export PATH="${CUDA_ROOT}/bin:${TENSORRT_ROOT}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${TENSORRT_ROOT}/lib:\${LD_LIBRARY_PATH:-}"
EOF

source "${PREFIX}/env.sh"
python "${WORKSPACE_ROOT}/tools/verify_a2f3d_sdk_env.py" --toolchain-root "${PREFIX}"
