#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
ENV_NAME="${ENV_NAME:-pantomatrix-emage39}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda create -n "${ENV_NAME}" python=3.9 -y
conda activate "${ENV_NAME}"
conda install -c conda-forge ffmpeg git-lfs cmake ninja libstdcxx-ng -y

python -m pip install --upgrade pip==24.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install \
  librosa==0.10.2.post1 \
  transformers==4.30.2 \
  huggingface_hub==0.36.2 \
  omegaconf==2.2.3 \
  smplx==0.1.28 \
  scipy==1.11.4 \
  scikit-learn==1.3.2 \
  tqdm \
  yacs \
  easydict \
  json_tricks \
  tensorboardx \
  filterpy \
  cython \
  chumpy==0.70.0 \
  einops \
  joblib \
  ConfigArgParse \
  eval_type_backport \
  soundfile \
  soxr \
  numba \
  viser

echo "Environment ready: ${ENV_NAME}"
