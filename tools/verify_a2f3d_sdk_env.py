#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that the A2F-3D SDK env resolves the side-by-side CUDA/TensorRT toolchain.")
    parser.add_argument(
        "--toolchain-root",
        type=Path,
        default=Path(os.environ.get("A2F3D_TOOLCHAIN_ROOT", Path.home() / ".local" / "nvidia" / "a2f3d-sdk")),
    )
    return parser.parse_args(argv)


def _parse_cuda_version(nvcc_output: str) -> tuple[int, int]:
    match = re.search(r"release\s+(\d+)\.(\d+)", nvcc_output)
    if not match:
        raise RuntimeError("Unable to parse nvcc version output")
    return int(match.group(1)), int(match.group(2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cuda_root = args.toolchain_root / "cuda-12.9"
    tensorrt_root = args.toolchain_root / "tensorrt-10.13"
    nvcc = cuda_root / "bin" / "nvcc"

    if not nvcc.exists():
        raise SystemExit(f"Missing CUDA toolkit under {cuda_root}")
    if not tensorrt_root.exists():
        raise SystemExit(f"Missing TensorRT toolkit under {tensorrt_root}")

    completed = subprocess.run([str(nvcc), "--version"], check=True, capture_output=True, text=True)
    major, minor = _parse_cuda_version(completed.stdout + completed.stderr)
    if major >= 13:
        raise SystemExit(f"Unsupported CUDA detected from toolchain nvcc: {major}.{minor}")
    if (major, minor) < (12, 8):
        raise SystemExit(f"CUDA is too old for A2F-3D SDK: {major}.{minor}")

    trt_libs = list(tensorrt_root.rglob("libnvinfer.so*"))
    if not trt_libs:
        raise SystemExit(f"TensorRT libraries were not found under {tensorrt_root}")

    path_value = os.environ.get("PATH", "")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    cuda_home = os.environ.get("CUDA_HOME", "")
    if str(cuda_root / "bin") not in path_value:
        raise SystemExit("PATH is not preferring the side-by-side CUDA 12.9 bin directory")
    if str(cuda_root) not in cuda_home:
        raise SystemExit("CUDA_HOME is not pointing at the side-by-side CUDA 12.9 toolkit")
    if str(cuda_root / "lib64") not in ld_library_path:
        raise SystemExit("LD_LIBRARY_PATH is not preferring the side-by-side CUDA 12.9 libraries")
    if str(tensorrt_root / "lib") not in ld_library_path and str(tensorrt_root / "targets") not in ld_library_path:
        raise SystemExit("LD_LIBRARY_PATH is not exposing TensorRT 10.13 libraries")

    print(f"[ok] toolchain_root={args.toolchain_root}")
    print(f"[ok] cuda_version={major}.{minor}")
    print(f"[ok] tensorrt_libs={len(trt_libs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
