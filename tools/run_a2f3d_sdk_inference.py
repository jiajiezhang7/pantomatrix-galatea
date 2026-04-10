#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local NVIDIA Audio2Face-3D SDK wrapper command.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, default=None)
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_runtime_env(config: dict[str, Any], sdk_root: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in config.get("env", {}).items():
        env[str(key)] = str(value)
    if sdk_root is not None:
        env.setdefault("A2F3D_SDK_ROOT", str(sdk_root))
    return env


def build_sdk_env(config: dict[str, Any], sdk_root: Path | None) -> dict[str, str]:
    env = build_runtime_env(config, sdk_root)
    toolchain_root = Path(env.get("A2F3D_TOOLCHAIN_ROOT", str(Path.home() / ".local" / "nvidia" / "a2f3d-sdk")))
    cuda_root = Path(env.get("CUDA_HOME", str(toolchain_root / "cuda-12.9")))
    tensorrt_root = Path(env.get("TENSORRT_ROOT_DIR", str(toolchain_root / "tensorrt-10.13")))
    env["CUDA_HOME"] = str(cuda_root)
    env["CUDA_PATH"] = str(cuda_root)
    env["TENSORRT_ROOT_DIR"] = str(tensorrt_root)
    env["PATH"] = f"{tensorrt_root / 'bin'}:{cuda_root / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib64'}:{tensorrt_root / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return env


def _merge_json(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_json(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def stage_profile_config(config: dict[str, Any], run_dir: Path) -> tuple[dict[str, Any], Path | None]:
    base_profile_dir_value = config.get("base_profile_dir")
    if not base_profile_dir_value:
        return deepcopy(config), None

    base_profile_dir = Path(base_profile_dir_value)
    if not base_profile_dir.exists():
        raise FileNotFoundError(f"A2F base_profile_dir not found: {base_profile_dir}")

    profile_name = str(config.get("profile_name", base_profile_dir.name))
    staged_root = run_dir / f"{profile_name}_profile"
    if staged_root.exists():
        shutil.rmtree(staged_root)
    shutil.copytree(base_profile_dir, staged_root)

    for relative_path, overrides in (config.get("profile_overrides") or {}).items():
        target_path = staged_root / relative_path
        if not target_path.exists():
            raise FileNotFoundError(f"A2F profile override target not found: {target_path}")
        current = json.loads(target_path.read_text(encoding="utf-8"))
        merged = _merge_json(current, overrides)
        target_path.write_text(json.dumps(merged, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    resolved = deepcopy(config)
    resolved["staged_profile_dir"] = str(staged_root)
    resolved["profile_name"] = profile_name
    resolved["model_identity"] = str(config.get("model_identity", base_profile_dir.name))
    resolved["model_json"] = str(staged_root / str(config.get("model_json_relpath", "model.json")))

    snapshot = {
        "profile_name": profile_name,
        "mode": str(config.get("mode", "regression")),
        "base_profile_dir": str(base_profile_dir),
        "staged_profile_dir": str(staged_root),
        "model_identity": resolved["model_identity"],
        "model_json": resolved["model_json"],
        "frame_rate": int(config.get("frame_rate", 30)),
        "use_gpu_solver": bool(config.get("use_gpu_solver", False)),
        "device_id": int(config.get("device_id", 0)),
        "tuning_summary": config.get("tuning_summary") or {},
        "profile_overrides": config.get("profile_overrides") or {},
        "quality_study_group": config.get("quality_study_group"),
    }
    snapshot_path = run_dir / "profile_config_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    resolved["profile_snapshot_path"] = str(snapshot_path)
    return resolved, snapshot_path


def ensure_audio_is_16k_mono(audio_path: Path, temp_dir: Path) -> Path:
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sample_rate = sf.read(str(audio_path), always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)
    if data.dtype.kind != "f":
        data = data.astype(np.float32)
    if sample_rate != 16000:
        gcd = np.gcd(sample_rate, 16000)
        data = resample_poly(data, 16000 // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = 16000
    output_path = temp_dir / f"{audio_path.stem}_16k_mono.wav"
    sf.write(output_path, data, sample_rate)
    return output_path


def build_sdk_export_binary(config: dict[str, Any], sdk_root: Path | None) -> Path:
    sdk_root = sdk_root or Path(config["sdk_root"])
    build_dir = Path(config.get("build_dir", sdk_root / "_build" / "release"))
    workspace_root = _workspace_root()
    source_path = workspace_root / "tools" / "a2f3d_sdk_export.cpp"
    output_path = Path(config.get("bridge_binary", workspace_root / "output" / "tool_cache" / "a2f3d_sdk_export"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_mtime >= source_path.stat().st_mtime:
        return output_path

    audiofile_include = sdk_root / "_deps" / "target-deps" / "audiofile" / "include"
    sdk_include = build_dir / "audio2x-sdk" / "include"
    sdk_lib = build_dir / "audio2x-sdk" / "lib"
    compile_command = [
        "g++",
        "-std=c++17",
        "-O2",
        str(source_path),
        "-o",
        str(output_path),
        "-I",
        str(sdk_include),
        "-I",
        str(audiofile_include),
        "-L",
        str(sdk_lib),
        "-laudio2x",
        f"-Wl,-rpath,{sdk_lib}",
    ]
    subprocess.run(compile_command, check=True)
    return output_path


def run_sdk_bridge(args: argparse.Namespace, config: dict[str, Any]) -> None:
    sdk_root = Path(config.get("sdk_root") or args.sdk_root or "")
    model_json = Path(config["model_json"])
    if not model_json.exists():
        raise FileNotFoundError(f"A2F model_json not found: {model_json}")

    bridge_binary = build_sdk_export_binary(config, sdk_root if str(sdk_root) else None)
    with tempfile.TemporaryDirectory(prefix="a2f3d_audio_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        normalized_audio = ensure_audio_is_16k_mono(args.audio, temp_dir)
        command = [
            str(bridge_binary),
            "--model-json",
            str(model_json),
            "--audio-wav",
            str(normalized_audio),
            "--output-json",
            str(args.output_json),
            "--frame-rate",
            str(int(config.get("frame_rate", 30))),
            "--mode",
            str(config.get("mode", "regression")),
            "--device-id",
            str(int(config.get("device_id", 0))),
        ]
        if config.get("mode") == "diffusion" and "identity_index" in config:
            command.extend(["--identity-index", str(int(config.get("identity_index", 0)))])
        if bool(config.get("constant_noise", False)):
            command.append("--constant-noise")
        if bool(config.get("use_gpu_solver", False)):
            command.append("--gpu-solver")
        subprocess.run(command, check=True, env=build_sdk_env(config, sdk_root if str(sdk_root) else None))


def annotate_output_json(output_json: Path, config: dict[str, Any]) -> None:
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    metadata["profile_name"] = str(config.get("profile_name", "unnamed"))
    metadata["mode"] = str(config.get("mode", "regression"))
    metadata["model_identity"] = str(config.get("model_identity", config.get("profile_name", "unknown")))
    metadata["quality_study_group"] = config.get("quality_study_group")
    metadata["tuning_summary"] = config.get("tuning_summary") or {}
    if config.get("profile_snapshot_path"):
        metadata["profile_snapshot_path"] = str(config["profile_snapshot_path"])
    payload["metadata"] = metadata
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config_file)
    config, _ = stage_profile_config(config, args.output_json.parent)

    raw_source = config.get("raw_output_json")
    if raw_source:
        shutil.copy2(Path(raw_source), args.output_json)
        annotate_output_json(args.output_json, config)
        return 0

    if config.get("model_json"):
        run_sdk_bridge(args, config)
        annotate_output_json(args.output_json, config)
        return 0

    command_template = config.get("command")
    if not command_template:
        raise SystemExit(
            "A2F-3D SDK config must define either 'raw_output_json' for a precomputed payload or 'command' for live inference."
        )

    formatted_command = str(command_template).format(
        audio=str(args.audio.resolve()),
        output_json=str(args.output_json),
        sdk_root=str(args.sdk_root or config.get("sdk_root", "")),
        build_dir=str(config.get("build_dir", "")),
        toolchain_root=str(config.get("toolchain_root", "")),
        model_dir=str(config.get("model_dir", "")),
    )
    subprocess.run(formatted_command, shell=True, check=True, env=build_runtime_env(config, args.sdk_root))
    if not args.output_json.exists():
        raise FileNotFoundError(f"A2F-3D SDK command did not produce expected JSON: {args.output_json}")
    annotate_output_json(args.output_json, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
