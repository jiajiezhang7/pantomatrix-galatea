#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


class ComponentRunResult:
    def __init__(
        self,
        *,
        status: str,
        fps: float,
        frame_count: int,
        payload_path: Path | None,
        raw_payload_path: Path | None = None,
        provider_name: str | None = None,
        normalization_policy: str = "provider-native",
        provider_notes: list[str] | None = None,
        quality_notes: list[str] | None = None,
        timestamps: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.fps = fps
        self.frame_count = frame_count
        self.payload_path = payload_path
        self.raw_payload_path = raw_payload_path
        self.provider_name = provider_name
        self.normalization_policy = normalization_policy
        self.provider_notes = provider_notes or []
        self.quality_notes = quality_notes or []
        self.timestamps = timestamps or []
        self.metadata = metadata or {}


class FaceProviderConfig:
    def __init__(
        self,
        *,
        input_json: Path | None,
        command: str,
        python_bin: str,
        repo_root: Path | None,
        output_json_name: str,
        model_weight: Path | None = None,
        config_file: Path | None = None,
    ) -> None:
        self.input_json = input_json
        self.command = command
        self.python_bin = python_bin
        self.repo_root = repo_root
        self.output_json_name = output_json_name
        self.model_weight = model_weight
        self.config_file = config_file


class FaceAdapter:
    provider_name = "base"

    def __init__(self, *, provider_config: FaceProviderConfig, audio_path: Path) -> None:
        self.provider_config = provider_config
        self.audio_path = audio_path

    def run(self, args: argparse.Namespace, scratch_dir: Path) -> ComponentRunResult:
        raise NotImplementedError


class LamFaceAdapter(FaceAdapter):
    provider_name = "lam"

    def run(self, args: argparse.Namespace, scratch_dir: Path) -> ComponentRunResult:
        if self.provider_config.input_json:
            frames, fps = _load_face_payload(self.provider_config.input_json)
            return ComponentRunResult(
                status="ready",
                fps=fps,
                frame_count=len(frames),
                payload_path=self.provider_config.input_json,
                raw_payload_path=self.provider_config.input_json,
                provider_name=self.provider_name,
                normalization_policy="provider-native",
                provider_notes=["Precomputed LAM payload is passed through without repository-level renormalization."],
                quality_notes=["used precomputed face provider json", "provider=lam"],
                timestamps=[float(frame["time_sec"]) for frame in frames],
                metadata={"mode": "precomputed", "provider": self.provider_name},
            )

        if self.provider_config.command:
            return self._run_external_command(scratch_dir)
        return self._run_official_lam(scratch_dir)

    def _run_external_command(self, scratch_dir: Path) -> ComponentRunResult:
        if not self.provider_config.command:
            raise SystemExit(
                "Face provider output is required: pass --face-input-json or configure --face-command."
            )

        provider_output_dir = scratch_dir / f"{self.provider_name}_output"
        provider_output_dir.mkdir(parents=True, exist_ok=True)
        formatted_command = self.provider_config.command.format(
            audio=str(self.audio_path),
            output_dir=str(provider_output_dir),
            repo=str(self.provider_config.repo_root or ""),
            python=self.provider_config.python_bin,
        )
        subprocess.run(formatted_command, shell=True, check=True)
        provider_json = provider_output_dir / self.provider_config.output_json_name
        if not provider_json.exists():
            raise FileNotFoundError(
                f"Face provider command did not produce expected JSON: {provider_json}"
            )
        frames, fps = _load_face_payload(provider_json)
        return ComponentRunResult(
            status="ready",
            fps=fps,
            frame_count=len(frames),
            payload_path=provider_json,
            raw_payload_path=provider_json,
            provider_name=self.provider_name,
            normalization_policy="provider-native",
            provider_notes=["External LAM output is treated as provider-native coefficients."],
            quality_notes=["generated via external face provider command", "provider=lam"],
            timestamps=[float(frame["time_sec"]) for frame in frames],
            metadata={"mode": "generated", "command": formatted_command, "provider": self.provider_name},
        )

    def _run_official_lam(self, scratch_dir: Path) -> ComponentRunResult:
        if self.provider_config.repo_root is None:
            raise SystemExit("LAM official-run requires --face-repo")

        provider_output_dir = scratch_dir / f"{self.provider_name}_output"
        provider_output_dir.mkdir(parents=True, exist_ok=True)
        command, official_json = build_lam_official_command(
            provider_config=self.provider_config,
            audio_path=self.audio_path,
            output_dir=provider_output_dir,
        )
        subprocess.run(command, check=True, cwd=self.provider_config.repo_root)
        if not official_json.exists():
            raise FileNotFoundError(f"LAM official run did not produce expected JSON: {official_json}")

        converted_payload = convert_lam_official_json_to_face_payload(official_json)
        hybrid_json = provider_output_dir / "arkit_blendshapes.json"
        hybrid_json.write_text(json.dumps(converted_payload, ensure_ascii=True), encoding="utf-8")

        frames = converted_payload["frames"]
        fps = float(converted_payload["fps"])
        return ComponentRunResult(
            status="ready",
            fps=fps,
            frame_count=len(frames),
            payload_path=hybrid_json,
            raw_payload_path=official_json,
            provider_name=self.provider_name,
            normalization_policy="provider-native",
            provider_notes=["LAM eye-related coefficients are preserved inside the face payload."],
            quality_notes=["generated via official LAM inference", "provider=lam"],
            timestamps=[float(frame["time_sec"]) for frame in frames],
            metadata={
                "mode": "official-run",
                "provider": self.provider_name,
                "official_output_json": str(official_json),
                "command": command,
            },
        )


class A2F3DSdkFaceAdapter(FaceAdapter):
    provider_name = "a2f-3d-sdk"

    def run(self, args: argparse.Namespace, scratch_dir: Path) -> ComponentRunResult:
        if self.provider_config.input_json:
            converted_payload = convert_a2f3d_sdk_json_to_face_payload(self.provider_config.input_json)
            converted_json = scratch_dir / f"{self.provider_name}_precomputed.json"
            converted_json.write_text(json.dumps(converted_payload, ensure_ascii=True), encoding="utf-8")
            frames = converted_payload["frames"]
            fps = float(converted_payload["fps"])
            return ComponentRunResult(
                status="ready",
                fps=fps,
                frame_count=len(frames),
                payload_path=converted_json,
                raw_payload_path=self.provider_config.input_json,
                provider_name=self.provider_name,
                normalization_policy="raw-a2f",
                provider_notes=_default_a2f_provider_notes(),
                quality_notes=["used precomputed A2F-3D SDK json", "provider=a2f-3d-sdk"],
                timestamps=[float(frame["time_sec"]) for frame in frames],
                metadata={"mode": "precomputed", "provider": self.provider_name},
            )

        if self.provider_config.command:
            return self._run_external_command(scratch_dir)
        return self._run_sdk_wrapper(scratch_dir)

    def _run_external_command(self, scratch_dir: Path) -> ComponentRunResult:
        provider_output_dir = scratch_dir / f"{self.provider_name}_output"
        provider_output_dir.mkdir(parents=True, exist_ok=True)
        formatted_command = self.provider_config.command.format(
            audio=str(self.audio_path),
            output_dir=str(provider_output_dir),
            repo=str(self.provider_config.repo_root or ""),
            python=self.provider_config.python_bin,
            config=str(self.provider_config.config_file or ""),
        )
        subprocess.run(formatted_command, shell=True, check=True)
        raw_json = provider_output_dir / self.provider_config.output_json_name
        if not raw_json.exists():
            raise FileNotFoundError(f"A2F-3D SDK command did not produce expected JSON: {raw_json}")
        return self._build_sdk_result(raw_json=raw_json)

    def _run_sdk_wrapper(self, scratch_dir: Path) -> ComponentRunResult:
        if self.provider_config.config_file is None:
            raise SystemExit("A2F-3D SDK run requires --face-config-file")
        provider_output_dir = scratch_dir / f"{self.provider_name}_output"
        provider_output_dir.mkdir(parents=True, exist_ok=True)
        command, raw_json = build_a2f3d_sdk_command(
            provider_config=self.provider_config,
            audio_path=self.audio_path,
            output_dir=provider_output_dir,
        )
        subprocess.run(command, check=True)
        if not raw_json.exists():
            raise FileNotFoundError(f"A2F-3D SDK wrapper did not produce expected JSON: {raw_json}")
        return self._build_sdk_result(raw_json=raw_json, command=command)

    def _build_sdk_result(self, *, raw_json: Path, command: list[str] | None = None) -> ComponentRunResult:
        converted_payload = convert_a2f3d_sdk_json_to_face_payload(raw_json)
        hybrid_json = raw_json.parent / "arkit_blendshapes.json"
        hybrid_json.write_text(json.dumps(converted_payload, ensure_ascii=True), encoding="utf-8")
        frames = converted_payload["frames"]
        fps = float(converted_payload["fps"])
        return ComponentRunResult(
            status="ready",
            fps=fps,
            frame_count=len(frames),
            payload_path=hybrid_json,
            raw_payload_path=raw_json,
            provider_name=self.provider_name,
            normalization_policy="raw-a2f",
            provider_notes=_default_a2f_provider_notes(),
            quality_notes=["generated via A2F-3D SDK wrapper", "provider=a2f-3d-sdk"],
            timestamps=[float(frame["time_sec"]) for frame in frames],
            metadata={
                "mode": "sdk-run",
                "provider": self.provider_name,
                "raw_output_json": str(raw_json),
                "command": command or [],
            },
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the hybrid EMAGE + face-provider pipeline.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--face-provider", type=str, default="lam")

    # Body lane
    parser.add_argument("--body-npz", type=Path, default=None)
    parser.add_argument("--emage-python", type=str, default=sys.executable)
    parser.add_argument(
        "--emage-script",
        type=Path,
        default=workspace_root / "tools" / "run_emage_audio_inference.py",
    )

    # Face lane
    parser.add_argument("--face-input-json", type=Path, default=None)
    parser.add_argument("--face-command", type=str, default="")
    parser.add_argument("--face-python", type=str, default="")
    parser.add_argument("--face-repo", type=Path, default=None)
    parser.add_argument("--face-output-json-name", type=str, default="")
    parser.add_argument("--face-model-weight", type=Path, default=None)
    parser.add_argument("--face-config-file", type=Path, default=None)

    # Backward-compatible LAM aliases
    parser.add_argument("--lam-json", type=Path, default=None)
    parser.add_argument("--lam-command", type=str, default="")
    parser.add_argument("--lam-python", type=str, default="python")
    parser.add_argument("--lam-repo", type=Path, default=None)
    parser.add_argument("--lam-output-json-name", type=str, default="arkit_blendshapes.json")
    parser.add_argument("--lam-weight", type=Path, default=None)
    parser.add_argument("--lam-config-file", type=Path, default=None)
    return parser.parse_args(argv)


def build_face_provider_config(args: argparse.Namespace) -> FaceProviderConfig:
    return FaceProviderConfig(
        input_json=args.face_input_json or args.lam_json,
        command=args.face_command or args.lam_command,
        python_bin=args.face_python or args.lam_python,
        repo_root=args.face_repo or args.lam_repo,
        output_json_name=args.face_output_json_name or args.lam_output_json_name,
        model_weight=args.face_model_weight or args.lam_weight,
        config_file=args.face_config_file or args.lam_config_file,
    )


def build_face_provider(
    *,
    provider: str,
    provider_config: FaceProviderConfig,
    audio_path: Path,
) -> FaceAdapter:
    registry = {
        "lam": LamFaceAdapter,
        "a2f-3d-sdk": A2F3DSdkFaceAdapter,
    }
    adapter_cls = registry.get(provider)
    if adapter_cls is None:
        raise SystemExit(f"Unsupported face provider: {provider}")
    return adapter_cls(provider_config=provider_config, audio_path=audio_path)


def build_canonical_timeline(duration_sec: float, target_fps: float) -> list[float]:
    if duration_sec <= 0:
        return []
    frame_count = max(1, int(round(duration_sec * target_fps)))
    return [frame_idx / target_fps for frame_idx in range(frame_count)]


def _component_duration(frame_count: int, fps: float) -> float:
    if fps <= 0 or frame_count <= 0:
        return 0.0
    return frame_count / fps


def _safe_audio_duration(audio_path: Path) -> float | None:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        if frame_rate <= 0:
            return None
        return frame_count / frame_rate
    except (wave.Error, FileNotFoundError, EOFError):
        return None


def _ensure_output_layout(output_dir: Path) -> dict[str, Path]:
    layout = {
        "root": output_dir,
        "body": output_dir / "body",
        "face": output_dir / "face",
        "metrics": output_dir / "metrics",
        "logs": output_dir / "logs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _relative_to_output(output_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _load_face_payload(face_json: Path) -> tuple[list[dict[str, Any]], float]:
    payload = json.loads(face_json.read_text(encoding="utf-8"))
    frames = payload["frames"] if isinstance(payload, dict) else payload
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No face frames found in {face_json}")
    fps = float(payload.get("fps", _infer_fps_from_frames(frames))) if isinstance(payload, dict) else _infer_fps_from_frames(frames)
    return frames, fps


def _infer_fps_from_frames(frames: list[dict[str, Any]]) -> float:
    if len(frames) > 1:
        delta = float(frames[1]["time_sec"]) - float(frames[0]["time_sec"])
        if delta > 0:
            return 1.0 / delta
    return 30.0


def _normalize_face_frames(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    normalized = []
    blendshape_names: set[str] = set()
    for idx, frame in enumerate(frames):
        blendshapes = dict(frame["blendshapes"])
        blendshape_names.update(blendshapes)
        normalized.append(
            {
                "frame": int(frame.get("frame", idx)),
                "time_sec": float(frame["time_sec"]),
                "blendshapes": {name: float(value) for name, value in blendshapes.items()},
            }
        )
    return normalized, sorted(blendshape_names)


def _extract_eye_blendshape_names(blendshape_names: list[str]) -> list[str]:
    return [name for name in blendshape_names if name.startswith("eye")]


def _default_a2f_provider_notes() -> list[str]:
    return [
        "A2F-3D output is preserved in raw-fidelity mode without repository-level renormalization.",
        "Eye-look coefficients may be omitted or remain zero in provider-native A2F output.",
        "MouthClose follows NVIDIA semantics and is not reinterpreted by this repository.",
    ]


def convert_lam_official_json_to_face_payload(official_json_path: Path) -> dict[str, Any]:
    payload = json.loads(official_json_path.read_text(encoding="utf-8"))
    names = payload.get("names") or payload.get("metadata", {}).get("blendshape_names")
    frames = payload.get("frames", [])
    if not names or not frames:
        raise ValueError(f"LAM official JSON missing names/frames: {official_json_path}")

    fps = payload.get("metadata", {}).get("fps")
    if fps is None:
        if len(frames) > 1:
            delta = float(frames[1]["time"]) - float(frames[0]["time"])
            if delta <= 0:
                raise ValueError(f"Unable to infer fps from LAM official JSON: {official_json_path}")
            fps = 1.0 / delta
        else:
            raise ValueError(f"LAM official JSON missing fps metadata: {official_json_path}")

    converted_frames = []
    for frame_idx, frame in enumerate(frames):
        weights = frame["weights"]
        if len(weights) != len(names):
            raise ValueError("LAM official JSON names/weights length mismatch")
        converted_frames.append(
            {
                "frame": frame_idx,
                "time_sec": float(frame.get("time", frame_idx / float(fps))),
                "blendshapes": {name: float(value) for name, value in zip(names, weights)},
            }
        )

    return {
        "source_face_json": str(official_json_path),
        "fps": float(fps),
        "frame_count": len(converted_frames),
        "frames": converted_frames,
        "names": list(names),
    }


def convert_a2f3d_sdk_json_to_face_payload(raw_json_path: Path) -> dict[str, Any]:
    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"A2F-3D SDK JSON missing frames: {raw_json_path}")

    names = payload.get("names") or metadata.get("blendshape_names")
    fps = payload.get("fps") or metadata.get("fps")
    if fps is None:
        fps = _infer_fps_from_frames(
            [
                {
                    "time_sec": float(frame.get("time_sec", frame.get("time", 0.0))),
                    "blendshapes": frame.get("blendshapes", {}),
                }
                for frame in frames
            ]
        )

    converted_frames = []
    inferred_names: set[str] = set(names or [])
    for frame_idx, frame in enumerate(frames):
        if "blendshapes" in frame:
            blendshapes = {str(name): float(value) for name, value in frame["blendshapes"].items()}
        else:
            weights = frame.get("weights")
            if not names or weights is None or len(weights) != len(names):
                raise ValueError(f"A2F-3D SDK JSON missing usable blendshape payload: {raw_json_path}")
            blendshapes = {name: float(value) for name, value in zip(names, weights)}
        inferred_names.update(blendshapes)
        converted_frames.append(
            {
                "frame": frame_idx,
                "time_sec": float(frame.get("time_sec", frame.get("time", frame_idx / float(fps)))),
                "blendshapes": blendshapes,
            }
        )

    blendshape_names = sorted(inferred_names)
    return {
        "provider": "a2f-3d-sdk",
        "source_face_json": str(raw_json_path),
        "fps": float(fps),
        "frame_count": len(converted_frames),
        "frames": converted_frames,
        "names": blendshape_names,
        "eye_blendshape_names": _extract_eye_blendshape_names(blendshape_names),
        "normalization_policy": str(metadata.get("normalization_policy", "raw-a2f")),
        "provider_notes": metadata.get("notes") or _default_a2f_provider_notes(),
    }


def build_lam_official_command(
    *,
    provider_config: FaceProviderConfig,
    audio_path: Path,
    output_dir: Path,
) -> tuple[list[str], Path]:
    if provider_config.repo_root is None:
        raise SystemExit("LAM official-run requires --face-repo")

    repo_root = provider_config.repo_root
    config_file = provider_config.config_file or (repo_root / "configs" / "lam_audio2exp_config_streaming.py")
    model_weight = provider_config.model_weight or (repo_root / "pretrained_models" / "lam_audio2exp_streaming.tar")
    output_json = output_dir / provider_config.output_json_name
    save_path = output_dir / "exp_audio2exp"
    resolved_audio = audio_path.resolve()

    command = [
        provider_config.python_bin or "python",
        str(repo_root / "inference.py"),
        "--config-file",
        str(config_file),
        "--options",
        f"save_path={save_path}",
        f"save_json_path={output_json}",
        f"weight={model_weight}",
        f"audio_input={resolved_audio}",
        "ex_vol=False",
    ]
    return command, output_json


def build_a2f3d_sdk_command(
    *,
    provider_config: FaceProviderConfig,
    audio_path: Path,
    output_dir: Path,
) -> tuple[list[str], Path]:
    if provider_config.config_file is None:
        raise SystemExit("A2F-3D SDK run requires --face-config-file")

    workspace_root = Path(__file__).resolve().parents[1]
    output_json = output_dir / provider_config.output_json_name
    command = [
        provider_config.python_bin or "python",
        str(workspace_root / "tools" / "run_a2f3d_sdk_inference.py"),
        "--audio",
        str(audio_path.resolve()),
        "--output-json",
        str(output_json),
        "--config-file",
        str(provider_config.config_file),
    ]
    if provider_config.repo_root is not None:
        command.extend(["--sdk-root", str(provider_config.repo_root)])
    return command, output_json


def align_face_frames(
    source_frames: list[dict[str, Any]],
    timeline: list[float],
    blendshape_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    normalized_frames, inferred_names = _normalize_face_frames(source_frames)
    names = blendshape_names or inferred_names
    source_times = np.array([frame["time_sec"] for frame in normalized_frames], dtype=np.float32)
    target_times = np.array(timeline, dtype=np.float32)

    aligned = []
    for frame_idx, time_sec in enumerate(target_times):
        weights = {}
        for name in names:
            source_values = np.array(
                [frame["blendshapes"].get(name, 0.0) for frame in normalized_frames],
                dtype=np.float32,
            )
            value = float(np.interp(time_sec, source_times, source_values))
            weights[name] = min(1.0, max(0.0, value))
        aligned.append({"frame": frame_idx, "time_sec": float(time_sec), "blendshapes": weights})
    return aligned


def _interpolate_matrix(source: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if len(source_times) == 1:
        return np.repeat(source[:1], repeats=len(target_times), axis=0)

    columns = []
    for column_idx in range(source.shape[1]):
        columns.append(np.interp(target_times, source_times, source[:, column_idx]))
    return np.stack(columns, axis=1).astype(np.float32)


def align_body_motion(body_npz: Path, timeline: list[float], output_path: Path) -> tuple[Path, float]:
    motion = np.load(body_npz, allow_pickle=True)
    source_fps = float(motion["mocap_frame_rate"]) if "mocap_frame_rate" in motion.files else 30.0
    source_frame_count = int(motion["poses"].shape[0])
    source_times = np.array([frame_idx / source_fps for frame_idx in range(source_frame_count)], dtype=np.float32)
    target_times = np.array(timeline, dtype=np.float32)

    poses = _interpolate_matrix(motion["poses"].astype(np.float32), source_times, target_times)
    expressions = _interpolate_matrix(motion["expressions"].astype(np.float32), source_times, target_times)
    trans = _interpolate_matrix(motion["trans"].astype(np.float32), source_times, target_times)

    np.savez(
        output_path,
        betas=motion["betas"].astype(np.float32),
        poses=poses,
        expressions=expressions,
        trans=trans,
        model=motion["model"],
        gender=motion["gender"],
        mocap_frame_rate=np.array(30.0, dtype=np.float32),
    )
    return output_path, source_fps


def _copy_or_run_emage(args: argparse.Namespace, scratch_dir: Path) -> ComponentRunResult:
    if args.body_npz:
        body_npz = args.body_npz
        if not body_npz.exists():
            raise FileNotFoundError(f"Body NPZ not found: {body_npz}")
        motion = np.load(body_npz, allow_pickle=True)
        fps = float(motion["mocap_frame_rate"]) if "mocap_frame_rate" in motion.files else 30.0
        frame_count = int(motion["poses"].shape[0])
        return ComponentRunResult(
            status="ready",
            fps=fps,
            frame_count=frame_count,
            payload_path=body_npz,
            quality_notes=["used precomputed body npz"],
            timestamps=[frame_idx / fps for frame_idx in range(frame_count)],
            metadata={"mode": "precomputed"},
        )

    audio_dir = scratch_dir / "emage_audio"
    output_dir = scratch_dir / "emage_output"
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_audio = audio_dir / args.audio.name
    shutil.copy2(args.audio, staged_audio)

    command = [
        args.emage_python,
        str(args.emage_script),
        "--audio-folder",
        str(audio_dir),
        "--save-folder",
        str(output_dir),
    ]
    subprocess.run(command, check=True)

    body_npz = output_dir / f"{args.audio.stem}_output.npz"
    motion = np.load(body_npz, allow_pickle=True)
    fps = float(motion["mocap_frame_rate"]) if "mocap_frame_rate" in motion.files else 30.0
    frame_count = int(motion["poses"].shape[0])
    return ComponentRunResult(
        status="ready",
        fps=fps,
        frame_count=frame_count,
        payload_path=body_npz,
        quality_notes=["generated via run_emage_audio_inference.py"],
        timestamps=[frame_idx / fps for frame_idx in range(frame_count)],
        metadata={"mode": "generated", "command": command},
    )


def _write_log(log_path: Path, lines: list[str]) -> None:
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_face_payload(
    source_face_json: Path,
    timeline: list[float],
    output_path: Path,
) -> tuple[Path, float, int, list[str]]:
    frames, source_fps = _load_face_payload(source_face_json)
    _, blendshape_names = _normalize_face_frames(frames)
    aligned_frames = align_face_frames(frames, timeline, blendshape_names=blendshape_names)
    payload = {
        "source_face_json": str(source_face_json),
        "fps": 30.0,
        "frame_count": len(aligned_frames),
        "names": blendshape_names,
        "eye_blendshape_names": _extract_eye_blendshape_names(blendshape_names),
        "frames": aligned_frames,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return output_path, source_fps, len(aligned_frames), blendshape_names


def _write_provider_metadata(output_path: Path, result: ComponentRunResult, output_dir: Path) -> None:
    payload = {
        "provider": result.provider_name,
        "normalization_policy": result.normalization_policy,
        "raw_payload_path": _relative_to_output(output_dir, result.raw_payload_path),
        "notes": result.provider_notes,
        "metadata": result.metadata,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _component_manifest_entry(result: ComponentRunResult, output_dir: Path) -> dict[str, Any]:
    return {
        "status": result.status,
        "fps": result.fps,
        "frame_count": result.frame_count,
        "payload_path": _relative_to_output(output_dir, result.payload_path),
        "provider_name": result.provider_name,
        "raw_payload_path": _relative_to_output(output_dir, result.raw_payload_path),
        "normalization_policy": result.normalization_policy,
        "quality_notes": result.quality_notes,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layout = _ensure_output_layout(args.output_dir)
    log_lines = [f"audio={args.audio}", f"target_fps={args.target_fps}", f"face_provider={args.face_provider}"]

    timings: dict[str, float] = {}
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="hybrid_pipeline_") as scratch_dir_str:
        scratch_dir = Path(scratch_dir_str)
        face_provider = build_face_provider(
            provider=args.face_provider,
            provider_config=build_face_provider_config(args),
            audio_path=args.audio,
        )

        lane_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as executor:
            body_future = executor.submit(_copy_or_run_emage, args, scratch_dir)
            face_future = executor.submit(face_provider.run, args, scratch_dir)
            body_result = body_future.result()
            face_result = face_future.result()
        timings["body"] = time.perf_counter() - lane_start
        timings["face"] = max(0.0, timings["body"])

        duration_candidates = [
            _safe_audio_duration(args.audio),
            _component_duration(body_result.frame_count, body_result.fps),
            _component_duration(face_result.frame_count, face_result.fps),
        ]
        duration_sec = max(candidate for candidate in duration_candidates if candidate is not None)
        timeline = build_canonical_timeline(duration_sec=duration_sec, target_fps=args.target_fps)

        alignment_start = time.perf_counter()
        body_output_path, body_source_fps = align_body_motion(body_result.payload_path, timeline, layout["body"] / "body.npz")
        face_output_path, face_source_fps, face_frame_count, face_blendshape_names = _write_face_payload(
            face_result.payload_path,
            timeline,
            layout["face"] / "arkit_blendshapes.json",
        )
        _write_provider_metadata(layout["face"] / "provider_metadata.json", face_result, args.output_dir)
        timings["alignment"] = time.perf_counter() - alignment_start

        body_component = ComponentRunResult(
            status="ready",
            fps=args.target_fps,
            frame_count=len(timeline),
            payload_path=body_output_path,
            quality_notes=body_result.quality_notes + [f"source_fps={body_source_fps}"],
            timestamps=list(timeline),
            metadata=body_result.metadata,
        )
        face_component = ComponentRunResult(
            status="ready",
            fps=args.target_fps,
            frame_count=face_frame_count,
            payload_path=face_output_path,
            raw_payload_path=face_result.raw_payload_path,
            provider_name=face_result.provider_name,
            normalization_policy=face_result.normalization_policy,
            provider_notes=face_result.provider_notes,
            quality_notes=face_result.quality_notes
            + [f"source_fps={face_source_fps}", f"eye_blendshape_count={len(_extract_eye_blendshape_names(face_blendshape_names))}"],
            timestamps=list(timeline),
            metadata=face_result.metadata,
        )

        manifest = {
            "audio_path": str(args.audio),
            "target_fps": float(args.target_fps),
            "duration_sec": float(duration_sec),
            "frame_count": len(timeline),
            "components": {
                "emage_body": _component_manifest_entry(body_component, args.output_dir),
                "face_provider": _component_manifest_entry(face_component, args.output_dir),
            },
            "component_aliases": {
                "lam_face": "face_provider",
            },
            "sources": {
                "body": str(body_result.payload_path),
                "face": str(face_result.payload_path),
                "face_provider": args.face_provider,
            },
            "alignment_policy": {
                "timeline_source": "audio_duration_or_component_fallback",
                "target_fps": float(args.target_fps),
                "resample": "linear",
            },
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")

        timings["total"] = time.perf_counter() - start_time
        metrics_payload = {
            "timings_sec": timings,
            "frame_count": len(timeline),
        }
        (layout["metrics"] / "run_metrics.json").write_text(
            json.dumps(metrics_payload, ensure_ascii=True),
            encoding="utf-8",
        )

        log_lines.extend(
            [
                f"body_status={body_component.status}",
                f"face_status={face_component.status}",
                f"frame_count={len(timeline)}",
            ]
        )
        _write_log(layout["logs"] / "pipeline.log", log_lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
