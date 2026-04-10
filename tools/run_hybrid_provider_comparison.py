#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run one EMAGE body source against multiple face providers.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--body-npz", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", default=["a2f-3d-sdk", "lam"])
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--render-script", type=Path, default=workspace_root / "tools" / "render_hybrid_showcase_video.py")
    parser.add_argument("--a2f-config-file", type=Path, default=None)
    parser.add_argument("--a2f-python", type=str, default=sys.executable)
    parser.add_argument("--a2f-sdk-root", type=Path, default=None)
    parser.add_argument("--lam-python", type=str, default="/home/vanto/anaconda3/envs/lam-a2e310/bin/python")
    parser.add_argument("--lam-repo", type=Path, default=workspace_root / "third_party" / "LAM_Audio2Expression")
    return parser.parse_args(argv)


def build_provider_output_dir(output_root: Path, provider: str) -> Path:
    return output_root / provider


def _provider_command(args: argparse.Namespace, provider: str, output_dir: Path) -> list[str]:
    workspace_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(workspace_root / "tools" / "hybrid_pipeline.py"),
        "--audio",
        str(args.audio),
        "--output-dir",
        str(output_dir),
        "--body-npz",
        str(args.body_npz),
        "--face-provider",
        provider,
        "--target-fps",
        str(args.target_fps),
    ]
    if provider == "a2f-3d-sdk":
        if args.a2f_config_file is None:
            raise SystemExit("--a2f-config-file is required when providers include a2f-3d-sdk")
        command.extend(
            [
                "--face-config-file",
                str(args.a2f_config_file),
                "--face-python",
                args.a2f_python,
            ]
        )
        if args.a2f_sdk_root is not None:
            command.extend(["--face-repo", str(args.a2f_sdk_root)])
    elif provider == "lam":
        command.extend(["--face-python", args.lam_python, "--face-repo", str(args.lam_repo)])
    else:
        raise SystemExit(f"Unsupported provider for comparison: {provider}")
    return command


def build_render_command(
    *,
    args: argparse.Namespace,
    provider: str,
    run_dir: Path,
    showcase_video: Path,
) -> list[str]:
    return [
        args.lam_python,
        str(args.render_script),
        "--audio",
        str(args.audio),
        "--face-json",
        str(run_dir / "face" / "arkit_blendshapes.json"),
        "--face-provider",
        provider,
        "--output",
        str(showcase_video),
        "--lam-python",
        args.lam_python,
    ]


def write_comparison_manifest(
    *,
    manifest_path: Path,
    audio_path: Path,
    provider_outputs: dict[str, dict[str, Any]],
) -> None:
    manifest_path.write_text(
        json.dumps(
            {
                "audio_path": str(audio_path),
                "providers": provider_outputs,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    provider_outputs: dict[str, dict[str, Any]] = {}

    for provider in args.providers:
        run_dir = build_provider_output_dir(args.output_root, provider)
        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(_provider_command(args, provider, run_dir), check=True)

        showcase_video = run_dir / "showcase.mp4"
        subprocess.run(build_render_command(args=args, provider=provider, run_dir=run_dir, showcase_video=showcase_video), check=True)
        provider_outputs[provider] = {
            "run_dir": str(run_dir),
            "video": str(showcase_video),
            "manifest": str(run_dir / "manifest.json"),
        }

    write_comparison_manifest(
        manifest_path=args.output_root / "comparison_manifest.json",
        audio_path=args.audio,
        provider_outputs=provider_outputs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
