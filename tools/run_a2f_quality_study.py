#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tools import face_quality_report


DEFAULT_PROFILES = ("raw-regression", "tuned-regression-v1", "tuned-regression-v2", "lam-baseline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused Audio2Face quality study on one clip.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--body-npz", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--render-script", type=Path, default=WORKSPACE_ROOT / "tools" / "render_hybrid_showcase_video.py")
    parser.add_argument("--a2f-python", type=str, default=sys.executable)
    parser.add_argument("--a2f-sdk-root", type=Path, default=WORKSPACE_ROOT / "third_party" / "Audio2Face-3D-SDK")
    parser.add_argument("--lam-python", type=str, default="/home/vanto/anaconda3/envs/lam-a2e310/bin/python")
    parser.add_argument("--lam-repo", type=Path, default=WORKSPACE_ROOT / "third_party" / "LAM_Audio2Expression")
    return parser.parse_args(argv)


def build_study_output_dir(output_root: Path, audio_path: Path) -> Path:
    return output_root / audio_path.stem


def profile_config_path(profile_name: str) -> Path:
    return WORKSPACE_ROOT / "configs" / "a2f_profiles" / f"{profile_name}.json"


def build_hybrid_command(args: argparse.Namespace, profile_name: str, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(WORKSPACE_ROOT / "tools" / "hybrid_pipeline.py"),
        "--audio",
        str(args.audio),
        "--output-dir",
        str(run_dir),
        "--body-npz",
        str(args.body_npz),
        "--target-fps",
        str(args.target_fps),
    ]
    if profile_name == "lam-baseline":
        command.extend(["--face-provider", "lam", "--face-python", args.lam_python, "--face-repo", str(args.lam_repo)])
    else:
        command.extend(
            [
                "--face-provider",
                "a2f-3d-sdk",
                "--face-python",
                args.a2f_python,
                "--face-repo",
                str(args.a2f_sdk_root),
                "--face-config-file",
                str(profile_config_path(profile_name)),
            ]
        )
    return command


def build_render_command(args: argparse.Namespace, profile_name: str, run_dir: Path) -> list[str]:
    return [
        args.lam_python,
        str(args.render_script),
        "--audio",
        str(args.audio),
        "--face-json",
        str(run_dir / "face" / "arkit_blendshapes.json"),
        "--face-provider",
        "lam" if profile_name == "lam-baseline" else "a2f-3d-sdk",
        "--output",
        str(run_dir / "showcase.mp4"),
        "--lam-python",
        args.lam_python,
    ]


def write_study_manifest(
    *,
    manifest_path: Path,
    audio_path: Path,
    profile_outputs: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    manifest_path.write_text(
        json.dumps(
            {
                "audio_path": str(audio_path),
                "profiles": profile_outputs,
                "summary": summary,
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    study_dir = build_study_output_dir(args.output_root, args.audio)
    study_dir.mkdir(parents=True, exist_ok=True)

    profile_outputs: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for profile_name in args.profiles:
        run_dir = study_dir / profile_name
        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(build_hybrid_command(args, profile_name, run_dir), check=True)
        subprocess.run(build_render_command(args, profile_name, run_dir), check=True)

        report_path = run_dir / "metrics" / "face_quality_report.json"
        if not report_path.exists():
            report = face_quality_report.analyze_face_quality(
                audio_path=args.audio,
                face_json=run_dir / "face" / "arkit_blendshapes.json",
                label=profile_name,
            )
            report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        reports[profile_name] = json.loads(report_path.read_text(encoding="utf-8"))

        snapshot_path = run_dir / "face" / "profile_config_snapshot.json"
        tuning_summary = {}
        if snapshot_path.exists():
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            tuning_summary = snapshot.get("tuning_summary") or {}
        profile_outputs[profile_name] = {
            "run_dir": str(run_dir),
            "video": str(run_dir / "showcase.mp4"),
            "quality_report": str(report_path),
            "manifest": str(run_dir / "manifest.json"),
            "profile_config_snapshot": str(snapshot_path) if snapshot_path.exists() else None,
            "tuning_summary": tuning_summary,
        }

    summary = face_quality_report.summarize_quality_study(reports)
    write_study_manifest(
        manifest_path=study_dir / "study_manifest.json",
        audio_path=args.audio,
        profile_outputs=profile_outputs,
        summary=summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
