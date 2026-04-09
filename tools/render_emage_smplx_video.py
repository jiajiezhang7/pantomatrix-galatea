#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path


RENDER_ENGINES = ("CYCLES", "BLENDER_EEVEE_NEXT")
FFMPEG_FALLBACKS = (
    Path("/home/vanto/anaconda3/envs/pantomatrix-emage39/bin/ffmpeg"),
    Path("/home/vanto/anaconda3/envs/galatea/bin/ffmpeg"),
    Path("/home/vanto/anaconda3/envs/convofusion-demo/bin/ffmpeg"),
    Path("/home/vanto/anaconda3/envs/a2p-blackwell/bin/ffmpeg"),
)


@dataclasses.dataclass(frozen=True)
class RenderJob:
    clip_name: str
    npz_path: Path
    audio_path: Path
    output_dir: Path
    preview_path: Path
    video_path: Path
    work_dir: Path
    silent_video_path: Path
    log_path: Path


def resolve_binary(name: str, fallbacks: list[Path] | tuple[Path, ...]) -> Path:
    path = shutil.which(name)
    if path:
        return Path(path)
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find executable '{name}'")


def normalize_clip_name(name: str) -> str:
    for suffix in ("_output", "_upper_only"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def find_matching_audio(npz_path: Path, audio_dir: Path) -> Path:
    clip_name = normalize_clip_name(npz_path.stem)
    candidate = audio_dir / f"{clip_name}.wav"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No matching audio for {npz_path.name} under {audio_dir}")


def build_render_job(npz_path: Path, audio_dir: Path, output_root: Path, log_root: Path) -> RenderJob:
    clip_name = normalize_clip_name(npz_path.stem)
    audio_path = find_matching_audio(npz_path, audio_dir)
    output_dir = output_root / clip_name
    work_dir = output_dir / "work"
    return RenderJob(
        clip_name=clip_name,
        npz_path=npz_path,
        audio_path=audio_path,
        output_dir=output_dir,
        preview_path=output_dir / "preview.png",
        video_path=output_dir / f"{clip_name}.mp4",
        work_dir=work_dir,
        silent_video_path=work_dir / "silent.mp4",
        log_path=log_root / f"{clip_name}.log",
    )


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render EMAGE .npz clips to Blender SMPL-X videos with audio.")
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=workspace_root / "third_party" / "blender" / "blender-4.2.2-linux-x64" / "blender",
    )
    parser.add_argument(
        "--blender-script",
        type=Path,
        default=workspace_root / "tools" / "blender" / "render_emage_smplx_scene.py",
    )
    parser.add_argument(
        "--addon-root",
        type=Path,
        default=workspace_root / "third_party" / "PantoMatrix" / "blender_assets" / "unpacked" / "smplx_addon",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=workspace_root / "data" / "emage_outputs",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=workspace_root / "data" / "audio",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspace_root / "data" / "blender_renders_smplx",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=workspace_root / "demo_logs" / "blender_smplx",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=Path,
        default=None,
        help="Optional explicit ffmpeg binary. Defaults to PATH or known local conda environments.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", choices=("AUTO", "GPU", "CPU"), default="AUTO")
    parser.add_argument(
        "--engine",
        choices=("AUTO",) + RENDER_ENGINES,
        default="AUTO",
        help="AUTO tries Cycles first and falls back to Eevee Next.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        action="append",
        default=[],
        help="Optional explicit .npz path(s). Defaults to all *.npz under --npz-dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_jobs(args: argparse.Namespace) -> list[RenderJob]:
    npz_paths = args.npz or sorted(args.npz_dir.glob("*.npz"))
    jobs = []
    for npz_path in npz_paths:
        jobs.append(build_render_job(npz_path, args.audio_dir, args.output_root, args.log_root))
    return jobs


def run_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout)
    return completed


def render_with_engine(args: argparse.Namespace, job: RenderJob, engine: str) -> subprocess.CompletedProcess[str]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.blender_bin),
        "-b",
        "--python",
        str(args.blender_script),
        "--",
        "--npz",
        str(job.npz_path),
        "--addon-root",
        str(args.addon_root),
        "--preview-path",
        str(job.preview_path),
        "--silent-video-path",
        str(job.silent_video_path),
        "--engine",
        engine,
        "--device",
        args.device,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    return run_command(command, job.log_path)


def mux_audio(job: RenderJob) -> None:
    ffmpeg_bin = resolve_binary("ffmpeg", FFMPEG_FALLBACKS)
    command = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(job.silent_video_path),
        "-i",
        str(job.audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(job.video_path),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with job.log_path.open("a") as handle:
        handle.write("\n\n--- ffmpeg mux ---\n")
        handle.write(completed.stdout)
    completed.check_returncode()


def render_job(args: argparse.Namespace, job: RenderJob) -> None:
    engines = [args.engine] if args.engine != "AUTO" else list(RENDER_ENGINES)
    last_error: subprocess.CompletedProcess[str] | None = None
    for engine in engines:
        completed = render_with_engine(args, job, engine)
        if completed.returncode == 0:
            mux_audio(job)
            return
        last_error = completed
        with job.log_path.open("a") as handle:
            handle.write(f"\n\n[wrapper] engine {engine} failed with code {completed.returncode}\n")
    if last_error is not None:
        raise RuntimeError(f"Blender render failed for {job.clip_name}; see {job.log_path}")


def main() -> int:
    args = parse_args()
    if args.ffmpeg_bin is not None:
        global FFMPEG_FALLBACKS
        FFMPEG_FALLBACKS = (args.ffmpeg_bin,) + FFMPEG_FALLBACKS
    jobs = collect_jobs(args)
    if not jobs:
        print(f"[warn] no .npz files found under {args.npz_dir}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for job in jobs:
        print(f"[render] {job.clip_name}")
        print(f"  npz={job.npz_path}")
        print(f"  wav={job.audio_path}")
        print(f"  out={job.video_path}")
        if args.dry_run:
            continue
        try:
            render_job(args, job)
        except Exception as exc:  # pragma: no cover - end-to-end surface
            failures.append(f"{job.clip_name}: {exc}")
            print(f"[fail] {job.clip_name}: {exc}", file=sys.stderr)

    if failures:
        print("[summary] failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
