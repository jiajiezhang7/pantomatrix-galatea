#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

X11_SOCKET_DIR = Path("/tmp/.X11-unix")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render a side-by-side hybrid body + face-provider showcase mp4.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--face-json", type=Path, required=True, help="Face-provider JSON payload.")
    parser.add_argument("--body-video", type=Path, default=None)
    parser.add_argument("--avatar-dir", type=Path, default=workspace_root / "third_party" / "LAM_Audio2Expression" / "assets" / "sample_lam" / "status" / "arkitWithBSData")
    parser.add_argument("--lam-python", type=str, default="/home/vanto/anaconda3/envs/lam-a2e310/bin/python")
    parser.add_argument("--preview-port", type=int, default=7861)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-provider", choices=("lam", "a2f-3d-sdk"), default="lam")
    parser.add_argument("--geckodriver", type=str, default="/snap/bin/geckodriver")
    parser.add_argument("--firefox-binary", type=str, default="/usr/bin/firefox")
    parser.add_argument("--face-width", type=int, default=420)
    parser.add_argument("--face-height", type=int, default=760)
    parser.add_argument("--render-mode", choices=("auto", "browser", "coeff2d"), default="auto")
    return parser.parse_args(argv)


def resolve_render_mode(face_provider: str, render_mode: str) -> str:
    if render_mode != "auto":
        return render_mode
    if face_provider == "a2f-3d-sdk":
        return "coeff2d"
    return "browser"


def build_face_render_package(
    *,
    base_asset_dir: Path,
    bsdata_json: Path,
    zip_path: Path,
    package_name: str,
) -> None:
    required_assets = ["animation.glb", "skin.glb", "offset.ply", "vertex_order.json"]
    package_root = base_asset_dir.parent.name
    zip_exe = shutil.which("zip")
    if zip_exe:
        with tempfile.TemporaryDirectory(prefix="face_render_pkg_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            package_dir = temp_dir / package_root / "arkitWithBSData"
            package_dir.mkdir(parents=True, exist_ok=True)
            for asset_name in required_assets:
                shutil.copy2(base_asset_dir / asset_name, package_dir / asset_name)
            shutil.copy2(bsdata_json, package_dir / "bsData.json")
            if zip_path.exists():
                zip_path.unlink()
            subprocess.run(
                [zip_exe, "-r", str(zip_path), package_root],
                check=True,
                cwd=temp_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for asset_name in required_assets:
            asset_path = base_asset_dir / asset_name
            arcname = f"{package_root}/arkitWithBSData/{asset_name}"
            info = zipfile.ZipInfo(filename=arcname, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, asset_path.read_bytes())

        info = zipfile.ZipInfo(
            filename=f"{package_root}/arkitWithBSData/bsData.json",
            date_time=(2026, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, bsdata_json.read_bytes())


def build_capture_schedule(duration_sec: float, fps: int) -> list[float]:
    frame_count = max(1, int(round(duration_sec * fps)))
    return [frame_idx / fps for frame_idx in range(frame_count)]


def build_ffmpeg_stack_command(
    *,
    body_video: Path,
    face_video: Path,
    audio_file: Path,
    output_video: Path,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(body_video),
        "-i",
        str(face_video),
        "-i",
        str(audio_file),
        "-filter_complex",
        "[0:v]scale=-2:720[body];[1:v]scale=-2:720[face];[body][face]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-map",
        "2:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(output_video),
    ]


def build_face_video_command(*, frames_pattern: Path, fps: int, output_video: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_pattern),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]


def build_xvfb_command(*, display: str, width: int, height: int, depth: int) -> list[str]:
    return [
        "Xvfb",
        display,
        "-screen",
        "0",
        f"{width}x{height}x{depth}",
    ]


def find_available_x_display(*, start: int = 99, end: int = 199) -> str:
    for display_num in range(start, end):
        if not (X11_SOCKET_DIR / f"X{display_num}").exists():
            return f":{display_num}"
    raise RuntimeError("No available X display slots found")


def _find_body_video(audio_path: Path, workspace_root: Path) -> Path:
    stem = audio_path.stem
    matches = sorted((workspace_root / "data" / "rendered_videos").rglob(f"{stem}_output.mp4"))
    if not matches:
        raise FileNotFoundError(f"No rendered body video found for {stem}")
    return matches[0]


def _load_duration(face_json: Path) -> float:
    payload = json.loads(face_json.read_text(encoding="utf-8"))
    if "metadata" in payload:
        frame_count = int(payload["metadata"]["frame_count"])
        fps = float(payload["metadata"]["fps"])
    else:
        frame_count = int(payload["frame_count"])
        fps = float(payload["fps"])
    return frame_count / fps


def _run_ffmpeg(command: list[str]) -> None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg_exe, *command[1:]]
    subprocess.run(command, check=True)


def _capture_face_frames(
    *,
    app_url: str,
    zip_url: str,
    schedule: list[float],
    screenshot_dir: Path,
    geckodriver_path: str,
    firefox_binary: str,
    width: int,
    height: int,
) -> None:
    import os
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    options = Options()
    options.add_argument("--new-instance")
    browser_env = os.environ.copy()
    display = find_available_x_display()
    xvfb_cmd = build_xvfb_command(display=display, width=1920, height=1080, depth=24)
    xvfb_exe = shutil.which(xvfb_cmd[0])
    if xvfb_exe is None:
        raise RuntimeError("Xvfb is required for browser face rendering on this server")
    xvfb_cmd[0] = xvfb_exe
    xvfb_proc = subprocess.Popen(xvfb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    browser_env["DISPLAY"] = display
    browser_env["GDK_BACKEND"] = "x11"
    browser_env.pop("MOZ_HEADLESS", None)
    browser_env.pop("MOZ_WEBRENDER", None)
    service = Service(executable_path=geckodriver_path, env=browser_env)
    try:
        time.sleep(2)
        driver = webdriver.Firefox(service=service, options=options)
    except Exception:
        xvfb_proc.terminate()
        try:
            xvfb_proc.wait(timeout=5)
        except Exception:
            xvfb_proc.kill()
        raise
    driver.set_window_size(width, height)
    try:
        driver.get(app_url)
        time.sleep(8)
        driver.execute_script("window.start(arguments[0]);", zip_url)
        time.sleep(8)
        render_state = driver.execute_script(
            "return {renderChildren: document.getElementById('render-root') ? document.getElementById('render-root').children.length : -1, canvases: document.getElementsByTagName('canvas').length};"
        )
        if render_state["renderChildren"] <= 0 and render_state["canvases"] <= 0:
            raise RuntimeError("Browser face renderer did not create any render nodes")
        start = time.monotonic()
        for frame_idx, target_time in enumerate(schedule):
            while True:
                now = time.monotonic() - start
                if now >= target_time:
                    break
                time.sleep(min(0.01, target_time - now))
            driver.save_screenshot(str(screenshot_dir / f"frame_{frame_idx:05d}.png"))
    finally:
        driver.quit()
        xvfb_proc.terminate()
        try:
            xvfb_proc.wait(timeout=5)
        except Exception:
            xvfb_proc.kill()


def _build_face_video(frames_dir: Path, fps: int, output_video: Path) -> None:
    command = build_face_video_command(
        frames_pattern=frames_dir / "frame_%05d.png",
        fps=fps,
        output_video=output_video,
    )
    _run_ffmpeg(command)


def _load_face_frames(face_json: Path) -> tuple[list[dict], float]:
    payload = json.loads(face_json.read_text(encoding="utf-8"))
    if payload.get("frames") and payload["frames"] and "blendshapes" in payload["frames"][0]:
        return payload["frames"], float(payload["fps"])

    names = payload["names"]
    fps = float(payload["metadata"]["fps"])
    frames = []
    for idx, frame in enumerate(payload["frames"]):
        frames.append(
            {
                "frame": idx,
                "time_sec": float(frame["time"]),
                "blendshapes": {name: float(value) for name, value in zip(names, frame["weights"])},
            }
        )
    return frames, fps


def _draw_face_frame(blendshapes: dict[str, float], width: int, height: int, provider_label: str):
    import cv2
    import numpy as np

    img = np.full((height, width, 3), 250, dtype=np.uint8)
    center = (width // 2, height // 2)
    face_radius_x = width // 3
    face_radius_y = int(height * 0.38)
    cv2.ellipse(img, center, (face_radius_x, face_radius_y), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img, center, (face_radius_x, face_radius_y), 0, 0, 360, (60, 60, 60), 3)

    eye_y = int(height * 0.38)
    eye_x_offset = width // 7
    eye_w = width // 12
    blink_l = float(blendshapes.get("eyeBlinkLeft", 0.0))
    blink_r = float(blendshapes.get("eyeBlinkRight", 0.0))
    eye_h_l = max(2, int((1.0 - min(1.0, blink_l)) * height * 0.018))
    eye_h_r = max(2, int((1.0 - min(1.0, blink_r)) * height * 0.018))

    look_x = (
        float(blendshapes.get("eyeLookOutRight", 0.0))
        - float(blendshapes.get("eyeLookInRight", 0.0))
        + float(blendshapes.get("eyeLookInLeft", 0.0))
        - float(blendshapes.get("eyeLookOutLeft", 0.0))
    ) * width * 0.01
    look_y = (
        float(blendshapes.get("eyeLookDownLeft", 0.0))
        + float(blendshapes.get("eyeLookDownRight", 0.0))
        - float(blendshapes.get("eyeLookUpLeft", 0.0))
        - float(blendshapes.get("eyeLookUpRight", 0.0))
    ) * height * 0.01

    left_eye = (center[0] - eye_x_offset, eye_y)
    right_eye = (center[0] + eye_x_offset, eye_y)
    cv2.ellipse(img, left_eye, (eye_w, eye_h_l), 0, 0, 360, (40, 40, 40), 2)
    cv2.ellipse(img, right_eye, (eye_w, eye_h_r), 0, 0, 360, (40, 40, 40), 2)
    cv2.circle(img, (int(left_eye[0] + look_x), int(left_eye[1] + look_y)), max(2, eye_w // 4), (30, 30, 30), -1)
    cv2.circle(img, (int(right_eye[0] + look_x), int(right_eye[1] + look_y)), max(2, eye_w // 4), (30, 30, 30), -1)

    brow_inner = float(blendshapes.get("browInnerUp", 0.0))
    brow_up_l = float(blendshapes.get("browOuterUpLeft", 0.0)) - float(blendshapes.get("browDownLeft", 0.0))
    brow_up_r = float(blendshapes.get("browOuterUpRight", 0.0)) - float(blendshapes.get("browDownRight", 0.0))
    brow_base_y = int(height * 0.30 - brow_inner * height * 0.02)
    cv2.line(
        img,
        (left_eye[0] - eye_w, int(brow_base_y - brow_up_l * height * 0.02)),
        (left_eye[0] + eye_w, int(brow_base_y + brow_up_l * height * 0.01)),
        (50, 50, 50),
        3,
    )
    cv2.line(
        img,
        (right_eye[0] - eye_w, int(brow_base_y + brow_up_r * height * 0.01)),
        (right_eye[0] + eye_w, int(brow_base_y - brow_up_r * height * 0.02)),
        (50, 50, 50),
        3,
    )

    nose_y = int(height * 0.50)
    cv2.line(img, (center[0], eye_y + 25), (center[0], nose_y), (90, 90, 90), 2)

    jaw_open = float(blendshapes.get("jawOpen", 0.0))
    mouth_pucker = float(blendshapes.get("mouthPucker", 0.0))
    smile = (
        float(blendshapes.get("mouthSmileLeft", 0.0))
        + float(blendshapes.get("mouthSmileRight", 0.0))
        - float(blendshapes.get("mouthFrownLeft", 0.0))
        - float(blendshapes.get("mouthFrownRight", 0.0))
    ) * 0.5
    mouth_shift = (float(blendshapes.get("mouthRight", 0.0)) - float(blendshapes.get("mouthLeft", 0.0))) * width * 0.03
    mouth_y = int(height * 0.65)
    mouth_w = max(20, int(width * (0.16 - mouth_pucker * 0.07)))
    mouth_h = max(4, int(height * (0.01 + jaw_open * 0.05)))
    mouth_center = (int(center[0] + mouth_shift), mouth_y)
    curvature = int(smile * height * 0.05)
    pts = np.array(
        [
            [mouth_center[0] - mouth_w, mouth_y + curvature],
            [mouth_center[0], mouth_y - curvature],
            [mouth_center[0] + mouth_w, mouth_y + curvature],
        ],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], False, (40, 40, 40), 4)
    if mouth_h > 6:
        cv2.ellipse(img, mouth_center, (mouth_w // 2, mouth_h), 0, 0, 360, (80, 20, 20), 2)

    cv2.putText(img, provider_label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    return img


def _build_face_video_from_coeffs(
    face_json: Path,
    output_video: Path,
    width: int,
    height: int,
    fps: int,
    provider_label: str,
) -> None:
    import cv2

    frames, source_fps = _load_face_frames(face_json)
    duration_sec = len(frames) / max(source_fps, 1.0)
    schedule = build_capture_schedule(duration_sec, fps)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {output_video}")
    try:
        for target_time in schedule:
            frame_idx = min(len(frames) - 1, int(round(target_time * source_fps)))
            frame = frames[frame_idx]
            image = _draw_face_frame(frame["blendshapes"], width, height, provider_label)
            writer.write(image)
    finally:
        writer.release()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace_root = Path(__file__).resolve().parents[1]
    body_video = args.body_video or _find_body_video(args.audio, workspace_root)
    duration_sec = _load_duration(args.face_json)
    capture_schedule = build_capture_schedule(duration_sec=duration_sec, fps=args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chosen_render_mode = resolve_render_mode(args.face_provider, args.render_mode)
    provider_label = "A2F-3D Face" if args.face_provider == "a2f-3d-sdk" else "LAM Face"

    with tempfile.TemporaryDirectory(prefix="hybrid_showcase_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        assets_dir = temp_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        package_name = "status_hybrid"
        zip_path = assets_dir / f"{package_name}.zip"
        build_face_render_package(
            base_asset_dir=args.avatar_dir,
            bsdata_json=args.face_json,
            zip_path=zip_path,
            package_name=package_name,
        )

        app_log = temp_dir / "preview.log"
        app_command = [
            args.lam_python,
            str(workspace_root / "tools" / "lam_face_preview_app.py"),
            "--assets-dir",
            str(assets_dir),
            "--zip-name",
            zip_path.name,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.preview_port),
            "--width",
            str(args.face_width),
            "--height",
            str(args.face_height),
        ]
        face_video = temp_dir / "face.mp4"
        if chosen_render_mode == "browser":
            app_proc = subprocess.Popen(app_command, stdout=app_log.open("w"), stderr=subprocess.STDOUT)
            screenshot_dir = temp_dir / "face_frames"
            try:
                app_url = f"http://127.0.0.1:{args.preview_port}"
                zip_url = f"gradio_api/file={zip_path}"
                time.sleep(8)
                _capture_face_frames(
                    app_url=app_url,
                    zip_url=zip_url,
                    schedule=capture_schedule,
                    screenshot_dir=screenshot_dir,
                    geckodriver_path=args.geckodriver,
                    firefox_binary=args.firefox_binary,
                    width=args.face_width,
                    height=args.face_height,
                )
                _build_face_video(screenshot_dir, args.fps, face_video)
            except Exception:
                if args.render_mode == "browser":
                    raise
                _build_face_video_from_coeffs(
                    args.face_json,
                    face_video,
                    args.face_width,
                    args.face_height,
                    args.fps,
                    provider_label,
                )
            finally:
                app_proc.terminate()
                try:
                    app_proc.wait(timeout=5)
                except Exception:
                    app_proc.kill()
                    app_proc.wait(timeout=5)
        else:
            _build_face_video_from_coeffs(
                args.face_json,
                face_video,
                args.face_width,
                args.face_height,
                args.fps,
                provider_label,
            )

        command = build_ffmpeg_stack_command(
            body_video=body_video,
            face_video=face_video,
            audio_file=args.audio,
            output_video=args.output,
        )
        _run_ffmpeg(command)

    print(f"[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
