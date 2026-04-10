from __future__ import annotations

import json
import zipfile
from pathlib import Path

from conftest import load_tool_module


def test_build_face_render_package_writes_expected_zip_layout(tmp_path: Path):
    showcase = load_tool_module("hybrid_showcase_video_test", "render_hybrid_showcase_video.py")

    asset_root = tmp_path / "status" / "arkitWithBSData"
    asset_root.mkdir(parents=True)
    for name in ["animation.glb", "skin.glb", "offset.ply", "vertex_order.json"]:
        (asset_root / name).write_text(name, encoding="utf-8")

    bsdata = tmp_path / "bsData.json"
    bsdata.write_text(json.dumps({"metadata": {"frame_count": 2}, "frames": []}), encoding="utf-8")
    zip_path = tmp_path / "status_hybrid.zip"

    showcase.build_face_render_package(
        base_asset_dir=asset_root,
        bsdata_json=bsdata,
        zip_path=zip_path,
        package_name="status_hybrid",
    )

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())

    assert "status/arkitWithBSData/animation.glb" in names
    assert "status/arkitWithBSData/skin.glb" in names
    assert "status/arkitWithBSData/bsData.json" in names


def test_build_ffmpeg_stack_command_uses_hstack_and_audio_map():
    showcase = load_tool_module("hybrid_showcase_video_test", "render_hybrid_showcase_video.py")

    command = showcase.build_ffmpeg_stack_command(
        body_video=Path("/tmp/body.mp4"),
        face_video=Path("/tmp/face.mp4"),
        audio_file=Path("/tmp/audio.wav"),
        output_video=Path("/tmp/out.mp4"),
    )

    joined = " ".join(str(part) for part in command)
    assert command[0] == "ffmpeg"
    assert "hstack=inputs=2" in joined
    assert "-map 2:a:0" in joined
    assert str(Path("/tmp/out.mp4")) in joined


def test_build_face_video_command_normalizes_to_even_dimensions():
    showcase = load_tool_module("hybrid_showcase_video_test", "render_hybrid_showcase_video.py")

    command = showcase.build_face_video_command(
        frames_pattern=Path("/tmp/frames/frame_%05d.png"),
        fps=15,
        output_video=Path("/tmp/face.mp4"),
    )

    joined = " ".join(str(part) for part in command)
    assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in joined


def test_default_render_mode_prefers_coeff2d_for_a2f_provider():
    showcase = load_tool_module("hybrid_showcase_video_test", "render_hybrid_showcase_video.py")

    args = showcase.parse_args(
        [
            "--audio",
            "/tmp/audio.wav",
            "--face-json",
            "/tmp/face.json",
            "--output",
            "/tmp/out.mp4",
            "--face-provider",
            "a2f-3d-sdk",
        ]
    )

    assert showcase.resolve_render_mode(args.face_provider, args.render_mode) == "coeff2d"


def test_default_render_mode_keeps_browser_for_lam_provider():
    showcase = load_tool_module("hybrid_showcase_video_test", "render_hybrid_showcase_video.py")

    args = showcase.parse_args(
        [
            "--audio",
            "/tmp/audio.wav",
            "--face-json",
            "/tmp/face.json",
            "--output",
            "/tmp/out.mp4",
            "--face-provider",
            "lam",
        ]
    )

    assert showcase.resolve_render_mode(args.face_provider, args.render_mode) == "browser"


def test_capture_schedule_matches_duration_and_fps():
    showcase = load_tool_module("hybrid_showcase_video_test", "render_hybrid_showcase_video.py")

    schedule = showcase.build_capture_schedule(duration_sec=1.0, fps=10)

    assert len(schedule) == 10
    assert schedule[0] == 0.0
    assert schedule[-1] == 0.9


def test_build_xvfb_command_uses_expected_screen_config():
    showcase = load_tool_module("hybrid_showcase_video_test", "render_hybrid_showcase_video.py")

    command = showcase.build_xvfb_command(display=":99", width=1920, height=1080, depth=24)

    assert command == [
        "Xvfb",
        ":99",
        "-screen",
        "0",
        "1920x1080x24",
    ]


def test_find_available_x_display_skips_taken_slots(tmp_path, monkeypatch):
    showcase = load_tool_module("hybrid_showcase_video_test", "render_hybrid_showcase_video.py")
    x11_dir = tmp_path / ".X11-unix"
    x11_dir.mkdir()
    (x11_dir / "X99").write_text("", encoding="utf-8")
    (x11_dir / "X100").write_text("", encoding="utf-8")
    monkeypatch.setattr(showcase, "X11_SOCKET_DIR", x11_dir)

    assert showcase.find_available_x_display(start=99, end=103) == ":101"
