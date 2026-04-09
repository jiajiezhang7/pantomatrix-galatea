from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import load_tool_module


def _write_face_json(path: Path, frame_count: int, fps: float = 30.0) -> Path:
    frames = []
    for frame in range(frame_count):
        frames.append(
            {
                "frame": frame,
                "time_sec": frame / fps,
                "blendshapes": {
                    "jawOpen": min(1.0, frame / max(1, frame_count - 1)),
                    "eyeBlinkLeft": 0.25,
                    "eyeBlinkRight": 0.5,
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "source_audio": "demo.wav",
                "fps": fps,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_body_npz(path: Path, frame_count: int, fps: float = 30.0) -> Path:
    np.savez(
        path,
        betas=np.zeros(300, dtype=np.float32),
        poses=np.zeros((frame_count, 165), dtype=np.float32),
        expressions=np.zeros((frame_count, 100), dtype=np.float32),
        trans=np.zeros((frame_count, 3), dtype=np.float32),
        model=np.array("smplx"),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(fps),
    )
    return path


def test_build_canonical_timeline_uses_audio_duration_and_target_fps():
    hybrid = load_tool_module("hybrid_pipeline_test", "hybrid_pipeline.py")

    timeline = hybrid.build_canonical_timeline(duration_sec=1.0, target_fps=30.0)

    assert len(timeline) == 30
    assert timeline[0] == pytest.approx(0.0)
    assert timeline[-1] == pytest.approx(29.0 / 30.0)


def test_align_face_frames_resamples_to_requested_timeline():
    hybrid = load_tool_module("hybrid_pipeline_test", "hybrid_pipeline.py")

    source_frames = [
        {"frame": 0, "time_sec": 0.0, "blendshapes": {"jawOpen": 0.0}},
        {"frame": 1, "time_sec": 0.5, "blendshapes": {"jawOpen": 0.5}},
        {"frame": 2, "time_sec": 1.0, "blendshapes": {"jawOpen": 1.0}},
    ]

    aligned = hybrid.align_face_frames(source_frames, [0.0, 0.25, 0.5, 0.75], blendshape_names=["jawOpen"])

    assert [frame["frame"] for frame in aligned] == [0, 1, 2, 3]
    assert [frame["time_sec"] for frame in aligned] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert aligned[1]["blendshapes"]["jawOpen"] == pytest.approx(0.25)
    assert aligned[-1]["blendshapes"]["jawOpen"] == pytest.approx(0.75)


def test_pipeline_writes_hybrid_result_contract_without_separate_eye_lane(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_test", "hybrid_pipeline.py")

    audio_path = tmp_path / "demo.wav"
    body_npz = _write_body_npz(tmp_path / "body_source.npz", frame_count=30)
    face_json = _write_face_json(tmp_path / "face_source.json", frame_count=24, fps=24.0)
    audio_path.write_bytes(b"RIFFdemo")

    exit_code = hybrid.main(
        [
            "--audio",
            str(audio_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--body-npz",
            str(body_npz),
            "--face-input-json",
            str(face_json),
        ]
    )

    assert exit_code == 0

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    face_payload = json.loads((tmp_path / "run" / "face" / "arkit_blendshapes.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "run" / "metrics" / "run_metrics.json").read_text(encoding="utf-8"))

    assert manifest["frame_count"] == 30
    assert manifest["components"]["emage_body"]["status"] == "ready"
    assert manifest["components"]["face_provider"]["status"] == "ready"
    assert manifest["component_aliases"]["lam_face"] == "face_provider"
    assert set(manifest["components"]) == {"emage_body", "face_provider"}
    assert "eye" not in manifest["sources"]
    assert face_payload["fps"] == pytest.approx(30.0)
    assert len(face_payload["frames"]) == 30
    assert metrics["timings_sec"]["alignment"] >= 0.0


def test_pipeline_logs_include_face_provider_only(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_test", "hybrid_pipeline.py")

    audio_path = tmp_path / "demo.wav"
    body_npz = _write_body_npz(tmp_path / "body_source.npz", frame_count=12)
    face_json = _write_face_json(tmp_path / "face_source.json", frame_count=12)
    audio_path.write_bytes(b"RIFFdemo")

    exit_code = hybrid.main(
        [
            "--audio",
            str(audio_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--body-npz",
            str(body_npz),
            "--face-input-json",
            str(face_json),
        ]
    )

    assert exit_code == 0

    log_text = (tmp_path / "run" / "logs" / "pipeline.log").read_text(encoding="utf-8")
    metrics = json.loads((tmp_path / "run" / "metrics" / "run_metrics.json").read_text(encoding="utf-8"))
    assert "face_provider=lam" in log_text
    assert "body_status=ready" in log_text
    assert set(metrics) == {"timings_sec", "frame_count"}


def test_face_provider_registry_keeps_lam_swappable():
    hybrid = load_tool_module("hybrid_pipeline_test", "hybrid_pipeline.py")

    provider_config = hybrid.FaceProviderConfig(
        input_json=Path("/tmp/lam.json"),
        command="",
        python_bin="python",
        repo_root=None,
        output_json_name="arkit_blendshapes.json",
    )

    adapter = hybrid.build_face_provider(
        provider="lam",
        provider_config=provider_config,
        audio_path=Path("/tmp/demo.wav"),
    )

    assert adapter.provider_name == "lam"
    assert adapter.provider_config.output_json_name == "arkit_blendshapes.json"

    with pytest.raises(SystemExit, match="Unsupported face provider"):
        hybrid.build_face_provider(
            provider="nvidia-a2f-3d",
            provider_config=hybrid.FaceProviderConfig(
                input_json=None,
                command="",
                python_bin="python",
                repo_root=None,
                output_json_name="arkit_blendshapes.json",
            ),
            audio_path=Path("/tmp/demo.wav"),
        )


def test_build_face_provider_config_prefers_generic_option_names():
    hybrid = load_tool_module("hybrid_pipeline_test", "hybrid_pipeline.py")

    args = hybrid.parse_args(
        [
            "--audio",
            "/tmp/demo.wav",
            "--output-dir",
            "/tmp/run",
            "--face-input-json",
            "/tmp/face.json",
            "--face-command",
            "run-face",
            "--face-python",
            "python3.10",
            "--face-repo",
            "/tmp/provider-repo",
            "--face-output-json-name",
            "face.json",
        ]
    )
    config = hybrid.build_face_provider_config(args)

    assert config.input_json == Path("/tmp/face.json")
    assert config.command == "run-face"
    assert config.python_bin == "python3.10"
    assert config.repo_root == Path("/tmp/provider-repo")
    assert config.output_json_name == "face.json"


def test_build_face_provider_config_accepts_legacy_lam_aliases():
    hybrid = load_tool_module("hybrid_pipeline_test", "hybrid_pipeline.py")

    args = hybrid.parse_args(
        [
            "--audio",
            "/tmp/demo.wav",
            "--output-dir",
            "/tmp/run",
            "--lam-json",
            "/tmp/lam-face.json",
            "--lam-command",
            "run-lam",
            "--lam-python",
            "python3.10",
            "--lam-repo",
            "/tmp/lam-repo",
            "--lam-output-json-name",
            "lam-face.json",
        ]
    )
    config = hybrid.build_face_provider_config(args)

    assert config.input_json == Path("/tmp/lam-face.json")
    assert config.command == "run-lam"
    assert config.python_bin == "python3.10"
    assert config.repo_root == Path("/tmp/lam-repo")
    assert config.output_json_name == "lam-face.json"
