from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_tool_module


def _write_raw_a2f_json(path: Path) -> Path:
    payload = {
        "provider": "a2f-3d-sdk",
        "metadata": {
            "fps": 60.0,
            "frame_count": 2,
            "blendshape_names": ["jawOpen", "mouthClose", "eyeBlinkLeft"],
            "normalization_policy": "raw-a2f",
            "notes": [
                "Eye look coefficients may be omitted or remain zero in raw A2F output.",
                "MouthClose follows NVIDIA semantics.",
            ],
        },
        "frames": [
            {
                "time_sec": 0.0,
                "blendshapes": {"jawOpen": 0.1, "mouthClose": 0.2, "eyeBlinkLeft": 0.0},
            },
            {
                "time_sec": 1.0 / 60.0,
                "blendshapes": {"jawOpen": 0.4, "mouthClose": 0.5, "eyeBlinkLeft": 0.0},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_convert_raw_a2f_json_to_hybrid_contract_preserves_coefficients(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_a2f_test", "hybrid_pipeline.py")
    source = _write_raw_a2f_json(tmp_path / "a2f_raw.json")

    converted = hybrid.convert_a2f3d_sdk_json_to_face_payload(source)

    assert converted["fps"] == pytest.approx(60.0)
    assert converted["frame_count"] == 2
    assert converted["names"] == ["eyeBlinkLeft", "jawOpen", "mouthClose"]
    assert converted["frames"][0]["blendshapes"]["jawOpen"] == pytest.approx(0.1)
    assert converted["frames"][0]["blendshapes"]["mouthClose"] == pytest.approx(0.2)
    assert converted["frames"][0]["blendshapes"]["eyeBlinkLeft"] == pytest.approx(0.0)


def test_convert_raw_a2f_json_does_not_fabricate_missing_eye_channels(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_a2f_test", "hybrid_pipeline.py")
    source = tmp_path / "a2f_raw_partial.json"
    source.write_text(
        json.dumps(
            {
                "provider": "a2f-3d-sdk",
                "metadata": {
                    "fps": 30.0,
                    "frame_count": 1,
                    "blendshape_names": ["jawOpen"],
                },
                "frames": [{"time_sec": 0.0, "blendshapes": {"jawOpen": 0.25}}],
            }
        ),
        encoding="utf-8",
    )

    converted = hybrid.convert_a2f3d_sdk_json_to_face_payload(source)

    assert converted["names"] == ["jawOpen"]
    assert converted["eye_blendshape_names"] == []
    assert set(converted["frames"][0]["blendshapes"]) == {"jawOpen"}


def test_build_a2f3d_sdk_command_uses_wrapper_entrypoint(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_a2f_test", "hybrid_pipeline.py")
    config_file = tmp_path / "a2f-config.json"
    config_file.write_text("{}", encoding="utf-8")

    provider_config = hybrid.FaceProviderConfig(
        input_json=None,
        command="",
        python_bin="python3.10",
        repo_root=Path("/tmp/a2f-sdk"),
        output_json_name="a2f_raw.json",
        config_file=config_file,
    )

    command, output_json = hybrid.build_a2f3d_sdk_command(
        provider_config=provider_config,
        audio_path=Path("/tmp/demo.wav"),
        output_dir=tmp_path / "run",
    )

    assert command[0] == "python3.10"
    assert command[1].endswith("run_a2f3d_sdk_inference.py")
    assert "--config-file" in command
    assert str(config_file) in command
    assert "--audio" in command
    assert str(Path("/tmp/demo.wav").resolve()) in command
    assert output_json == tmp_path / "run" / "a2f_raw.json"


def test_a2f3d_adapter_requires_face_config_file(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_a2f_test", "hybrid_pipeline.py")

    adapter = hybrid.A2F3DSdkFaceAdapter(
        provider_config=hybrid.FaceProviderConfig(
            input_json=None,
            command="",
            python_bin="python",
            repo_root=Path("/tmp/a2f-sdk"),
            output_json_name="a2f_raw.json",
            config_file=None,
        ),
        audio_path=Path("/tmp/demo.wav"),
    )

    with pytest.raises(SystemExit, match="A2F-3D SDK run requires --face-config-file"):
        adapter.run(
            args=hybrid.parse_args(["--audio", "/tmp/demo.wav", "--output-dir", str(tmp_path / "out")]),
            scratch_dir=tmp_path,
        )


def test_a2f3d_adapter_converts_raw_output_and_preserves_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hybrid = load_tool_module("hybrid_pipeline_a2f_test", "hybrid_pipeline.py")
    config_file = tmp_path / "a2f-config.json"
    config_file.write_text("{}", encoding="utf-8")

    provider_config = hybrid.FaceProviderConfig(
        input_json=None,
        command="",
        python_bin="python3.10",
        repo_root=Path("/tmp/a2f-sdk"),
        output_json_name="a2f_raw.json",
        config_file=config_file,
    )
    adapter = hybrid.A2F3DSdkFaceAdapter(provider_config=provider_config, audio_path=Path("/tmp/demo.wav"))

    def fake_run(command, check, cwd=None):
        assert check is True
        output_path = Path(command[command.index("--output-json") + 1])
        _write_raw_a2f_json(output_path)
        return None

    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)
    result = adapter.run(
        args=hybrid.parse_args(
            [
                "--audio",
                "/tmp/demo.wav",
                "--output-dir",
                str(tmp_path / "run"),
                "--face-provider",
                "a2f-3d-sdk",
                "--face-config-file",
                str(config_file),
            ]
        ),
        scratch_dir=tmp_path / "scratch",
    )

    payload = json.loads(result.payload_path.read_text(encoding="utf-8"))
    assert result.status == "ready"
    assert result.metadata["mode"] == "sdk-run"
    assert result.metadata["provider"] == "a2f-3d-sdk"
    assert Path(result.metadata["raw_output_json"]).name == "a2f_raw.json"
    assert payload["frames"][0]["blendshapes"]["mouthClose"] == pytest.approx(0.2)


def test_stage_profile_config_applies_json_overrides_and_writes_snapshot(tmp_path: Path):
    a2f = load_tool_module("a2f_sdk_runner_test", "run_a2f3d_sdk_inference.py")
    base_profile = tmp_path / "mark"
    base_profile.mkdir()
    (base_profile / "model.json").write_text(json.dumps({"modelConfigPath": "model_config.json"}), encoding="utf-8")
    (base_profile / "model_config.json").write_text(
        json.dumps({"config": {"input_strength": 1.3, "lower_face_smoothing": 0.0023}}),
        encoding="utf-8",
    )
    (base_profile / "bs_skin_config.json").write_text(
        json.dumps({"blendshape_params": {"strengthTemporalSmoothing": 0.3}}),
        encoding="utf-8",
    )

    config = {
        "profile_name": "tuned-regression-v1",
        "base_profile_dir": str(base_profile),
        "model_json_relpath": "model.json",
        "profile_overrides": {
            "model_config.json": {"config": {"input_strength": 1.0, "lower_face_smoothing": 0.01}},
            "bs_skin_config.json": {"blendshape_params": {"strengthTemporalSmoothing": 0.8}},
        },
    }

    resolved_config, snapshot_path = a2f.stage_profile_config(config, tmp_path / "run")

    staged_root = Path(resolved_config["staged_profile_dir"])
    staged_model_config = json.loads((staged_root / "model_config.json").read_text(encoding="utf-8"))
    staged_bs_skin_config = json.loads((staged_root / "bs_skin_config.json").read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert Path(resolved_config["model_json"]) == staged_root / "model.json"
    assert staged_model_config["config"]["input_strength"] == pytest.approx(1.0)
    assert staged_model_config["config"]["lower_face_smoothing"] == pytest.approx(0.01)
    assert staged_bs_skin_config["blendshape_params"]["strengthTemporalSmoothing"] == pytest.approx(0.8)
    assert snapshot["profile_name"] == "tuned-regression-v1"
    assert snapshot["base_profile_dir"] == str(base_profile)
