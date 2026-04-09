from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_tool_module


def _write_official_lam_json(path: Path) -> Path:
    payload = {
        "names": ["jawOpen", "eyeBlinkLeft", "eyeBlinkRight"],
        "metadata": {
            "fps": 30.0,
            "frame_count": 2,
            "blendshape_names": ["jawOpen", "eyeBlinkLeft", "eyeBlinkRight"],
        },
        "frames": [
            {"weights": [0.1, 0.2, 0.3], "time": 0.0, "rotation": []},
            {"weights": [0.4, 0.5, 0.6], "time": 1.0 / 30.0, "rotation": []},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_convert_official_lam_json_to_hybrid_contract(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_lam_test", "hybrid_pipeline.py")
    source = _write_official_lam_json(tmp_path / "lam_official.json")

    converted = hybrid.convert_lam_official_json_to_face_payload(source)

    assert converted["fps"] == pytest.approx(30.0)
    assert converted["frame_count"] == 2
    assert converted["frames"][0]["frame"] == 0
    assert converted["frames"][0]["time_sec"] == pytest.approx(0.0)
    assert converted["frames"][0]["blendshapes"] == {
        "jawOpen": pytest.approx(0.1),
        "eyeBlinkLeft": pytest.approx(0.2),
        "eyeBlinkRight": pytest.approx(0.3),
    }


def test_convert_official_lam_json_preserves_eye_related_blendshape_names(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_lam_test", "hybrid_pipeline.py")
    source = _write_official_lam_json(tmp_path / "lam_official.json")

    converted = hybrid.convert_lam_official_json_to_face_payload(source)

    assert "eyeBlinkLeft" in converted["names"]
    assert "eyeBlinkRight" in converted["names"]


def test_build_lam_official_command_uses_repo_defaults(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_lam_test", "hybrid_pipeline.py")
    repo_root = tmp_path / "LAM_Audio2Expression"
    (repo_root / "configs").mkdir(parents=True)
    (repo_root / "pretrained_models").mkdir(parents=True)

    provider_config = hybrid.FaceProviderConfig(
        input_json=None,
        command="",
        python_bin="python3.10",
        repo_root=repo_root,
        output_json_name="arkit_blendshapes.json",
        model_weight=None,
        config_file=None,
    )

    command, output_json = hybrid.build_lam_official_command(
        provider_config=provider_config,
        audio_path=Path("/tmp/demo.wav"),
        output_dir=tmp_path / "run",
    )

    assert command[0] == "python3.10"
    assert command[1] == str(repo_root / "inference.py")
    assert "--config-file" in command
    assert str(repo_root / "configs" / "lam_audio2exp_config_streaming.py") in command
    assert any("audio_input=/tmp/demo.wav" == item for item in command)
    assert any("ex_vol=False" == item for item in command)
    assert any(item.startswith("save_json_path=") for item in command)
    assert any(str(repo_root / "pretrained_models" / "lam_audio2exp_streaming.tar") in item for item in command)
    assert output_json == tmp_path / "run" / "arkit_blendshapes.json"


def test_build_lam_official_command_resolves_relative_audio_path(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_lam_test", "hybrid_pipeline.py")
    repo_root = tmp_path / "LAM_Audio2Expression"
    (repo_root / "configs").mkdir(parents=True)
    (repo_root / "pretrained_models").mkdir(parents=True)
    rel_audio = Path("data/audio/sample.wav")

    provider_config = hybrid.FaceProviderConfig(
        input_json=None,
        command="",
        python_bin="python3.10",
        repo_root=repo_root,
        output_json_name="arkit_blendshapes.json",
        model_weight=None,
        config_file=None,
    )

    command, _ = hybrid.build_lam_official_command(
        provider_config=provider_config,
        audio_path=rel_audio,
        output_dir=tmp_path / "run",
    )

    assert any(f"audio_input={rel_audio.resolve()}" == item for item in command)


def test_lam_adapter_official_run_requires_repo_root(tmp_path: Path):
    hybrid = load_tool_module("hybrid_pipeline_lam_test", "hybrid_pipeline.py")

    adapter = hybrid.LamFaceAdapter(
        provider_config=hybrid.FaceProviderConfig(
            input_json=None,
            command="",
            python_bin="python",
            repo_root=None,
            output_json_name="arkit_blendshapes.json",
            model_weight=None,
            config_file=None,
        ),
        audio_path=Path("/tmp/demo.wav"),
    )

    with pytest.raises(SystemExit, match="LAM official-run requires --face-repo"):
        adapter.run(args=hybrid.parse_args(["--audio", "/tmp/demo.wav", "--output-dir", str(tmp_path / "out")]), scratch_dir=tmp_path)


def test_lam_adapter_official_run_converts_output_to_hybrid_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hybrid = load_tool_module("hybrid_pipeline_lam_test", "hybrid_pipeline.py")
    repo_root = tmp_path / "LAM_Audio2Expression"
    (repo_root / "configs").mkdir(parents=True)
    (repo_root / "pretrained_models").mkdir(parents=True)
    (repo_root / "inference.py").write_text("# stub", encoding="utf-8")

    provider_config = hybrid.FaceProviderConfig(
        input_json=None,
        command="",
        python_bin="python3.10",
        repo_root=repo_root,
        output_json_name="arkit_blendshapes.json",
        model_weight=None,
        config_file=None,
    )
    adapter = hybrid.LamFaceAdapter(provider_config=provider_config, audio_path=Path("/tmp/demo.wav"))

    def fake_run(command, check, cwd):
        assert check is True
        assert cwd == repo_root
        output_item = next(item for item in command if item.startswith("save_json_path="))
        output_path = Path(output_item.split("=", 1)[1])
        _write_official_lam_json(output_path)
        return None

    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)
    result = adapter.run(
        args=hybrid.parse_args(["--audio", "/tmp/demo.wav", "--output-dir", str(tmp_path / "out")]),
        scratch_dir=tmp_path / "scratch",
    )

    payload = json.loads(result.payload_path.read_text(encoding="utf-8"))
    assert result.status == "ready"
    assert result.metadata["mode"] == "official-run"
    assert payload["frame_count"] == 2
    assert payload["frames"][0]["blendshapes"]["jawOpen"] == pytest.approx(0.1)
