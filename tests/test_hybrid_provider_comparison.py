from __future__ import annotations

import json
from pathlib import Path

from conftest import load_tool_module


def test_build_provider_output_dir_uses_provider_name(tmp_path: Path):
    compare = load_tool_module("hybrid_provider_compare_test", "run_hybrid_provider_comparison.py")

    output_dir = compare.build_provider_output_dir(tmp_path / "runs", "a2f-3d-sdk")

    assert output_dir == tmp_path / "runs" / "a2f-3d-sdk"


def test_write_comparison_manifest_records_both_providers(tmp_path: Path):
    compare = load_tool_module("hybrid_provider_compare_test", "run_hybrid_provider_comparison.py")
    manifest_path = tmp_path / "comparison_manifest.json"

    compare.write_comparison_manifest(
        manifest_path=manifest_path,
        audio_path=Path("data/audio/audio-demo-female.wav"),
        provider_outputs={
            "a2f-3d-sdk": {"run_dir": "runs/a2f-3d-sdk", "video": "runs/a2f-3d-sdk/showcase.mp4"},
            "lam": {"run_dir": "runs/lam", "video": "runs/lam/showcase.mp4"},
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["audio_path"] == "data/audio/audio-demo-female.wav"
    assert set(manifest["providers"]) == {"a2f-3d-sdk", "lam"}
    assert manifest["providers"]["a2f-3d-sdk"]["video"] == "runs/a2f-3d-sdk/showcase.mp4"


def test_build_render_command_uses_lam_python_runtime(tmp_path: Path):
    compare = load_tool_module("hybrid_provider_compare_test", "run_hybrid_provider_comparison.py")
    args = compare.parse_args(
        [
            "--audio",
            "/tmp/audio.wav",
            "--body-npz",
            "/tmp/body.npz",
            "--output-root",
            "/tmp/runs",
            "--lam-python",
            "/opt/conda/envs/lam/bin/python",
        ]
    )

    command = compare.build_render_command(
        args=args,
        provider="a2f-3d-sdk",
        run_dir=tmp_path / "a2f-3d-sdk",
        showcase_video=tmp_path / "a2f-3d-sdk" / "showcase.mp4",
    )

    assert command[0] == "/opt/conda/envs/lam/bin/python"
    assert command[1].endswith("render_hybrid_showcase_video.py")
    assert "--face-provider" in command
