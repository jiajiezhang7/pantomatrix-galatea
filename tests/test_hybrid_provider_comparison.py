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
