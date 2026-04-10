from __future__ import annotations

import json
from pathlib import Path

from conftest import load_tool_module


def test_build_study_output_dir_uses_clip_stem(tmp_path: Path):
    study = load_tool_module("a2f_quality_study_test", "run_a2f_quality_study.py")

    output_dir = study.build_study_output_dir(tmp_path, Path("/tmp/audio-demo-female.wav"))

    assert output_dir == tmp_path / "audio-demo-female"


def test_write_study_manifest_records_profiles_and_recommendation(tmp_path: Path):
    study = load_tool_module("a2f_quality_study_test", "run_a2f_quality_study.py")
    manifest_path = tmp_path / "study_manifest.json"
    study.write_study_manifest(
        manifest_path=manifest_path,
        audio_path=Path("output/audio.wav"),
        profile_outputs={
            "raw-regression": {"run_dir": "study/raw-regression", "video": "study/raw-regression/showcase.mp4"},
            "tuned-regression-v1": {"run_dir": "study/tuned-regression-v1", "video": "study/tuned-regression-v1/showcase.mp4"},
        },
        summary={"recommended_profile": "tuned-regression-v1"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["audio_path"] == "output/audio.wav"
    assert manifest["summary"]["recommended_profile"] == "tuned-regression-v1"
    assert "raw-regression" in manifest["profiles"]
