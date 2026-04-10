from __future__ import annotations

import json
from pathlib import Path


REQUIRED_COMPONENTS = {"emage_body", "face_provider"}
REQUIRED_MANIFEST_KEYS = {
    "audio_path",
    "target_fps",
    "duration_sec",
    "frame_count",
    "components",
    "component_aliases",
    "sources",
    "alignment_policy",
}


def assert_manifest_contract(manifest: dict) -> None:
    assert REQUIRED_MANIFEST_KEYS.issubset(manifest)
    assert manifest["target_fps"] == 30.0
    assert manifest["frame_count"] > 0
    assert set(manifest["components"]) == REQUIRED_COMPONENTS
    assert manifest["component_aliases"]["lam_face"] == "face_provider"
    assert "provider_name" in manifest["components"]["face_provider"]
    assert "raw_payload_path" in manifest["components"]["face_provider"]
    assert "normalization_policy" in manifest["components"]["face_provider"]
    assert isinstance(manifest["sources"], dict)
    assert isinstance(manifest["alignment_policy"], dict)


def test_hybrid_manifest_contract_uses_body_and_face_provider_components_only():
    manifest = {
        "audio_path": "data/audio/audio-demo-female.wav",
        "target_fps": 30.0,
        "duration_sec": 1.0,
        "frame_count": 30,
        "components": {
            "emage_body": {
                "status": "ready",
                "fps": 30.0,
                "frame_count": 30,
                "payload_path": "body/body.npz",
                "quality_notes": [],
            },
            "face_provider": {
                "status": "ready",
                "fps": 30.0,
                "frame_count": 30,
                "payload_path": "face/arkit_blendshapes.json",
                "provider_name": "lam",
                "raw_payload_path": "face/provider_raw.json",
                "normalization_policy": "provider-native",
                "quality_notes": ["includes eye-related ARKit blendshapes from the face provider"],
            },
        },
        "component_aliases": {
            "lam_face": "face_provider",
        },
        "sources": {
            "body": "EMAGE",
            "face": "LAM_Audio2Expression",
            },
        "alignment_policy": {
            "timeline_source": "audio_duration",
            "target_fps": 30.0,
            "resample": "linear",
        },
    }

    assert_manifest_contract(manifest)


def test_hybrid_face_contract_can_carry_eye_related_blendshapes():
    face_frame = {
        "frame": 12,
        "time_sec": 0.4,
        "blendshapes": {
            "jawOpen": 0.5,
            "eyeBlinkLeft": 0.3,
            "eyeBlinkRight": 0.25,
            "eyeLookUpLeft": 0.1,
            "eyeLookUpRight": 0.08,
        },
    }

    assert set(face_frame) == {
        "frame",
        "time_sec",
        "blendshapes",
    }
    assert face_frame["blendshapes"]["eyeBlinkLeft"] == 0.3
    assert face_frame["blendshapes"]["eyeLookUpRight"] == 0.08


def test_hybrid_result_folder_layout_requires_body_face_metrics_and_logs(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    body_path = tmp_path / "body" / "body.npz"
    face_path = tmp_path / "face" / "arkit_blendshapes.json"
    provider_metadata_path = tmp_path / "face" / "provider_metadata.json"
    metrics_path = tmp_path / "metrics" / "run_metrics.json"
    log_path = tmp_path / "logs" / "pipeline.log"

    body_path.parent.mkdir(parents=True)
    face_path.parent.mkdir(parents=True)
    metrics_path.parent.mkdir(parents=True)
    log_path.parent.mkdir(parents=True)

    manifest_path.write_text(
        json.dumps(
            {
                "audio_path": "data/audio/audio-demo-female.wav",
                "target_fps": 30.0,
                "duration_sec": 1.0,
                "frame_count": 30,
                "components": {
                    "emage_body": {"status": "ready"},
                    "face_provider": {
                        "status": "ready",
                        "provider_name": "lam",
                        "raw_payload_path": "face/provider_raw.json",
                        "normalization_policy": "provider-native",
                    },
                },
                "component_aliases": {"lam_face": "face_provider"},
                "sources": {"body": "EMAGE", "face": "LAM_Audio2Expression"},
                "alignment_policy": {"timeline_source": "audio_duration"},
            }
        ),
        encoding="utf-8",
    )
    body_path.write_bytes(b"npz")
    face_path.write_text("[]", encoding="utf-8")
    provider_metadata_path.write_text("{}", encoding="utf-8")
    metrics_path.write_text("{}", encoding="utf-8")
    log_path.write_text("ok", encoding="utf-8")

    required_paths = [manifest_path, body_path, face_path, provider_metadata_path, metrics_path, log_path]
    assert all(path.exists() for path in required_paths)
