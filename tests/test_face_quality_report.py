from __future__ import annotations

import json
import wave
from pathlib import Path

from conftest import load_tool_module


def _write_test_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> Path:
    frame_count = int(duration_sec * sample_rate)
    samples = bytearray()
    for frame_idx in range(frame_count):
        amp = int(16000 * (1.0 if (frame_idx // 800) % 2 == 0 else 0.25))
        samples.extend(int(amp).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(samples))
    return path


def _write_face_payload(path: Path) -> Path:
    payload = {
        "fps": 30.0,
        "frame_count": 30,
        "names": ["jawOpen", "mouthPucker", "mouthShrugLower", "mouthStretchLeft", "mouthStretchRight"],
        "frames": [],
    }
    for frame_idx in range(30):
        t = frame_idx / 30.0
        jaw = 0.8 if frame_idx % 6 in (1, 2) else 0.1
        payload["frames"].append(
            {
                "frame": frame_idx,
                "time_sec": t,
                "blendshapes": {
                    "jawOpen": jaw,
                    "mouthPucker": jaw * 0.4,
                    "mouthShrugLower": jaw * 0.2,
                    "mouthStretchLeft": jaw * 0.3,
                    "mouthStretchRight": jaw * 0.35,
                },
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_analyze_face_quality_report_contains_alignment_and_curve_metrics(tmp_path: Path):
    report_mod = load_tool_module("face_quality_report_test", "face_quality_report.py")
    wav_path = _write_test_wav(tmp_path / "demo.wav")
    face_path = _write_face_payload(tmp_path / "face.json")

    report = report_mod.analyze_face_quality(audio_path=wav_path, face_json=face_path, label="raw-regression")

    assert report["label"] == "raw-regression"
    assert report["fps"] == 30.0
    assert report["curve_metrics"]["jawOpen"]["peak_count"] > 0
    assert "best_shift_frames" in report["alignment"]["audio_envelope_vs_jawOpen"]
    assert "mean_abs_delta" in report["curve_metrics"]["mouthPucker"]


def test_summarize_quality_study_prefers_lower_jitter_and_peak_density():
    report_mod = load_tool_module("face_quality_report_test", "face_quality_report.py")
    summary = report_mod.summarize_quality_study(
        {
            "raw-regression": {
                "label": "raw-regression",
                "curve_metrics": {"jawOpen": {"peak_density_hz": 3.0, "mean_abs_delta": 0.12}},
                "alignment": {"audio_envelope_vs_jawOpen": {"best_shift_frames": 0, "corr": 0.4}},
            },
            "tuned-regression-v1": {
                "label": "tuned-regression-v1",
                "curve_metrics": {"jawOpen": {"peak_density_hz": 1.8, "mean_abs_delta": 0.07}},
                "alignment": {"audio_envelope_vs_jawOpen": {"best_shift_frames": 0, "corr": 0.42}},
            },
            "lam-baseline": {
                "label": "lam-baseline",
                "curve_metrics": {"jawOpen": {"peak_density_hz": 1.6, "mean_abs_delta": 0.06}},
                "alignment": {"audio_envelope_vs_jawOpen": {"best_shift_frames": 0, "corr": 0.2}},
            },
        }
    )

    assert summary["recommended_profile"] == "tuned-regression-v1"
    assert summary["candidate_rankings"][0]["profile"] == "tuned-regression-v1"
