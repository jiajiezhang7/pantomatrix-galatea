#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


MOUTH_METRIC_KEYS = (
    "jawOpen",
    "mouthPucker",
    "mouthShrugLower",
    "mouthStretchLeft",
    "mouthStretchRight",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze face-motion quality against source audio.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--face-json", type=Path, required=True)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_face_payload(face_json: Path) -> dict[str, Any]:
    return json.loads(face_json.read_text(encoding="utf-8"))


def _read_waveform(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(frame_count)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported wav sample width: {sample_width}")
    waveform = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    return waveform, sample_rate


def compute_audio_envelope(audio_path: Path, fps: float) -> np.ndarray:
    waveform, sample_rate = _read_waveform(audio_path)
    hop = max(1, int(round(sample_rate / max(fps, 1.0))))
    values = []
    for start in range(0, len(waveform), hop):
        chunk = waveform[start : start + hop]
        if len(chunk) == 0:
            break
        values.append(float(np.sqrt(np.mean(np.square(chunk)))))
    envelope = np.asarray(values, dtype=np.float32)
    if envelope.size == 0:
        return envelope
    max_value = float(envelope.max())
    if max_value > 0:
        envelope /= max_value
    return envelope


def _curve_metrics(values: np.ndarray, fps: float) -> dict[str, float]:
    peaks = 0
    for idx in range(1, len(values) - 1):
        if values[idx] > values[idx - 1] and values[idx] >= values[idx + 1] and values[idx] > float(values.mean()):
            peaks += 1
    return {
        "mean": float(values.mean()) if len(values) else 0.0,
        "max": float(values.max()) if len(values) else 0.0,
        "std": float(values.std()) if len(values) else 0.0,
        "mean_abs_delta": float(np.abs(np.diff(values)).mean()) if len(values) > 1 else 0.0,
        "peak_count": int(peaks),
        "peak_density_hz": float(peaks / max(len(values) / max(fps, 1.0), 1e-6)),
    }


def _best_shift_correlation(
    audio_envelope: np.ndarray, curve: np.ndarray, fps: float, max_shift_frames: int = 15
) -> dict[str, float]:
    best_shift = 0
    best_corr = -1.0
    if len(audio_envelope) == 0 or len(curve) == 0:
        return {"best_shift_frames": 0, "best_shift_sec": 0.0, "corr": 0.0}

    for shift in range(-max_shift_frames, max_shift_frames + 1):
        if shift >= 0:
            audio_slice = audio_envelope[: max(0, len(curve) - shift)]
            curve_slice = curve[shift : shift + len(audio_slice)]
        else:
            audio_slice = audio_envelope[-shift:]
            curve_slice = curve[: len(audio_slice)]
        n = min(len(audio_slice), len(curve_slice))
        if n < 8:
            continue
        audio_slice = audio_slice[:n]
        curve_slice = curve_slice[:n]
        corr = float(np.corrcoef(audio_slice, curve_slice)[0, 1])
        if math.isnan(corr):
            corr = 0.0
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return {"best_shift_frames": int(best_shift), "best_shift_sec": float(best_shift / max(fps, 1.0)), "corr": float(best_corr)}


def analyze_face_quality(*, audio_path: Path, face_json: Path, label: str) -> dict[str, Any]:
    payload = load_face_payload(face_json)
    fps = float(payload.get("fps") or payload.get("metadata", {}).get("fps") or 30.0)
    frames = payload["frames"]
    curve_metrics: dict[str, dict[str, float]] = {}
    jaw_open_curve = np.asarray([float(frame["blendshapes"].get("jawOpen", 0.0)) for frame in frames], dtype=np.float32)
    for metric_name in MOUTH_METRIC_KEYS:
        values = np.asarray([float(frame["blendshapes"].get(metric_name, 0.0)) for frame in frames], dtype=np.float32)
        curve_metrics[metric_name] = _curve_metrics(values, fps)

    try:
        audio_envelope = compute_audio_envelope(audio_path, fps)
        min_len = min(len(audio_envelope), len(jaw_open_curve))
        alignment = _best_shift_correlation(audio_envelope[:min_len], jaw_open_curve[:min_len], fps)
        waveform, sample_rate = _read_waveform(audio_path)
        audio_duration_sec = len(waveform) / max(sample_rate, 1)
        audio_error = None
    except (wave.Error, ValueError, FileNotFoundError):
        alignment = {"best_shift_frames": 0, "best_shift_sec": 0.0, "corr": 0.0}
        audio_duration_sec = 0.0
        audio_error = "audio_unavailable_for_quality_alignment"
    return {
        "label": label,
        "audio_path": str(audio_path),
        "face_json": str(face_json),
        "fps": fps,
        "frame_count": len(frames),
        "audio_duration_sec": float(audio_duration_sec),
        "audio_error": audio_error,
        "curve_metrics": curve_metrics,
        "alignment": {"audio_envelope_vs_jawOpen": alignment},
    }


def summarize_quality_study(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = reports.get("raw-regression") or {}
    lam = reports.get("lam-baseline") or {}
    rankings = []
    for profile_name, report in reports.items():
        if profile_name in {"raw-regression", "lam-baseline"}:
            continue
        jaw = report["curve_metrics"]["jawOpen"]
        raw_jaw = raw.get("curve_metrics", {}).get("jawOpen", {})
        lam_jaw = lam.get("curve_metrics", {}).get("jawOpen", {})
        shift_penalty = abs(report["alignment"]["audio_envelope_vs_jawOpen"]["best_shift_frames"])
        score = 0.0
        score += max(0.0, float(raw_jaw.get("peak_density_hz", jaw["peak_density_hz"])) - jaw["peak_density_hz"])
        score += max(0.0, float(raw_jaw.get("mean_abs_delta", jaw["mean_abs_delta"])) - jaw["mean_abs_delta"]) * 10.0
        if lam_jaw:
            score -= abs(jaw["peak_density_hz"] - float(lam_jaw.get("peak_density_hz", jaw["peak_density_hz"])))
            score -= abs(jaw["mean_abs_delta"] - float(lam_jaw.get("mean_abs_delta", jaw["mean_abs_delta"]))) * 5.0
        score -= shift_penalty * 0.25
        rankings.append({"profile": profile_name, "score": float(score)})

    rankings.sort(key=lambda item: item["score"], reverse=True)
    return {
        "recommended_profile": rankings[0]["profile"] if rankings else "raw-regression",
        "candidate_rankings": rankings,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    label = args.label or args.face_json.stem
    report = analyze_face_quality(audio_path=args.audio, face_json=args.face_json, label=label)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
