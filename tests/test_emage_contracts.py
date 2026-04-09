from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import EMAGE_SAMPLE_OUTPUTS, load_tool_module


EXPECTED_EMAGE_KEYS = {
    "betas",
    "poses",
    "expressions",
    "trans",
    "model",
    "gender",
    "mocap_frame_rate",
}


@pytest.mark.parametrize("npz_path", EMAGE_SAMPLE_OUTPUTS, ids=lambda path: path.stem)
def test_emage_demo_outputs_match_current_contract(npz_path: Path):
    motion = np.load(npz_path, allow_pickle=True)

    assert set(motion.files) == EXPECTED_EMAGE_KEYS
    assert motion["betas"].shape == (300,)
    assert motion["poses"].ndim == 2
    assert motion["poses"].shape[1] == 165
    assert motion["expressions"].shape == (motion["poses"].shape[0], 100)
    assert motion["trans"].shape == (motion["poses"].shape[0], 3)
    assert float(motion["mocap_frame_rate"]) == pytest.approx(30.0)
    assert isinstance(motion["model"].item(), str)
    assert isinstance(motion["gender"].item(), str)


def test_export_arkit_from_npz_writes_current_json_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    export_arkit = load_tool_module("export_arkit_from_npz_test", "export_arkit_from_npz.py")

    source_npz = tmp_path / "sample_motion.npz"
    output_json = tmp_path / "arkit_blendshapes.json"
    mapping_matrix = tmp_path / "mat_final.npy"
    arkit_zip = tmp_path / "ARkit_FLAME.zip"

    frames = 2
    poses = np.zeros((frames, 165), dtype=np.float32)
    poses[:, 66:69] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    expressions = np.linspace(0.0, 1.0, num=frames * 100, dtype=np.float32).reshape(frames, 100)

    np.savez(
        source_npz,
        betas=np.zeros(300, dtype=np.float32),
        poses=poses,
        expressions=expressions,
        trans=np.zeros((frames, 3), dtype=np.float32),
        mocap_frame_rate=np.int64(30),
    )
    np.save(mapping_matrix, np.eye(3, 103, dtype=np.float32))

    with zipfile.ZipFile(arkit_zip, "w") as zf:
        for name in ("jawOpen", "eyeBlinkLeft", "eyeBlinkRight"):
            zf.writestr(f"bs/exp/{name}.obj", "o mesh\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_arkit_from_npz.py",
            "--npz",
            str(source_npz),
            "--mat",
            str(mapping_matrix),
            "--arkit-zip",
            str(arkit_zip),
            "--output",
            str(output_json),
        ],
    )

    assert export_arkit.main() == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["source_npz"] == str(source_npz)
    assert payload["fps"] == pytest.approx(30.0)
    assert "mapping_note" in payload
    assert len(payload["frames"]) == frames

    expected_names = {"jawOpen", "eyeBlinkLeft", "eyeBlinkRight"}
    for frame_index, frame in enumerate(payload["frames"]):
        assert set(frame) == {"frame", "time_sec", "blendshapes"}
        assert frame["frame"] == frame_index
        assert frame["time_sec"] == pytest.approx(frame_index / 30.0)
        assert set(frame["blendshapes"]) == expected_names
        assert all(0.0 <= value <= 1.0 for value in frame["blendshapes"].values())
