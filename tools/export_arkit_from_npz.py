#!/usr/bin/env python3
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Approximate ARKit blendshape weights from EMAGE FLAME outputs."
    )
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument(
        "--mat",
        type=Path,
        default=workspace_root / "third_party" / "PantoMatrix" / "beat2_tools" / "mat_final.npy",
    )
    parser.add_argument(
        "--arkit-zip",
        type=Path,
        default=workspace_root / "third_party" / "PantoMatrix" / "beat2_tools" / "ARkit_FLAME.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace_root / "demo_artifacts" / "arkit" / "emage_arkit.json",
    )
    parser.add_argument("--clip", action="store_true", default=True)
    return parser.parse_args()


def arkit_names_from_zip(arkit_zip: Path) -> list[str]:
    with zipfile.ZipFile(arkit_zip) as zf:
        names = [
            Path(name).stem
            for name in zf.namelist()
            if name.startswith("bs/exp/") and name.endswith(".obj")
        ]
    return names


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    motion = np.load(args.npz, allow_pickle=True)
    expressions = motion["expressions"].astype(np.float32)
    jaw = motion["poses"][:, 22 * 3 : 23 * 3].astype(np.float32)
    flame103 = np.concatenate([expressions, jaw], axis=1)

    mat = np.load(args.mat).astype(np.float32)
    if mat.shape[1] != flame103.shape[1]:
        raise ValueError(f"Mapping matrix expects {mat.shape[1]} dims, got {flame103.shape[1]}")

    # `mat_final.npy` is published as ARKit -> FLAME. For demo export we solve the inverse
    # least-squares problem frame-by-frame with the Moore-Penrose pseudoinverse.
    arkit = flame103 @ np.linalg.pinv(mat)
    if args.clip:
        arkit = np.clip(arkit, 0.0, 1.0)

    names = arkit_names_from_zip(args.arkit_zip)
    if len(names) != arkit.shape[1]:
        raise ValueError(f"Found {len(names)} ARKit names, but weights have {arkit.shape[1]} columns")

    fps = float(motion["mocap_frame_rate"]) if "mocap_frame_rate" in motion.files else 30.0
    records = []
    for idx, weights in enumerate(arkit):
        records.append(
            {
                "frame": idx,
                "time_sec": idx / fps,
                "blendshapes": {name: float(value) for name, value in zip(names, weights)},
            }
        )

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_npz": str(args.npz),
                "fps": fps,
                "mapping_note": "Inverse least-squares reconstruction from official ARKit->FLAME matrix.",
                "frames": records,
            },
            f,
            ensure_ascii=True,
        )
    print(f"[ok] wrote {args.output} with {len(records)} frames and {len(names)} blendshapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
