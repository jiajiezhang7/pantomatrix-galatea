#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


LOCAL_UPPER_MASK = np.array(
    [
        False, False, False, True, False, False, True, False, False, True,
        False, False, True, True, True, True, True, True, True, True,
        True, True, False, False, False, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True,
    ],
    dtype=bool,
)

# Preserve jaw + both eyes in addition to the official upper-body mask.
FACE_JOINT_IDS = np.array([22, 23, 24], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Freeze lower body and root translation while preserving upper body, hands, and face."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace_root / "data" / "emage_outputs_upper_only" / "upper_only_output.npz",
    )
    return parser.parse_args()


def build_upper_only_npz(input_path: Path, output_path: Path) -> None:
    data = np.load(input_path, allow_pickle=True)
    poses = data["poses"].astype(np.float32).reshape(-1, 55, 3)
    trans = data["trans"].astype(np.float32)

    keep_mask = LOCAL_UPPER_MASK.copy()
    keep_mask[FACE_JOINT_IDS] = True

    base_pose = np.repeat(poses[:1], poses.shape[0], axis=0)
    base_pose[:, keep_mask, :] = poses[:, keep_mask, :]
    frozen_trans = np.repeat(trans[:1], trans.shape[0], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        betas=data["betas"],
        poses=base_pose.reshape(poses.shape[0], -1),
        expressions=data["expressions"],
        trans=frozen_trans,
        model=data["model"],
        gender=data["gender"],
        mocap_frame_rate=data["mocap_frame_rate"],
    )


def main() -> int:
    args = parse_args()
    build_upper_only_npz(args.input, args.output)
    print(f"[ok] wrote upper-body-only npz to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
