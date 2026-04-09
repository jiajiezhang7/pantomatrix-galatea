#!/usr/bin/env python3
import argparse
import threading
import time
from pathlib import Path

import numpy as np
import smplx
import torch
import viser


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Preview EMAGE body motion in Viser.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument(
        "--smplx-root",
        type=Path,
        default=workspace_root / "third_party" / "PantoMatrix" / "emage_evaltools" / "smplx_models",
    )
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    return parser.parse_args()


def build_vertices(npz_path: Path, smplx_root: Path):
    motion = np.load(npz_path, allow_pickle=True)
    poses = motion["poses"].astype(np.float32)
    trans = motion["trans"].astype(np.float32)
    betas = motion["betas"].astype(np.float32)

    model = smplx.create(
        model_path=str(smplx_root),
        model_type="smplx",
        gender="NEUTRAL_2020",
        num_betas=300,
        num_expression_coeffs=100,
        ext="npz",
        use_pca=False,
        use_face_contour=False,
        batch_size=1,
    ).eval()

    betas_ts = torch.from_numpy(np.repeat(betas[None, :], 1, axis=0))
    vertices = []
    for frame in range(poses.shape[0]):
        pose = poses[frame]
        out = model(
            betas=betas_ts,
            transl=torch.from_numpy(trans[frame : frame + 1]),
            global_orient=torch.from_numpy(pose[0:3].reshape(1, 3)),
            body_pose=torch.from_numpy(pose[3 : 22 * 3].reshape(1, 63)),
            jaw_pose=torch.from_numpy(pose[22 * 3 : 23 * 3].reshape(1, 3)),
            leye_pose=torch.from_numpy(pose[23 * 3 : 24 * 3].reshape(1, 3)),
            reye_pose=torch.from_numpy(pose[24 * 3 : 25 * 3].reshape(1, 3)),
            left_hand_pose=torch.from_numpy(pose[25 * 3 : 40 * 3].reshape(1, 45)),
            right_hand_pose=torch.from_numpy(pose[40 * 3 : 55 * 3].reshape(1, 45)),
            expression=torch.zeros(1, 100),
            return_verts=True,
        )
        vertices.append(out.vertices[0].detach().cpu().numpy())
    return motion, np.stack(vertices, axis=0), np.asarray(model.faces, dtype=np.int32)


def main() -> int:
    args = parse_args()
    motion, vertices, faces = build_vertices(args.npz, args.smplx_root)
    fps = float(motion["mocap_frame_rate"]) if "mocap_frame_rate" in motion.files else 30.0

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.5, section_size=1.0)
    server.scene.add_frame("/world", show_axes=False)

    frame_slider = server.gui.add_slider("frame", min=0, max=int(vertices.shape[0] - 1), step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", initial_value=False)
    fps_slider = server.gui.add_slider("fps", min=1, max=60, step=1, initial_value=int(fps))

    mesh_holder = {"handle": None}

    def render_frame(frame_idx: int) -> None:
        if mesh_holder["handle"] is not None:
            mesh_holder["handle"].remove()
        mesh_holder["handle"] = server.scene.add_mesh_simple(
            "/emage_body",
            vertices=vertices[frame_idx],
            faces=faces,
            color=(120, 190, 255),
        )

    @frame_slider.on_update
    def _(_event) -> None:
        render_frame(int(frame_slider.value))

    def autoplay() -> None:
        while True:
            if playing.value:
                next_frame = (int(frame_slider.value) + 1) % vertices.shape[0]
                frame_slider.value = next_frame
            time.sleep(1.0 / max(1, int(fps_slider.value)))

    render_frame(0)
    thread = threading.Thread(target=autoplay, daemon=True)
    thread.start()

    print(f"Viser running at http://{server.get_host()}:{server.get_port()} for {args.npz}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
