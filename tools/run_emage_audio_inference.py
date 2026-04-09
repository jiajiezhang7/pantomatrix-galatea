#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    repo_root = workspace_root / "third_party" / "PantoMatrix"
    model_root = repo_root / "hf_cache" / "emage_audio"
    default_audio = workspace_root / "demo_assets" / "audio"
    default_save = workspace_root / "demo_artifacts" / "emage"

    parser = argparse.ArgumentParser(
        description="Run EMAGE audio inference without importing visualization dependencies."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--model-root", type=Path, default=model_root)
    parser.add_argument("--audio-folder", type=Path, default=default_audio)
    parser.add_argument("--save-folder", type=Path, default=default_save)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_models(repo_root: Path, model_root: Path, device: torch.device):
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from emage_utils.motion_io import beat_format_save
    from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

    face_motion_vq = EmageVQVAEConv.from_pretrained(model_root / "emage_vq" / "face").to(device)
    upper_motion_vq = EmageVQVAEConv.from_pretrained(model_root / "emage_vq" / "upper").to(device)
    lower_motion_vq = EmageVQVAEConv.from_pretrained(model_root / "emage_vq" / "lower").to(device)
    hands_motion_vq = EmageVQVAEConv.from_pretrained(model_root / "emage_vq" / "hands").to(device)
    global_motion_ae = EmageVAEConv.from_pretrained(model_root / "emage_vq" / "global").to(device)
    motion_vq = EmageVQModel(
        face_model=face_motion_vq,
        upper_model=upper_motion_vq,
        lower_model=lower_motion_vq,
        hands_model=hands_motion_vq,
        global_model=global_motion_ae,
    ).to(device)
    motion_vq.eval()

    model = EmageAudioModel.from_pretrained(model_root).to(device)
    model.eval()
    return model, motion_vq, beat_format_save


def run_one_audio(
    model,
    motion_vq,
    beat_format_save,
    audio_path: Path,
    save_folder: Path,
    device: torch.device,
) -> Path:
    audio, _ = librosa.load(audio_path, sr=model.cfg.audio_sr, mono=True)
    audio_ts = torch.from_numpy(audio).float().to(device).unsqueeze(0)
    speaker_id = torch.zeros(1, 1, dtype=torch.long, device=device)
    trans = torch.zeros(1, 1, 3, device=device)

    with torch.no_grad():
        latent_dict = model.inference(audio_ts, speaker_id, motion_vq, masked_motion=None, mask=None)
        all_pred = motion_vq.decode(
            face_latent=latent_dict["rec_face"] if model.cfg.lf > 0 and model.cfg.cf == 0 else None,
            upper_latent=latent_dict["rec_upper"] if model.cfg.lu > 0 and model.cfg.cu == 0 else None,
            lower_latent=latent_dict["rec_lower"] if model.cfg.ll > 0 and model.cfg.cl == 0 else None,
            hands_latent=latent_dict["rec_hands"] if model.cfg.lh > 0 and model.cfg.ch == 0 else None,
            face_index=torch.argmax(latent_dict["cls_face"], dim=2) if model.cfg.cf > 0 else None,
            upper_index=torch.argmax(latent_dict["cls_upper"], dim=2) if model.cfg.cu > 0 else None,
            lower_index=torch.argmax(latent_dict["cls_lower"], dim=2) if model.cfg.cl > 0 else None,
            hands_index=torch.argmax(latent_dict["cls_hands"], dim=2) if model.cfg.ch > 0 else None,
            get_global_motion=True,
            ref_trans=trans[:, 0],
        )

    motion_pred = all_pred["motion_axis_angle"].detach().cpu().numpy()
    face_pred = all_pred["expression"].detach().cpu().numpy()
    trans_pred = all_pred["trans"].detach().cpu().numpy()

    frames = motion_pred.shape[1]
    motion_flat = motion_pred.reshape(frames, -1)
    face_flat = face_pred.reshape(frames, -1)
    trans_flat = trans_pred.reshape(frames, -1)

    output_path = save_folder / f"{audio_path.stem}_output.npz"
    beat_format_save(
        str(output_path),
        motion_flat,
        upsample=30 // model.cfg.pose_fps,
        expressions=face_flat,
        trans=trans_flat,
    )
    return output_path


def main() -> int:
    args = parse_args()
    args.audio_folder.mkdir(parents=True, exist_ok=True)
    args.save_folder.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(args.audio_folder.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {args.audio_folder}")
    if not args.model_root.exists():
        raise FileNotFoundError(f"Model root not found: {args.model_root}")

    device = torch.device(args.device)
    start_time = time.time()
    model, motion_vq, beat_format_save = load_models(args.repo_root, args.model_root, device)

    generated = []
    for audio_path in wav_files:
        output_path = run_one_audio(model, motion_vq, beat_format_save, audio_path, args.save_folder, device)
        generated.append(output_path)
        data = np.load(output_path, allow_pickle=True)
        print(
            f"[ok] {audio_path.name} -> {output_path.name} "
            f"poses={data['poses'].shape} expressions={data['expressions'].shape} trans={data['trans'].shape}"
        )

    elapsed = time.time() - start_time
    print(f"generated {len(generated)} file(s) in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
