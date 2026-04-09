#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


HF_REPO = "3DAIGC/LAM_audio2exp"
HF_FILES = (
    "LAM_audio2exp_assets.tar",
    "LAM_audio2exp_streaming.tar",
)
OSS_URLS = {
    "LAM_audio2exp_assets.tar": "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_assets.tar",
    "LAM_audio2exp_streaming.tar": "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_streaming.tar",
}


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download LAM Audio2Expression assets and weights.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=workspace_root / "third_party" / "LAM_Audio2Expression",
    )
    return parser.parse_args()


def hf_download(repo_root: Path) -> bool:
    hf_cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf_cli is None:
        return False

    command = [hf_cli, "download", HF_REPO, "--local-dir", str(repo_root)]
    result = subprocess.run(command, check=False)
    return result.returncode == 0


def oss_download(repo_root: Path) -> None:
    for filename in HF_FILES:
        target = repo_root / filename
        with urllib.request.urlopen(OSS_URLS[filename]) as response, target.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)


def extract_archives(repo_root: Path) -> None:
    for filename in HF_FILES:
        archive_path = repo_root / filename
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing archive: {archive_path}")
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(repo_root)
        archive_path.unlink()


def main() -> int:
    args = parse_args()
    args.repo_root.mkdir(parents=True, exist_ok=True)

    downloaded = hf_download(args.repo_root)
    if not downloaded:
        oss_download(args.repo_root)

    extract_archives(args.repo_root)
    print(f"[ok] LAM assets ready under {args.repo_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
