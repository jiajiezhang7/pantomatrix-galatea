#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    repo_root = workspace_root / "third_party" / "PantoMatrix"

    parser = argparse.ArgumentParser(description="Download EMAGE model and tool assets.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="",
        help="Optional alternate Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root
    kwargs = {"endpoint": args.hf_endpoint} if args.hf_endpoint else {}
    snapshot_download(
        repo_id="H-Liu1997/emage_audio",
        local_dir=str(repo_root / "hf_cache" / "emage_audio"),
        **kwargs,
    )
    snapshot_download(
        repo_id="H-Liu1997/emage_evaltools",
        local_dir=str(repo_root / "emage_evaltools"),
        **kwargs,
    )
    for filename in ("mat_final.npy", "ARkit_FLAME.zip"):
        hf_hub_download(
            repo_id="H-Liu1997/BEAT2_Tools",
            repo_type="dataset",
            filename=filename,
            local_dir=str(repo_root / "beat2_tools"),
            **kwargs,
        )
    print(f"[ok] assets downloaded under {repo_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
