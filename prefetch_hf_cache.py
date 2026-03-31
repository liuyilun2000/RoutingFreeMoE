#!/usr/bin/env python3
"""
Prefetch Hugging Face repos into local cache for offline/compute-node usage.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def load_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HF_READ_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            return token

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"HF_TOKEN", "HF_READ_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
            continue
        token = value.strip().strip('"').strip("'")
        if token:
            return token
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HF model repos into cache.")
    parser.add_argument(
        "--repos",
        nargs="+",
        default=[
            "liuyilun2000/routingfreemoe-baseline-final-model",
            "liuyilun2000/routingfreemoe-rf-final-model",
        ],
        help="One or more HF repo ids to cache.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Repo revision to cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_HOME", ""),
        help="Optional cache root (HF_HOME). Empty uses default HF cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = load_hf_token()
    if not token:
        raise RuntimeError(
            "No HF token found. Set HF_TOKEN/HF_READ_TOKEN or put one in ../.env."
        )

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        print(f"Using HF_HOME={args.cache_dir}")

    for repo in args.repos:
        print(f"\nCaching {repo}@{args.revision} ...")
        path = snapshot_download(
            repo_id=repo,
            revision=args.revision,
            token=token,
            resume_download=True,
        )
        print(f"Cached at: {path}")

    print("\nDone. Repos are available in local HF cache.")


if __name__ == "__main__":
    main()
