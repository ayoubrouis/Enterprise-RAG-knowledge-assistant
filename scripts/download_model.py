"""Download the local LLM weights into models/ (resumable, retry-safe).

This avoids the Hugging Face Hub downloader entirely (which can stall on slow
networks) by streaming the files directly with HTTP Range requests. Downloads
are resumable: re-running this script continues where it left off.

Default mirror: https://hf-mirror.com  (set HF_ENDPOINT to override, e.g. the
official https://huggingface.co if your connection to it is fast).

Usage:
    python scripts/download_model.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

# Make `app` importable when running this script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
REPO = settings.LLM_MODEL  # hub id, e.g. google/flan-t5-base
DEST_DIR = settings.ROOT_DIR / "models" / REPO.split("/")[-1]

MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


def remote_size(url: str) -> int:
    """Determine the remote file size, tolerating servers that skip HEAD
    Content-Length (some mirrors). Fallback: a 1-byte Range request."""
    try:
        head = requests.head(url, allow_redirects=True, timeout=60)
        if "Content-Length" in head.headers:
            return int(head.headers["Content-Length"])
    except requests.RequestException:
        pass
    resp = requests.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
    resp.raise_for_status()
    # e.g. "bytes 0-0/989920386"
    return int(resp.headers.get("Content-Range", "").split("/")[-1])


def download_with_resume(url: str, path: Path) -> None:
    """Stream ``url`` to ``path``, resuming any partially downloaded file."""
    part = path.with_suffix(path.suffix + ".part")
    total = remote_size(url)

    if path.exists() and path.stat().st_size == total:
        print(f"  already complete: {path.name}")
        return

    pos = part.stat().st_size if part.exists() else 0
    print(f"  {path.name}: {pos / 1e6:.1f}/{total / 1e6:.0f} MB")
    while pos < total:
        headers = {"Range": f"bytes={pos}-"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as resp:
                resp.raise_for_status()
                with open(part, "ab") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
                            pos += len(chunk)
                            if pos % (20 << 20) == 0:
                                print(f"  {path.name}: {pos / 1e6:.0f}/{total / 1e6:.0f} MB")
        except requests.RequestException as exc:
            print(f"  interrupted ({exc}); retrying in 5s...")
            import time

            time.sleep(5)
    part.rename(path)
    print(f"  complete: {path.name} ({total / 1e6:.0f} MB)")


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading `{REPO}` -> {DEST_DIR}")
    for name in MODEL_FILES:
        url = f"{BASE}/{REPO}/resolve/main/{name}"
        try:
            download_with_resume(url, DEST_DIR / name)
        except requests.HTTPError as exc:
            print(f"  skipping {name}: {exc}")


if __name__ == "__main__":
    main()
