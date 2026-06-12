"""Fetch an OpenVINO Whisper model for AirSketch voice input.

Tries two strategies, in order:
  1. Download a pre-converted OpenVINO IR Whisper model from HuggingFace
     (no `optimum` needed) into models/whisper-base-ov.
  2. Fall back to `optimum-cli export openvino` if the pre-converted repo
     isn't reachable and optimum[openvino] is installed.

Usage:
    python -m training.setup_whisper
    python -m training.setup_whisper --repo OpenVINO/whisper-tiny-fp16-ov
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "models" / "whisper-base-ov"

# Pre-converted OpenVINO IR Whisper repos (community/official). First that works wins.
DEFAULT_REPOS = [
    "OpenVINO/whisper-base-fp16-ov",
    "OpenVINO/whisper-tiny-fp16-ov",
]

# What openvino-genai WhisperPipeline expects to find in the model dir
_REQUIRED_HINTS = ("openvino_encoder_model.xml", "openvino_decoder_model.xml")


def _has_ov_whisper(d: Path) -> bool:
    if not d.exists():
        return False
    names = {p.name for p in d.glob("*.xml")}
    return any(h in names for h in _REQUIRED_HINTS)


def try_download(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed.")
        return False
    print(f"Downloading {repo} -> {OUT_DIR} ...")
    try:
        local = snapshot_download(repo_id=repo)
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}")
        return False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Copy the resolved snapshot into our canonical location
    for item in Path(local).iterdir():
        dest = OUT_DIR / item.name
        if item.is_file() and not dest.exists():
            shutil.copy2(item, dest)
    ok = _has_ov_whisper(OUT_DIR)
    print("  OK" if ok else "  downloaded but no OV whisper XML found")
    return ok


def try_optimum_export(hf_model: str = "openai/whisper-base") -> bool:
    print(f"Falling back to optimum-cli export of {hf_model} ...")
    print("  (requires: pip install optimum[openvino])")
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli",
        "export", "openvino", "--model", hf_model,
        "--task", "automatic-speech-recognition-with-past",
        str(OUT_DIR),
    ]
    try:
        subprocess.run(cmd, check=True)
        return _has_ov_whisper(OUT_DIR)
    except Exception as e:
        print(f"  optimum export failed: {type(e).__name__}: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Set up an OpenVINO Whisper model")
    p.add_argument("--repo", default=None, help="Specific pre-converted HF repo to use")
    p.add_argument("--hf-model", default="openai/whisper-base",
                   help="HF model to export with optimum if download fails")
    args = p.parse_args()

    if _has_ov_whisper(OUT_DIR):
        print(f"Whisper model already present at {OUT_DIR}")
        return

    repos = [args.repo] if args.repo else DEFAULT_REPOS
    for repo in repos:
        if try_download(repo):
            print(f"\nWhisper ready at: {OUT_DIR}")
            return

    if try_optimum_export(args.hf_model):
        print(f"\nWhisper ready at: {OUT_DIR}")
        return

    print("\nCould not set up a Whisper model automatically.")
    print("Options:")
    print("  1) pip install optimum[openvino] && python -m training.setup_whisper")
    print("  2) Manually place an OpenVINO Whisper IR in models/whisper-base-ov/")
    sys.exit(1)


if __name__ == "__main__":
    main()
