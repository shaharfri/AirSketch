"""Download the OpenVINO model weights AirSketch needs (they are NOT stored in git).

Run this once on a fresh clone to fetch the models for the features you want.
Each download is idempotent — it skips a model that is already present.

Usage:
    python fetch_models.py                 # speaker model only (fixes dictation)
    python fetch_models.py --speaker       # WeSpeaker  (dictation / teacher voice)
    python fetch_models.py --whisper       # Whisper base  (voice STT)
    python fetch_models.py --whisper-small # Whisper small (more accurate, ~0.5 GB)
    python fetch_models.py --llm           # Qwen2.5-3B    (board AI summary, ~1.8 GB)
    python fetch_models.py --vlm           # Qwen2-VL      (notebook titling, ~1.7 GB)
    python fetch_models.py --all           # all of the above

Requires internet and `huggingface_hub` (installed by `pip install -e .`).

Not handled here:
  - Hand-tracking model — auto-downloads at first run.
  - Sketch CNN (models/sketch_classifier.*) — copy it from a working install,
    or retrain with: python -m training.download_quickdraw && python -m training.train_sketch_cnn
  - PP-OCR (models/ppocr/) — copy that folder from a working install (its file
    layout is custom); see docs/model_download.md.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def _dir_has_xml(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.xml"))


def fetch_speaker() -> None:
    """WeSpeaker ResNet34 ONNX -> models/speaker/onnx__model.onnx (dictation)."""
    dest = MODELS / "speaker"
    if (dest / "onnx__model.onnx").exists():
        print("[speaker] already present — skipping")
        return
    from huggingface_hub import hf_hub_download
    repo = "onnx-community/wespeaker-voxceleb-resnet34-LM"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[speaker] downloading {repo} (~26 MB) ...", flush=True)
    onnx = hf_hub_download(repo_id=repo, filename="onnx/model.onnx")
    shutil.copy(onnx, dest / "onnx__model.onnx")           # the only file the app needs
    for cfg in ("config.json", "preprocessor_config.json"):
        try:
            shutil.copy(hf_hub_download(repo_id=repo, filename=cfg), dest / cfg)
        except Exception:
            pass
    print(f"[speaker] OK -> {dest / 'onnx__model.onnx'}")


def _snapshot(repo: str, dirname: str, what: str) -> None:
    dest = MODELS / dirname
    if _dir_has_xml(dest):
        print(f"[{dirname}] already present — skipping")
        return
    from huggingface_hub import snapshot_download
    print(f"[{dirname}] downloading {repo} ({what}) ...", flush=True)
    local = snapshot_download(repo_id=repo)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local, dest, dirs_exist_ok=True)
    print(f"[{dirname}] OK -> {dest}")


def fetch_whisper() -> None:
    _snapshot("OpenVINO/whisper-base-fp16-ov", "whisper-base-ov", "voice STT, ~150 MB")


def fetch_whisper_small() -> None:
    _snapshot("OpenVINO/whisper-small-fp16-ov", "whisper-small-ov", "voice STT, ~0.5 GB")


def fetch_llm() -> None:
    _snapshot("EmbeddedLLM/Qwen2.5-3B-Instruct-int4-sym-ov",
              "qwen2.5-3b-instruct-ov", "board summary, ~1.8 GB")


def fetch_vlm() -> None:
    _snapshot("cydxg/Qwen2-VL-2B-Instruct-OpenVINO-INT4",
              "qwen2-vl-2b-ov", "notebook titling, ~1.7 GB")


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch AirSketch model weights (not in git).")
    p.add_argument("--speaker", action="store_true", help="WeSpeaker (dictation)")
    p.add_argument("--whisper", action="store_true", help="Whisper base (voice)")
    p.add_argument("--whisper-small", action="store_true", help="Whisper small")
    p.add_argument("--llm", action="store_true", help="Qwen2.5-3B (board summary)")
    p.add_argument("--vlm", action="store_true", help="Qwen2-VL (notebook titling)")
    p.add_argument("--all", action="store_true", help="all of the above")
    args = p.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        print("ERROR: huggingface_hub is missing. Run: pip install -e .  (or pip install huggingface_hub)")
        return 1

    jobs = []
    if args.all:
        jobs = [fetch_speaker, fetch_whisper, fetch_whisper_small, fetch_llm, fetch_vlm]
    else:
        if args.speaker:        jobs.append(fetch_speaker)
        if args.whisper:        jobs.append(fetch_whisper)
        if args.whisper_small:  jobs.append(fetch_whisper_small)
        if args.llm:            jobs.append(fetch_llm)
        if args.vlm:            jobs.append(fetch_vlm)
        if not jobs:            jobs = [fetch_speaker]   # default: the dictation fix

    ok = True
    for job in jobs:
        try:
            job()
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  FAILED: {job.__name__}: {type(e).__name__}: {e}")
    print("\nDone." if ok else "\nDone (with errors — see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
