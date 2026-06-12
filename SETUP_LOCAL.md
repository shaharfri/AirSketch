# Running AirSketch on your own PC

This package contains everything to run AirSketch **without a VDI** — which fixes
the camera/mic freezes caused by VDI multimedia redirection.

## What's included
- All source code, tests, training scripts, docs
- The **trained CNN** (`models/sketch_classifier.*`, `class_names.json`)
- The **MediaPipe hand model** (`models/hand_landmarker.task`)
- The **Whisper voice model** (`models/whisper-base-ov`) — so voice works offline

## What's NOT included (re-downloadable on demand)
- The Qwen2-VL model (~1.7 GB) — only needed for `--vlm`. It downloads the first
  time you pass `--vlm`.
- The raw Quick, Draw! dataset — only needed if you want to re-train the CNN.

## Setup (Windows PowerShell)

```powershell
# Requires Python 3.10 or 3.11
cd AirSketch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

(macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`)

## Run

```powershell
# Classroom challenge game (geometry — primitive-judged)
python -m airsketch.main --classroom --theme geometry

# Object challenges (house, cat, tree, ... — CNN-judged; the trained model is included)
python -m airsketch.main --classroom --theme objects

# With voice (mic should "just work" off-VDI):
python -m airsketch.main --classroom --voice

# The freehand notebook
python -m airsketch.main

# Add Qwen-VL semantic analysis (downloads ~1.7 GB first time)
python -m airsketch.main --vlm
```

## Notes for a physical PC
- No VDI = no camera/mic redirection conflict, so the **V-key voice hang should be
  gone** and the camera should not need the `--no-rotate` / rotation workarounds
  (drop `--no-rotate`; add `--mirror` only if you want a mirror view).
- If your webcam index isn't 0: `--camera 1` (etc.).
- Run the tests to confirm the environment: `pytest tests/ -q` (expect 71 passing).

## Controls (classroom)
SPACE = start challenge · ENTER = submit · C = clear · T = theme · V = talk · Q = quit
