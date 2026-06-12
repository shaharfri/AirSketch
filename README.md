# AirSketch

**Air-drawing app with multi-stroke notebook, live snap-to-shape, CNN sketch classification, and optional Qwen-VL semantic enrichment.**

AirSketch merges two earlier projects:
- **AirDraw AR / AirNotes** — multi-stroke diagrams, live snap-to-shape, joint-angle gesture detection, HTML notebook export, Qwen2-VL semantic analysis.
- **Skysketch** — Quick, Draw! CNN sketch classifier, OpenVINO `InferenceEngine` abstraction, test suite, training pipeline, optional voice (Whisper) and image generation (LCM).

## Features

| Capability | Source | Status |
|---|---|---|
| Multi-stroke "diagram" model with auto-finalize | AirDraw | ✅ |
| Live snap-to-shape on pen-up (wobbly → clean) | AirDraw | ✅ |
| Joint-angle gesture detection + open-palm pause | AirDraw | ✅ |
| Per-stroke primitive classifier (line/arrow/circle/triangle/rectangle/curve) | AirDraw | ✅ |
| **CNN sketch classifier** (house, car, cat, tree, star, flower, sun, airplane, fish) | Skysketch | ✅ |
| OpenVINO `InferenceEngine` abstraction | Skysketch | ✅ |
| Quick, Draw! training pipeline (PyTorch → ONNX → OpenVINO IR) | Skysketch | ✅ |
| HTML notebook export with embedded thumbnails | AirDraw | ✅ |
| Qwen2-VL semantic enrichment (optional) | AirDraw | ✅ |
| Test suite (pytest) | Skysketch | ✅ Adapted |
| Voice input (Whisper STT) | Skysketch | ✅ Optional |
| Image generation (LCM img2img) | Skysketch | ✅ Optional |

## Analyzer chain

When a diagram finalizes, three tiers run in order:

1. **Per-stroke primitives** — geometric classification (instant, always available)
2. **Quick-Draw CNN** — classifies the rendered canvas as house / car / cat / tree / star / flower / sun / airplane / fish (requires trained model)
3. **Qwen2-VL** — semantic enrichment: topic, description, composition naming (requires `--vlm`)

The chain degrades gracefully — if the CNN model isn't trained, it falls through to the geometric tier; if the VLM isn't loaded, you still get CNN labels.

## Setup

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```powershell
# Default: hand tracking + primitive classifier + CNN (if model is present)
python -m airsketch.main

# Enable Qwen2-VL (downloads ~1.7 GB on first run)
python -m airsketch.main --vlm

# Force device for CNN / VLM
python -m airsketch.main --cnn-device NPU --vlm --vlm-device GPU

# Disable CNN even if model exists
python -m airsketch.main --no-cnn

# Adjust diagram finalization timeout
python -m airsketch.main --pause-seconds 5

# VDI camera flags (rotation + mirror)
python -m airsketch.main --no-rotate --mirror
```

## Classroom mode (teacher / student)

A lesson-time activity layer. **Phase 1** is the challenge game:

```powershell
# Geometry challenges (judged by the primitive classifier — works without the CNN)
python -m airsketch.main --classroom --theme geometry

# Object challenges (house, cat, tree, ... — needs the trained CNN)
python -m airsketch.main --classroom --theme objects

# Mixed
python -m airsketch.main --classroom --theme mixed

# Or the dedicated entry point
python -m airsketch.classroom --theme geometry
```

**How the game works:**
1. Click the window, press **SPACE** — the app announces a challenge ("Draw a TRIANGLE!") with a 3-2-1 countdown.
2. A student draws in the air. Strokes snap to clean shapes live.
3. Press **ENTER** to submit (or let the 25 s timer run out).
4. The **Judge** scores the drawing 0-100 and awards 0-3 stars:
   - Geometry targets → judged by the per-stroke primitive classifier
   - Object targets (house, cat, …) → judged by the Quick-Draw CNN
5. **Pass** → confetti + stars + "CORRECT!"; **Fail** → "TRY AGAIN!" + a hint outline of the target.
6. Running score, stars, and round count in the HUD; full scoreboard printed on quit.

| Key | Action |
|---|---|
| **SPACE** | Start the next challenge (from READY) |
| **ENTER** | Submit the drawing |
| **C** | Clear current drawing |
| **T** | Cycle theme (geometry → objects → mixed) |
| **V** | Toggle voice recording (with `--voice`) |
| **Q / ESC** | Quit (prints scoreboard) |

### Voice control (Phase 2 — Whisper STT)

The teacher can drive the game by voice instead of the keyboard:

```powershell
# One-time: fetch an OpenVINO Whisper model
python -m training.setup_whisper

# Run the classroom with voice enabled
python -m airsketch.main --classroom --voice
# or: python -m airsketch.classroom --voice
```

Press **V** to start/stop recording. Recognized spoken commands:

| You say | Action |
|---|---|
| "draw a triangle" / "triangle" | Start a triangle challenge |
| "draw a house" / "draw a cat" | Start an object challenge (CNN-judged) |
| "next" / "another one" | Random challenge |
| "submit" / "done" / "check" | Submit the current drawing |
| "clear" / "erase" | Clear the drawing |
| "geometry" / "objects" / "mixed" | Switch theme |
| (anything else) | Treated as dictation |

Voice **degrades gracefully**: if `sounddevice` isn't installed, no mic is present, or the Whisper model is missing, voice is disabled with a message and the keyboard still works. (On a VDI, mic redirection may need to be enabled in the client.)

Planned next phases: **board capture** (Qwen-VL reads the physical board into lesson notes), **lesson report** (HTML summary of transcript + board notes + scoreboard).

## Keyboard Controls (notebook mode)

| Key | Action |
|---|---|
| **Point** with index finger (others curled) | Pen down — draw |
| **Open palm** or **fist** | Pen up |
| **Hold SPACE** | Force pen-down (override gesture) |
| `N` | Finalize current diagram, start a new one |
| `C` | Clear unfinalized strokes |
| `Z` | Undo last stroke |
| `R` | Re-run analysis on most recent diagram |
| `E` | Export notebook to HTML + JSON |
| `S` | Save snapshot |
| `Q` / `ESC` | Quit (auto-exports notebook) |

## Training the CNN classifier

The merged project ships with the training pipeline from Skysketch:

```bash
# 1. Download Quick, Draw! data (~90MB)
python -m training.download_quickdraw

# 2. Train (~5-10 min on M1 Mac, similar on Intel CPU)
python -m training.train_sketch_cnn

# This produces:
#   models/sketch_classifier.xml
#   models/sketch_classifier.bin
#   models/class_names.json
```

After training, the main app picks up the CNN automatically. Categories: triangle, square, circle, house, car, tree, star, cat, flower, sun, airplane, fish.

To add new categories, edit `CATEGORIES` in `training/download_quickdraw.py` and retrain.

## Tests

```bash
pytest tests/ -v
```

Covered:
- `test_primitive_classifier.py` — line / arrow / circle / triangle / rectangle / curve / dot
- `test_gesture_detector.py` — `IndexPointingDetector` joint-angle logic, `PinchDetector` hysteresis
- `test_notebook.py` — strokes / live snap / diagram finalize / analyzer pipeline
- `test_sketch_classifier.py` — CNN preprocessing + classification (mocked)

## Project structure

```
AirSketch/
├── src/airsketch/
│   ├── __init__.py
│   ├── main.py              # state machine, main loop, HUD
│   ├── config.py            # AppConfig, enums
│   ├── camera.py            # Camera class (rotation/mirror for VDI)
│   ├── video_source.py      # Lower-level VideoCapture wrapper
│   ├── hand_tracker.py      # MediaPipe Hands (Tasks API)
│   ├── gesture_detector.py  # IndexPointingDetector + PinchDetector
│   ├── primitive_classifier.py
│   ├── beautifier.py        # Snap a stroke to a clean primitive
│   ├── stroke.py            # Stroke / Diagram / Status / Analysis dataclasses
│   ├── notebook.py          # Notebook session manager + async analyzer
│   ├── inference_engine.py  # OpenVINO Core wrapper (from Skysketch)
│   ├── sketch_classifier.py # Quick-Draw CNN classifier (from Skysketch)
│   ├── sketch_cnn.py        # PyTorch model definition (training)
│   ├── diagram_analyzer.py  # LocalAnalyzer + CNNAnalyzer + Qwen-VL + ChainedAnalyzer
│   ├── shape_recognizer.py  # Whole-diagram geometric recognizer (legacy fallback)
│   ├── effects.py           # Visual effects chain
│   ├── overlay.py           # State badges, transition banners
│   ├── exporter.py          # HTML + JSON export
│   ├── utils.py             # Drawing helpers
│   ├── voice.py             # Whisper STT (optional)
│   ├── image_gen.py         # LCM img2img (optional)
│   └── llm_chat.py          # Local Qwen2.5 text LLM (optional)
├── tests/
├── training/
│   ├── download_quickdraw.py
│   └── train_sketch_cnn.py
├── docs/
│   ├── intel_npu_setup.md
│   └── model_download.md
├── models/                  # CNN + Qwen + Whisper IR weights live here
├── outputs/                 # Saved snapshots + HTML exports
├── requirements.txt
├── pyproject.toml
└── README.md
```

## OpenVINO devices

All inference components support OpenVINO device targeting:

```bash
# Get devices available on this machine
python -c "import openvino as ov; print(ov.Core().available_devices)"

# Use NPU for everything that supports it (Intel AI PC)
python -m airsketch.main --cnn-device NPU --vlm --vlm-device NPU
```

The components map to Intel hardware as follows:

- **Hand tracking** (MediaPipe TFLite) — CPU only (could be moved to NPU with an OpenVINO hand-pose conversion, future work)
- **CNN sketch classifier** — CPU / GPU / NPU
- **Qwen2-VL** — CPU / GPU / NPU (GPU recommended for latency)
- **Whisper STT** (optional) — CPU / GPU / NPU
- **LCM image gen** (optional) — GPU recommended

See [docs/intel_npu_setup.md](docs/intel_npu_setup.md) for full NPU setup.

## Lineage

- **AirDraw AR / AirNotes** — built first; focused on multi-stroke diagrams for hybrid classrooms / video calls
- **Skysketch** — built second; focused on instant shape detection + lightweight CNN
- **AirSketch** — the merge. Takes AirDraw's notebook UX + Skysketch's CNN.

Both predecessors remain in their respective directories for reference.
