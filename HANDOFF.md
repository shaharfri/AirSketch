# AirSketch — Session Handoff

> Read this top-to-bottom to inherit the full context. It captures the project,
> what's built, what works, the environment, every bug we already fixed (so you
> don't re-chase them), and the next steps.

---

## TL;DR — current status

**AirSketch** is an air-drawing app (track the index finger via webcam, draw shapes
in the air) with a **classroom teacher/student mode**. It's a merge of two earlier
prototypes (AirDraw AR + Skysketch).

- **Phase 1 (challenge game): DONE & working.** Teacher issues a "draw a X" challenge,
  student draws in the air, the app judges it (0–100 score + 0–3 stars) and plays a
  success/failure animation. **Retry added:** after a miss, press **R** (or say "again")
  to re-attempt the same challenge; the miss is dropped (doesn't count against the student).
- **Phase 2 (teacher voice): DONE & working on a real PC.** Press **V**, speak a command
  ("draw a triangle"), Whisper transcribes it, the game responds. **Voice now stable** —
  the intermittent `!!!!`/garbage was a real bug (uninitialized `sd.rec` buffer), now fixed.
  On the VDI the redirected mic may still deliver no audio (genuine VDI limitation).
- **CNN sketch classifier: TRAINED** (96% test accuracy) — recognizes house/cat/tree/
  star/flower/sun/airplane/fish/car in addition to geometric primitives.
- **Phase 3 (board capture): DONE & verified.** Press **B** (or say "read the board") to
  photograph the whiteboard and transcribe it. Uses **PP-OCR on OpenVINO** (NOT a VLM —
  the Qwen2-VL-2B cannot read text). Optional `--understand` adds a **Qwen2.5-3B** LLM that
  summarizes + structures the OCR text (topic, summary, cleaned key points).
- **Phase 4 (lesson report): DONE.** On exit the classroom writes a self-contained
  `outputs/lesson_<ts>.html` — challenge scoreboard + each board capture (embedded image,
  LLM summary/topic, key points, collapsible raw OCR) + teacher narration.
  `exporter.export_lesson_report()`.
- **Teacher voice (speaker recognition): DONE — needs real-voice threshold tuning.** With
  `--teacher-voice`, the **D key is the one Dictation button**: 1st press LEARNS your voice
  (say a sentence → WeSpeaker embedding profile, saved to `models/teacher_voice.json`); press
  D again to DICTATE. Dictation is **gated to the learned voice** (cosine ≥ `--speaker-threshold`,
  default 0.5) and accepted text becomes lesson narration in the report. STT is the existing
  Whisper. So learning + dictating needs **two D presses** (learn, then dictate). Plumbing
  verified headless; accuracy/threshold need testing on a real mic+voice.
- **175 tests pass** (105 prior + 26 OpenVINO hand tracker + launcher + Hebrew).
- **Hebrew language toggle** (`--lang he` / GUI dropdown): Whisper STT, voice commands,
  board LLM, and on-screen RTL text all support Hebrew on OpenVINO. Board OCR stays
  English (no OV Hebrew model). See SESSION_HANDOFF.md "Hebrew language support".

Last things shipped this session: voice uninitialized-buffer fix, retry, Phase 3 board
capture (PP-OCR/OpenVINO) + Phase B understanding LLM (Qwen2.5-3B). Removed the dead
unused `llm_chat.py` (old Qwen text LLM, never wired).

---

## Where everything lives

| Thing | Path |
|---|---|
| **Main project (work here)** | `C:\ssh-web-server-python\.claude\worktrees\AirSketch` |
| Distributable zip (for the user's PC) | `C:\ssh-web-server-python\.claude\worktrees\AirSketch_dist.zip` (~142 MB) |
| Predecessor: AirDraw AR / AirNotes | `C:\ssh-web-server-python\.claude\worktrees\vigilant-napier-1e175c\airdraw_ar` |
| Predecessor: Skysketch | `C:\ssh-web-server-python\.claude\worktrees\Skysketch` |
| Backups of airdraw_ar (pre-refactors) | `...\vigilant-napier-1e175c\airdraw_ar_backup_*` |

Note: the shell's default cwd in this session was the `vigilant-napier-1e175c` worktree,
but **all AirSketch work happens in the `AirSketch` worktree**. Commands were run with
`cd <AirSketch>` and `PYTHONPATH=src` (or after `pip install -e .`).

---

## What AirSketch is (lineage)

- **AirDraw AR / AirNotes** (built first): multi-stroke "notebook", live snap-to-shape,
  joint-angle gesture detection, HTML export, Qwen2-VL semantic analysis.
- **Skysketch** (built second): a trained Quick-Draw **CNN classifier**, a clean OpenVINO
  `InferenceEngine`, a **test suite**, a **training pipeline**, plus optional Whisper voice
  and LCM image generation.
- **AirSketch** = the merge: AirDraw's notebook/UX + Skysketch's CNN/tests/training/voice.

The whole thing is built to run **OpenVINO inference on Intel CPU/GPU/NPU** (the original
brief). Device is selectable via `--cnn-device` / `--vlm-device` (`AUTO|CPU|GPU|NPU`).

---

## Architecture (the `airsketch` package, under `src/airsketch/`)

Core (from AirDraw AR):
- `config.py` — `AppConfig` dataclass + enums. Many defaults live here.
- `camera.py` — `Camera` context-manager (webcam/video) with `rotate_180` + `mirror`.
- `video_source.py` — lower-level VideoCapture wrapper (multi-backend on Windows).
- `hand_tracker.py` — `HandTracker` ABC + `MediaPipeHandTracker` (Tasks API) + `create_hand_tracker(cfg)` factory. Returns 21 landmarks + index/thumb/wrist + hand_size. Auto-downloads `models/hand_landmarker.task`.
- `hand_tracker_ov.py` — `OpenVINOHandTracker`: BlazePalm palm detector + 21-pt landmark model on the **OpenVINO runtime** (drop-in for MediaPipe; same `process()->HandResult`). Pure-numpy/cv2 host code (SSD anchors, decode, weighted NMS, rotated-ROI crop, inverse landmark map, detect↔track state machine). Reads the two `.tflite` inside `models/hand_landmarker.task` directly (auto-extracts to `models/hand_ov/`). **Now the DEFAULT backend** (MediaPipe = fallback via `--hand-backend mediapipe`); device via `--hand-device {AUTO,CPU,GPU,NPU}` (graceful CPU fallback); `--hand-debug` overlay. **Both models normalize input to [0,1] (verified — not [-1,1]); the landmark presence/handedness outputs are already probabilities — do NOT sigmoid them.**
- `gesture_detector.py` — `IndexPointingDetector` (angle-based, the reliable one) + `PinchDetector`.
- `primitive_classifier.py` — per-stroke geometric primitives (line/arrow/circle/ellipse/rectangle/triangle/polygon/curve/dot).
- `beautifier.py` — `beautify_diagram` + `points_for_primitive` (live snap-to-shape rendering).
- `stroke.py` — `Stroke`, `Diagram`, `DiagramAnalysis`, `DiagramStatus` dataclasses.
- `notebook.py` — `Notebook` session manager (strokes → diagrams) + async analyzer executor.
- `diagram_analyzer.py` — analyzer chain: `LocalAnalyzer` (geometric) → `CNNAnalyzer` (Quick-Draw CNN) → `OpenVINOQwenVLAnalyzer` (Qwen2-VL) via `ChainedAnalyzer`; `create_analyzer()` factory degrades gracefully. Also `ensure_ov_tokenizer()` auto-converts a HF tokenizer to OpenVINO IR.
- `shape_recognizer.py` — whole-diagram geometric recognizer (legacy fallback).
- `exporter.py` — HTML + JSON notebook export.
- `utils.py` — drawing helpers (neon line, panels, text). `draw_panel` is now **rounded +
  bounds-safe** (blends only its sub-region; added `radius`/`accent` kwargs, signature
  backward-compatible) + `fill_rounded_rect`/`stroke_rounded_rect`. Classroom HUD restyled with
  a cohesive `UI_*` palette (accent bars, logo dot, rounded timer/banners) in `classroom/app.py`
  — **purely cosmetic, no logic/keys changed**; notebook mode inherits the rounded panels too.
  All key hints are consolidated into ONE bottom **`_draw_key_legend`** bar (per-state), and
  ALL center banners go through ONE priority-picked **`_draw_center_overlay`** (recording >
  board-reading > READY/ANNOUNCE) so banners can never stack ("window over a window" fixed).
  Don't re-add `_center_banner` calls elsewhere — route through `_draw_center_overlay`.

From Skysketch:
- `inference_engine.py` — OpenVINO Core wrapper.
- `sketch_classifier.py` — `SketchClassifier` (Quick-Draw CNN via OpenVINO). Expects **bright strokes on black bg** (Quick-Draw format).
- `sketch_cnn.py` — PyTorch model def (training only).
- `voice.py` — `VoiceRecorder` (Whisper STT). Uses `sd.rec()` into a **zero-filled** buffer
  (must stay zero-filled — see bug journey #13) + NaN/Inf scrub before transcription.
- `image_gen.py` — optional LCM sketch→image, NOT wired in. (`llm_chat.py` was removed — dead.)
- `effects.py`, `overlay.py` — copied in; `effects.py` particles inspired the celebration.

Board capture (Phase 3 / B):
- `ocr_reader.py` — `PPOCROpenVINOReader`: PaddleOCR PP-OCRv5 detection + English recognition
  (ONNX) on the OpenVINO runtime. DB-detection postproc + CTC decode, written by hand (no
  paddlepaddle dep). Models in `models/ppocr/`. `assemble_text()` orders lines for the note.
- `lesson_llm.py` — `LessonUnderstander`: Qwen2.5-3B-Instruct INT4 on OpenVINO GenAI. Lazy.
  `understand(text)` → summary/topic/key_points/corrected (`parse_understanding` is JSON-tolerant).
- `board_capture.py` — `BoardCapturer`: OCR (always) + optional LLM enrichment → `BoardNote`.

Teacher voice (speaker recognition, `--teacher-voice` / E):
- `speaker_id.py` — `compute_fbank()` (numpy 80-dim Kaldi-style log-mel + CMN), `SpeakerEmbedder`
  (WeSpeaker ResNet34 ONNX on OpenVINO → 256-d embedding), `SpeakerProfile` (enroll/save/load,
  cosine `matches`), `build_profile()`. `voice.py` now exposes `last_audio`; `voice_controller.py`
  adds `take_result()` (text+audio) used by the classroom loop instead of `poll()`.

Classroom mode (`src/airsketch/classroom/`):
- `challenge_engine.py` — curriculum (geometry/objects/mixed), `Challenge`, `ChallengeResult`, scoring, `next_challenge()` / `challenge_for(target)`.
- `judge.py` — judges a drawn `Diagram` vs target. Geometry → primitive classifier; objects → CNN on a black-bg render (`render_for_cnn`). 0–100 score, 0–3 stars, square≈rectangle/circle≈ellipse synonyms.
- `celebration.py` — `Celebration`: confetti+stars success, shake+hint failure.
- `app.py` — `ClassroomApp`: the game state machine (READY → ANNOUNCE → DRAWING → RESULT) + main loop + HUD + voice integration.
- `voice_commands.py` — `parse_command(text) -> Intent` (DRAW/NEXT/SUBMIT/CLEAR/SET_THEME/DICTATION), with synonyms + **fuzzy matching**.
- `voice_controller.py` — `VoiceController`: wraps `VoiceRecorder`, mic probe with timeout, `poll()` returns an Intent once per utterance.

Entry points:
- `python -m airsketch.main` — the freehand notebook.
- `python -m airsketch.main --classroom [--voice] [--theme ...]` — classroom game.
- `python -m airsketch.classroom [--voice] [--theme ...]` — classroom directly.

---

## What's built and working

- **Notebook mode**: freehand drawing, primitive recognition, CNN object recognition (now that the model is trained), optional Qwen-VL with `--vlm`, HTML export.
- **Classroom Phase 1**: full challenge game, keyboard-driven, with CNN-judged object
  challenges and animations. Verified: house drawing → CNN "house" 100% → 3 stars.
- **Classroom Phase 2 (voice)**: press V → "Listening…" (camera pauses) → speak →
  press V → transcribe → command fires. Works on a real PC mic.

---

## Known limitations / gotchas (IMPORTANT — do not re-chase)

1. **VDI voice may still capture nothing.** The intermittent `!!!!` that ALSO happened on the
   personal PC was a real bug (uninitialized buffer — fixed, see bug journey #13). After the
   fix, a true-silent capture (e.g. VDI redirected mic delivering no audio) yields `peak≈0` →
   a clean empty result, not garbage. If the VDI mic genuinely passes no audio, that part is a
   **VDI redirection limitation**; voice is best on a personal PC.
2. **Camera + mic simultaneously stalls the camera** on integrated camera+mic subsystems
   AND on the VDI. **Fix already in place**: while recording, the classroom loop reuses the
   last frame and skips `cam.read()` + hand tracking (see `app.py` `recording_now`). Don't
   undo this.
3. **`os._exit(0)`** is called in `main.py` and `classroom/__main__.py` finally-blocks (to
   kill MediaPipe/OpenVINO non-daemon threads on exit). It **hides tracebacks**. `run()` in
   `app.py` now has an `except Exception: traceback.print_exc(); flush` so crashes are
   visible — keep it. **Classroom `run()`'s finally** releases the camera on a SIDE THREAD
   (`cam.release()` + `tracker.release()`) so the slow Windows-MSMF release overlaps with the
   report write, waits up to 4 s for it (`rel.join(4.0)` — this is what turns the camera **LED
   off**; `os._exit` alone did NOT free the webcam on real hardware), then `os._exit(0)`. Keep
   the release bounded — never let it block exit indefinitely.
4. **Windows console is cp1252** → unicode (→, ★, emoji) in `print()` crashes with
   `UnicodeEncodeError`. Use ASCII in prints, or run with `PYTHONIOENCODING=utf-8`.
5. **`rotate_180` default is `True`** (the VDI camera is upside down). On a **normal PC use
   `--no-rotate`**. `--mirror` is optional (selfie view).
6. **`live_snap_enabled` default is `False`** (user wanted freehand kept as-drawn). Judging
   still works because the judge/notebook classify raw strokes at judge-time. `--snap` re-enables.
7. **`TaskStop` only kills the shell wrapper, not the Python child** — a killed training run
   left a zombie eating CPU. If you kill a long python job, also kill the `python.exe` child
   (e.g. via `Get-CimInstance Win32_Process` matching the command line).
8. **`pip install -e .` does NOT install `sounddevice`** (it's an optional `[voice]` extra).
   Voice needs `pip install sounddevice` explicitly.
9. **Whisper hallucinates on silence** (returns "Thanks for watching!" / "!!!!"). That's
   normal for non-speech audio and confirms the pipeline runs; it's not a bug.

---

## Environment

- **VDI** (`C:\ssh-web-server-python\...`): system **Python 3.13**, ALL deps installed
  (opencv, mediapipe, numpy, openvino, openvino-genai, openvino-tokenizers, huggingface_hub,
  transformers, sentencepiece, tiktoken, torch+torchvision CPU, sounddevice). Models present.
  Run with `PYTHONPATH=src` or after `pip install -e .`.
- **User's personal PC**: had **Python 3.14** (works, but 3.11 recommended for native libs);
  uses a `.venv`; needed `pip install -e .` + `pip install sounddevice`. PowerShell execution
  policy had to be set: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. `py` launcher works.

### Models (in `AirSketch/models/`)
- `sketch_classifier.xml` + `.bin` + `class_names.json` — **trained CNN, 96% test acc**,
  12 classes: triangle, square, circle, house, car, tree, star, cat, flower, sun, airplane, fish.
- `hand_landmarker.task` — MediaPipe hand model (~7.5 MB).
- `whisper-base-ov/` — OpenVINO Whisper (downloaded from `OpenVINO/whisper-base-fp16-ov`, ~148 MB).
- `ppocr/` — PP-OCRv5 detection (`detection__v5__det.onnx`, ~88 MB) + English recognition
  (`languages__english__rec.onnx`, ~8 MB) + dict, from HF `monkt/paddleocr-onnx`. Used by
  `ocr_reader.py` via OpenVINO (reads ONNX natively). Board transcription.
- Qwen2.5-3B understanding LLM: HF cache `models/models--EmbeddedLLM--Qwen2.5-3B-Instruct-int4-sym-ov`
  (~1.8 GB). Used by `lesson_llm.py` (`--understand`). Complete OV IR + tokenizer (no conversion).
  NOTE: the loaders now **prefer a bundled plain dir** if present — `lesson_llm` checks
  `models/qwen2.5-3b-instruct-ov` and `_try_vlm_analyzer` checks `models/qwen2-vl-2b-ov` (flat,
  symlink-resolved copies for the self-contained dist) BEFORE falling back to HF download. Those
  plain dirs are what the distributable zip ships.
- `speaker/` — WeSpeaker ResNet34 (`onnx__model.onnx`, ~26 MB) + configs, from HF
  `onnx-community/wespeaker-voxceleb-resnet34-LM`. Used by `speaker_id.py` via OpenVINO.
  Enrolled profile saved to `models/teacher_voice.json`.
- Qwen2-VL VLM: HF cache `models/models--cydxg--Qwen2-VL-2B-Instruct-OpenVINO-INT4` (~1.7 GB),
  repo `cydxg/Qwen2-VL-2B-Instruct-OpenVINO-INT4`. **Only used by notebook `--vlm`** diagram
  enrichment. NOTE: this model **cannot read text** — proven via `board_capture` testing — which
  is why Phase 3 uses PP-OCR, not the VLM. Tokenizer auto-converted by `ensure_ov_tokenizer()`.

### Training pipeline (`AirSketch/training/`)
- `download_quickdraw.py` — fetches 12 Quick-Draw categories → `data/quickdraw/quickdraw_split.npz`.
- `train_sketch_cnn.py` — trains the CNN, exports OpenVINO IR. **Augmentation was vectorized**
  (was per-image loop = far too slow); now ~8 s/epoch on CPU. Defaults to 15 epochs.
- `setup_whisper.py` — downloads a pre-converted OpenVINO Whisper model.

---

## How to run

```powershell
# VDI (Python 3.13, deps installed) — from the AirSketch dir:
python -m pip install -e .                 # once (registers the package)
python -m airsketch.main --classroom                 # game, keyboard (VDI camera needs default rotation)
python -m airsketch.main --classroom --theme objects # CNN object challenges
python -m airsketch.main --classroom --board         # + whiteboard OCR (PP-OCR/OpenVINO, ~96MB)
python -m airsketch.main --classroom --board --understand  # + Qwen2.5-3B summary (downloads ~1.8GB)
python -m airsketch.main --classroom --voice --teacher-voice  # enroll teacher (E), gate dictation
python -m airsketch.main --classroom                          # hand tracking on OpenVINO (DEFAULT now)
python -m airsketch.main --classroom --hand-device GPU        # try GPU/NPU (falls back to CPU)
python -m airsketch.main --classroom --hand-debug             # overlay palm-score/presence/state
python -m airsketch.main --classroom --hand-backend mediapipe # force the MediaPipe fallback
python -m airsketch.main --vlm                       # notebook + Qwen-VL (downloads 1.7GB first time)

# Test board OCR/LLM with NO camera (great for debugging):
python -m airsketch.board_capture sample_board.png               # OCR only
python -m airsketch.board_capture sample_board.png --understand  # OCR + LLM

# Personal PC:
py -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install sounddevice
python -m airsketch.main --classroom --voice --no-rotate --mirror   # voice works here
```

Classroom controls: **SPACE** start · point index finger to draw · **ENTER** submit ·
**C** clear · **R** retry (after a miss) · **T** theme · **V** talk/commands (press to start,
press again to stop) · **B** read the board (`--board`) · **D** dictation: 1st press learns
your voice, then press D to dictate · **L** (re)learn / switch to a new voice
(`--teacher-voice`) · **Q** quit. (All keys also shown in the in-app bottom legend.)

Voice commands: "draw a triangle/house/cat/…", "next", "submit", "clear",
"geometry"/"objects"/"mixed". Fuzzy-matched, English-forced, vocab-biased.

---

## Test status

`PYTHONPATH=src python -m pytest tests/ -q` → **175 passed**.
Files: `test_primitive_classifier`, `test_gesture_detector`, `test_notebook`,
`test_sketch_classifier`, `test_classroom` (+retry), `test_voice_commands` (+retry, board),
`test_voice_controller`, `test_board_capture`, `test_ocr_reader`, `test_lesson_llm`,
`test_lesson_report`, `test_speaker_id`, `test_hand_tracker_ov` (anchors/decode/NMS/ROI
math + real-model integration: blank-frame-not-visible regression guard, contract, device
fallback — integration tests auto-skip if OpenVINO/models are absent),
`test_launcher` (GUI flag-builder + windowed-stdout guards),
`test_hebrew` (Hebrew commands/morphology, Whisper token, LLM prompt, RTL renderer).

---

## The bug journey (already fixed — don't repeat)

1. Camera wouldn't open on VDI → multi-backend `VideoCapture` (CAP_ANY/MSMF/DSHOW).
2. MediaPipe legacy `mp.solutions` gone on Py3.13 → switched to **Tasks API** + model download.
3. Video upside-down on VDI → `rotate_180` default True (+ `--no-rotate` for PCs).
4. Camera LED / process won't die → `os._exit(0)` after cleanup.
5. Qwen repo `OpenVINO/Qwen2-VL-2B-Instruct-int4-ov` didn't exist → use `cydxg/...`.
6. Qwen load failed "tokenizer not available" → `ensure_ov_tokenizer()` auto-converts (needs `transformers`).
7. VLM gave wrong titles for simple shapes → made local/CNN authoritative, VLM only enriches.
8. CNN training painfully slow → **vectorized augmentation**; also killed a **zombie training process**.
9. Voice V-key: tried no-mic theory, then hang theory, then sd.rec switch, then camera-pause —
   **the actual bug was a missing `import math` in `app.py`** (REC-dot animation). `os._exit`
   was hiding the traceback the whole time. Lesson: surface tracebacks first.
10. Voice "ignores after many tries" → **result-ready flag** in `VoiceRecorder` + simplified `poll()`.
11. Whisper heard German/garbage → forced **English + initial_prompt + hotwords** vocab bias.
12. Strict command matching → **fuzzy matching** (difflib) in `voice_commands.py`.
13. **Voice intermittently returned `!!!!`/garbage even while speaking** → root cause was
    `sd.rec()` handing back an **uninitialized** buffer (`np.empty`); unfilled frames held
    NaN/Inf/huge floats that poisoned Whisper (and the zero-trim only stripped real zeros, so
    the garbage tail survived). Fix: pre-allocate a **zero-filled** buffer + pass it via
    `out=`, plus `np.nan_to_num`/clip scrub. Diagnosing it needed an amplitude print (peak/rms),
    since the tell was peak=NaN / 5e23. Don't revert the zero-fill.
14. **Tried Qwen2-VL-2B for board OCR → it can't read text** (only sees coarse color blobs,
    hallucinates coordinates; no prompt/chat/tag variant helped). Switched Phase 3 to PP-OCR
    on OpenVINO (accurate) + Qwen2.5-3B to summarize. Don't re-attempt VLM-based transcription
    with the 2B model.

---

## Open items / next steps

1. **Tune teacher-voice on real hardware:** the D-button flow (learn → dictate) is built +
   unit-tested, but `--speaker-threshold` (0.5) + single-utterance learning need validation with
   a real mic+voice. The D recording pauses the camera (camera+mic stall constraint). If it
   accepts/rejects wrongly, adjust the threshold (higher = stricter). **Press L to (re)learn /
   switch the active voice anytime** (overwrites `models/teacher_voice.json`) — D never prompts,
   it just dictates with the current voice; L is the explicit "change whose voice is recognized"
   action (a different teacher can take over). The report is now written **incrementally** (after every
   challenge / board capture / dictation) to one `outputs/lesson_<ts>.html` per session, plus a
   final write on exit — so a forced kill or stalled-camera exit no longer loses it. Window-close
   (X) now breaks the loop cleanly too. (Underlying `cam.read()` can still block after a mic
   recording on integrated camera+mic hardware — the incremental write is the safety net.)
2. **Voice accuracy lever:** if whisper-base still misses too often, add a
   `--whisper-model small` option (more accurate, ~3× slower, bigger download). `setup_whisper.py`
   already supports passing a different repo.
3. **Judging leniency (open question for the user):** with live-snap OFF, freehand triangles
   sometimes judge as "curve"/fail. Could lower the pass threshold or loosen triangle tolerance
   in `judge.py` / `primitive_classifier.py` if the user finds it too strict.
4. **Board OCR/LLM levers:** OCR is English-only (`models/ppocr/languages__english`); other
   `rec.onnx` languages exist in the same HF repo. The PP-OCR ONNX can later be converted to
   OpenVINO IR (`ovc`) — already runs on OV today. LLM model is swappable via `cfg.board_llm_model_id`.
5. **Optional Skysketch feature not yet wired:** `image_gen.py` (LCM sketch→image) is present
   but unused. (`llm_chat.py` was removed — it was dead.)
6. **Hand tracking on OpenVINO — DONE, DEFAULT, live-validated.** Implemented as
   `hand_tracker_ov.py` (`OpenVINOHandTracker`); now the default backend (MediaPipe is the
   automatic fallback via `--hand-backend mediapipe`). Validated live on the VDI (air-drawing
   scored real strokes; ~108 FPS CPU). `--hand-device {AUTO,CPU,GPU,NPU}` + `--hand-debug`.
   **Remaining (optional):** GPU/NPU on real Intel hardware (CPU was the only dev device);
   drop the `mediapipe` dep (only the fallback now); pre-convert `.tflite`→IR for NPU/faster
   load. Two bugs fixed in bring-up: input norm is [0,1] not [-1,1]; presence/handedness are
   already probabilities (don't sigmoid). See `OPENVINO_HAND_TRACKING_PLAN.md`.

---

## The user / context

- User: **shaharfr@checkpoint.com**. Works on a **Check Point VDI** (Citrix-style, camera+mic
  redirected) and also a **personal Windows PC** (OneDrive path under `Documents\study\AirSketch_dist`).
- The user is iterating feature-by-feature, prefers seeing a plan before big work, and is happy
  to test on real hardware. They explicitly asked to disable live-snap by default and to make
  the classroom voice game. They're patient with multi-step debugging but value getting to a
  working result.
- Confidentiality: this is internal Check Point work — keep artifacts within approved tooling.
