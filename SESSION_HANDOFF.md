# AirSketch — Session Handoff (read me first)

> Purpose: a fresh Claude Code session can read this single file and inherit the
> full context **and the working stance** of the previous session. It captures the
> project, the honest status, the OpenVINO reality, what we changed, the gotchas,
> and how the user likes to collaborate. Pair with `HANDOFF.md` (terser, code-focused)
> and `OPENVINO_HAND_TRACKING_PLAN.md` (the one open design doc).

---

## 0. Point of view / how to work here (important)

- **Be honest about status. Do not overclaim.** Mark uncertain things "to be verified".
  We explicitly distinguish *works* vs *partial* vs *missing/planned*.
- **Verify, don't assume.** When we said "OpenVINO", we proved it by actually loading/
  running models (OCR, LLM, speaker) — not by trusting labels. Prefer to run/test.
- **Consult before heavy/irreversible actions.** The user wants to choose models and
  approve big downloads / dependency installs (we used AskUserQuestion for OCR engine,
  LLM size, speaker model, install location). Don't silently pull multi-GB models or
  mutate the working environment.
- **Plan before big work.** Lay out a short plan; the user likes seeing it. For a large
  new feature (e.g. the hand-tracking port) write/confirm a design doc first.
- **Respect the environment.** This is internal Check Point work — keep artifacts on
  approved storage, never disable TLS/SSL or integrity checks to "make installs work"
  (we hit an npm EINTEGRITY behind the corporate proxy and pivoted instead of bypassing).
- **Don't break working features.** All UI/refactor changes this session were behavior-
  preserving; 105 tests stayed green throughout.
- **Slides/docs style the user likes:** very low text on slides, talk-don't-read,
  details in speaker notes; clean professional visuals.

---

## 1. What AirSketch is

An **air-drawing classroom app**: a webcam tracks the index finger, the student draws
shapes in the air, and a **teacher/student challenge game** judges the drawing (0–100
score + 0–3 stars) with celebration animations. Around that core: **voice commands**,
**whiteboard capture** (OCR → AI summary), **speaker-gated teacher dictation**, and an
auto-generated **HTML lesson report**. Technical north star: **run all AI on Intel via
OpenVINO (CPU/GPU/NPU)**. It's a merge of two earlier prototypes (AirDraw AR + Skysketch).

**Main work dir:** `C:\ssh-web-server-python\.claude\worktrees\AirSketch` (this repo).
The shell's default cwd may be a different worktree — always operate in `AirSketch`,
run with `PYTHONPATH=src` or after `pip install -e .`.

---

## 2. Status — Works / Partial / Missing (honest)

**WORKS (implemented + tested, several verified live this session):**
- Hand tracking + air drawing; index-pointing pen gesture.
- Shape recognition (geometric) + object recognition (Quick-Draw **CNN, 96%**, 12 classes).
- Challenge game: state machine, scoring, stars, celebration, **retry (R)**.
- Voice commands (Whisper) — the intermittent-failure bug is fixed (see gotchas).
- Whiteboard OCR (**PP-OCR**) — verified near-perfect transcription on a test board.
- AI lesson summary (**Qwen2.5-3B**) — verified correct topic/summary.
- Speaker-gated dictation (D learn/dictate, **L = learn/switch a new voice**).
- Self-contained **HTML lesson report** (written incrementally → survives a forced exit).
- **Hand tracking on OpenVINO — DEFAULT, validated live on the VDI.**
  `OpenVINOHandTracker` is now the default backend (`hand_tracker_backend="openvino"`);
  MediaPipe is the automatic fallback if OV fails to load. Confirmed live: air-drawing
  scored real strokes (triangle 92%, rectangle 90%) with `[hand] OpenVINO hand tracker on
  CPU`. ~108 FPS CPU. Closes the last non-OV gap — the whole app now runs on OpenVINO.
- **131 pytest tests pass** (105 prior + 26 new for the OV hand tracker).

**PARTIAL / TO BE VERIFIED:**
- OpenVINO hand tracking on **GPU/NPU** — not testable in the dev env (only CPU present).
  Graceful CPU fallback is wired; both models are fixed-shape (good NPU fits, no reshape).
- Teacher-voice accuracy: pipeline works; `--speaker-threshold` (0.5) + single-utterance
  enrollment need tuning on a real mic.
- NPU: OpenVINO supports it, but dynamic-shape models (PP-OCR detection, Whisper, the LLM)
  may not compile on NPU and fall back to CPU/GPU; fixed-shape models (CNN, speaker, the
  future hand models) are the good NPU fits.
- Notebook VLM (Qwen2-VL) "reads" diagrams but **cannot read text** — demoted to enrich-only.

**MISSING / PLANNED:**
- LCM image generation (`image_gen.py`) present but not wired.
- (Optional) drop the `mediapipe` dependency once OV hand tracking is signed off live and
  made the default; pre-convert the two `.tflite` to IR with `ovc` for faster load / NPU.

---

## 3. The OpenVINO reality (state this precisely)

"OpenVINO" here means the code **already imports `openvino` / `openvino_genai` and executes
the models on the OpenVINO runtime at runtime** — NOT a "convertible later" claim. Device
(`CPU/GPU/NPU`) is a live arg via `--cnn/ocr/llm/vlm-device` flags.

| Model | Call site | On OpenVINO at runtime? |
|---|---|---|
| CNN | `inference_engine.py` `ov.Core().compile_model` | yes |
| Whisper | `voice.py` `ov_genai.WhisperPipeline` | yes |
| PP-OCR | `ocr_reader.py` `core.compile_model(core.read_model(onnx))` | yes (verified) |
| Qwen2.5-3B | `lesson_llm.py` `ov_genai.LLMPipeline` | yes (verified) |
| WeSpeaker | `speaker_id.py` `core.compile_model(core.read_model(onnx))` | yes (verified) |
| Qwen2-VL | `diagram_analyzer.py` `ov_genai.VLMPipeline` | yes |
| **Hand tracking (default)** | `hand_tracker_ov.py` `OpenVINOHandTracker` | **yes — DEFAULT, validated live on the VDI** |
| Hand tracking (fallback) | `hand_tracker.py` MediaPipe `HandLandmarker` (TFLite) | no — automatic fallback only if OV fails to load (`--hand-backend mediapipe`) |

Formats, both ingested by OpenVINO directly (no external conversion step needed):
- **OpenVINO IR** (`.xml/.bin`): Whisper, Qwen2.5-3B, Qwen2-VL, CNN.
- **ONNX** read natively by `Core.read_model()`: PP-OCR, WeSpeaker (could pre-convert to IR
  with `ovc` for faster load; not required).

So: **all 7 model components now run on OpenVINO by default** — including hand tracking
(`OpenVINOHandTracker`, validated live on the VDI). MediaPipe remains only as an automatic
fallback. The honest phrasing is now "the whole app runs inference on OpenVINO (Intel
CPU/GPU/NPU)"; `mediapipe` is still a dependency purely as the fallback path.

---

## 4. Architecture / module map (`src/airsketch/`)

Core: `config.py` (AppConfig; now has `hand_tracker_backend` + `hand_device`), `camera.py`,
`video_source.py`, `hand_tracker.py` (MediaPipe + `HandTracker` ABC + `create_hand_tracker`
factory), `hand_tracker_ov.py` (`OpenVINOHandTracker` — BlazePalm + landmark on OpenVINO,
pure-numpy pre/post), `gesture_detector.py`, `primitive_classifier.py`, `beautifier.py`,
`stroke.py`, `notebook.py`, `shape_recognizer.py`, `exporter.py` (+ `export_lesson_report`),
`utils.py` (drawing; `draw_panel` is rounded/bounds-safe + `fill/stroke_rounded_rect`).

Inference/OpenVINO: `inference_engine.py`, `sketch_classifier.py` (CNN), `sketch_cnn.py`
(training), `voice.py` (Whisper STT; exposes `last_audio`), `diagram_analyzer.py` (analyzer
chain + VLM + model helpers `pick_device`/`download_vlm_model`/`ensure_ov_tokenizer`).

Board / understanding / speaker:
- `ocr_reader.py` — `PPOCROpenVINOReader` (PP-OCRv5 det+rec on OpenVINO; DB postproc + CTC,
  numpy; no paddle dep) + `assemble_text`.
- `board_capture.py` — `BoardCapturer`: OCR (always) + optional LLM enrichment → `BoardNote`.
- `lesson_llm.py` — `LessonUnderstander` (Qwen2.5-3B on OpenVINO-GenAI) + `parse_understanding`.
- `speaker_id.py` — fbank (numpy Kaldi-style + CMN) + `SpeakerEmbedder` (WeSpeaker) +
  `SpeakerProfile` + `build_profile`.

Classroom (`src/airsketch/classroom/`):
- `challenge_engine.py` (curriculum, `retry_last`), `judge.py`, `celebration.py`,
- `app.py` — game state machine + main loop + **all rendering** (UI restyled; single
  `_draw_center_overlay` modal, single bottom `_draw_key_legend`),
- `voice_commands.py` (`parse_command`; intents incl. RETRY, CAPTURE_BOARD),
- `voice_controller.py` (`take_result()` returns text+audio).

Entry: `python -m airsketch.main --classroom [...]` (also `python -m airsketch.classroom`,
`python -m airsketch.board_capture <image> [--understand]` for no-camera OCR/LLM test).

---

## 5. Run it (all features)

```powershell
# from the AirSketch dir, after: py -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -e .            # deps
python -m pip install sounddevice     # voice/dictation only

# everything on (board OCR + AI summary + voice + teacher dictation):
python -m airsketch.main --classroom --board --understand --voice --teacher-voice --no-rotate --mirror
# shorter (implications): --understand implies --board; --teacher-voice implies --voice
python -m airsketch.main --classroom --understand --teacher-voice --no-rotate --mirror
```
Keys (also in the in-app bottom legend): **SPACE** play · **ENTER** submit · **C** clear ·
**R** retry · **T** theme · **B** read board · **V** talk · **D** dictate (1st press learns) ·
**L** learn/switch a new voice · **Q** quit. On exit → `outputs\lesson_<ts>.html`.
On a laptop use `--no-rotate --mirror` (VDI camera is upside-down → default rotation).

---

## 6. Models on disk (`models/`)

Runtime-essential (bundled in the dist zip): `sketch_classifier.xml/.bin` (+ class_names),
`hand_landmarker.task` (MediaPipe), `whisper-base-ov/`, `ppocr/` (PP-OCRv5 ONNX + dict),
`speaker/` (WeSpeaker ONNX).

Big LLM/VLM (also bundled, as **plain symlink-resolved dirs** created this session):
- `models/qwen2.5-3b-instruct-ov` (~1.8 GB) — board understanding LLM.
- `models/qwen2-vl-2b-ov` (~1.7 GB) — notebook VLM.
- **Loaders now prefer these local dirs** (offline) before any HF download:
  `lesson_llm._ensure_pipe` checks `qwen2.5-3b-instruct-ov`; `diagram_analyzer._try_vlm_analyzer`
  checks `qwen2-vl-2b-ov`. The raw HF caches (`models/models--*`) are the fallback / dev source.
- `models/teacher_voice.json` = a personal enrolled-voice profile (NOT shipped; delete to re-learn).

`data/` (~1.3 GB Quick-Draw training set) and `.venv` are NOT needed at runtime (CNN is
already trained); excluded from the dist.

---

## 7. What we changed THIS session (changelog)

1. **Voice bug (real fix):** intermittent `!!!!`/garbage even while speaking was an
   **uninitialized `sd.rec` buffer** (NaN/Inf in unfilled frames poisoning Whisper). Fixed:
   zero-filled buffer + `out=` + NaN/Inf scrub in `voice.py`. (Diagnosed via an amplitude
   print that showed peak=NaN.) Don't revert the zero-fill.
2. **Retry:** miss a challenge → **R** / say "again" re-attempts the same target; the miss
   is dropped (doesn't count). `ChallengeEngine.retry_last`.
3. **Phase 3 board capture = OCR, NOT VLM.** We tested Qwen2-VL-2B and it **cannot read text**
   (only coarse colour blobs). Switched to **PP-OCR on OpenVINO** (`ocr_reader.py`) — verified
   near-perfect. Don't re-attempt VLM-based transcription with the 2B model.
4. **Phase B understanding LLM:** Qwen2.5-3B INT4 (`lesson_llm.py`) summarizes the OCR text →
   topic/summary/key points; verified offline.
5. Removed dead `llm_chat.py` (old unused Qwen helper).
6. **Teacher voice:** WeSpeaker speaker ID; **D** = learn-then-dictate (gated to enrolled
   voice), **L** = learn/switch a new voice (overwrites profile). Needs real-mic threshold tuning.
7. **Lesson report (Phase 4):** `export_lesson_report` → self-contained HTML (scoreboard +
   board notes + teacher narration). Written **incrementally** + window-close (X) handling +
   fast shutdown that releases the camera on a side thread then `os._exit` (the camera LED
   wasn't turning off otherwise). Bug fixed: dictation-only sessions used to write no report.
8. **UI restyle (cosmetic only):** `draw_panel` rounded/bounds-safe; cohesive `UI_*` palette;
   **consolidated all key hints into ONE bottom legend** (`_draw_key_legend`) and **all center
   banners into ONE priority-picked modal** (`_draw_center_overlay`) — fixed "window over a
   window" on D. Route future banners through `_draw_center_overlay`, keys through the legend.
9. **Distributable:** `..\AirSketch_dist.zip` (~3.85 GB) — fully self-contained, no downloads/
   training needed; includes all models (incl. teacher voice). Excludes `.venv`, `data/`, raw HF
   caches, `.pyc`, personal voice profile.
10. **Hand-tracking → OpenVINO plan:** `OPENVINO_HAND_TRACKING_PLAN.md` (verified the two
    MediaPipe TFLite sub-models load on OpenVINO; documented I/O, BlazePalm decode, ROI, phases).
12. **Hand tracking ported to OpenVINO (NEW):** `hand_tracker_ov.py` `OpenVINOHandTracker` —
    BlazePalm palm detector + 21-pt landmark model on the OpenVINO runtime, pure-numpy host
    code (2016 SSD anchors, decode, weighted NMS, rotated-ROI crop + inverse map, detect↔track
    state machine, EMA smoothing). Drop-in for MediaPipe (same `process()->HandResult`, fills
    `world_landmarks` from `Identity_3`). Factory `create_hand_tracker(cfg)`; `--hand-backend
    {openvino,mediapipe}` (**default openvino**) + `--hand-device {AUTO,CPU,GPU,NPU}` +
    `--hand-debug` in `main.py` and `classroom/app.py`. **NOW THE DEFAULT — validated live on
    the VDI** (air-drawing scored real strokes; `[hand] OpenVINO hand tracker on CPU`); ~108 FPS.
    MediaPipe kept only as automatic fallback. Two real bugs found & fixed during bring-up:
    (1) **input normalization is [0,1] (x/255), NOT [-1,1]** — wrong norm makes the palm
    detector emit phantom hands (saturates to 1.0 on blank input);
    (2) **the landmark model's presence/handedness outputs are already probabilities — do NOT
    sigmoid them.** The double-sigmoid pinned presence ≥ 0.5 forever, so the tracker could
    never decide it lost the hand and got stuck on a bad ROI (this was the "hand not recognized
    at all" live symptom). 26 new tests incl. a blank-frame-not-visible regression guard.
11. **Mid-project deck:** `AirSketch_MidProject_v2.pptx` (7 slides, uniform light theme,
    low-text, Hebrew speaker notes, includes a Target-Intel-hardware + model→component slide).
    Built with `python-pptx` (npm/pptxgenjs was blocked by the corporate proxy's EINTEGRITY).

---

## 8. Gotchas / don't-re-chase

- VDI camera is upside-down → `rotate_180` default True; use `--no-rotate` on a normal PC.
- `live_snap` default off (freehand kept as-drawn); judge classifies raw strokes.
- `pip install -e .` does NOT pull `sounddevice` (voice extra) — install separately.
- Windows console is cp1252 → use `PYTHONIOENCODING=utf-8` / `sys.stdout.reconfigure` or ASCII.
- `os._exit(0)` is used to kill MediaPipe/OpenVINO non-daemon threads on exit; classroom
  `run()` finally releases the camera on a side thread (≤4 s) THEN `os._exit` (LED off + fast).
- `cam.read()` can block after a mic recording (camera+mic conflict); voice is push-to-talk
  with the camera paused. Incremental report write is the safety net.
- VDI mic may deliver no audio (redirection) — voice is best on a real PC. After the buffer fix,
  true silence yields a clean empty result, not garbage.
- NPU dislikes dynamic shapes / large LLMs — expect CPU/GPU fallback there.
- Don't disable TLS/integrity to fix installs (org policy). npm is proxy-broken here; prefer
  python-pptx / pip which worked.

---

## 9. Tests

`PYTHONPATH=src python -m pytest tests/ -q` → **150 passed**. Files include
`test_primitive_classifier`, `test_gesture_detector`, `test_notebook`, `test_sketch_classifier`,
`test_classroom` (+retry), `test_voice_commands` (+retry/board), `test_voice_controller`,
`test_board_capture`, `test_ocr_reader`, `test_lesson_llm`, `test_lesson_report`, `test_speaker_id`,
`test_hand_tracker_ov` (OV hand tracker), `test_launcher` (GUI flag-builder).
UI/rendering is not unit-tested (no LibreOffice here to render slides/HUD for pixel QA).

## Hebrew language support (NEW)

Toggle via `--lang {en,he}` (both entry points) + a **Language** dropdown in the GUI
launcher; English stays default (`cfg.language`). Hebrew is implemented **only where it
stays on OpenVINO** (the user's constraint):
- **Whisper STT** (`voice.py`): multilingual `whisper-base-ov` has `<|he|>`; we stopped
  forcing `<|en|>` and pick the token + a Hebrew bias prompt. Covers voice triggers + dictation.
- **Voice command parsing** (`voice_commands.py`): Hebrew shape/verb keywords + Hebrew
  morphology (strips the attached article so "המשולש" matches "משולש"). English unchanged.
- **Board LLM** (`lesson_llm.py`): Qwen2.5-3B gets a Hebrew system+user prompt when `he`.
- **On-screen Hebrew text**: cv2 can't render Hebrew/RTL, so `hebrew_text.py` rasterizes via
  Pillow with an in-house RTL reorderer (Hebrew is non-cursive → reorder only, **no bidi
  dependency**) + a Windows Hebrew font (Arial/David), alpha-blitted onto the frame.
  `utils.draw_text_with_shadow` + `utils.text_size` are now Hebrew-aware (auto-upgrade any
  non-ASCII string; ASCII still uses the fast cv2 path). Classroom prompts/banners/legend/HUD
  are localized via `HE_TARGETS`/`HE_THEMES`/`HE_UI` + `self._t`/`_prompt_text`.
- **Speaker-gating** is language-agnostic — no change.
- **Board OCR: deferred** — no OpenVINO-runnable Hebrew OCR model exists (PaddleOCR has no
  Hebrew; HebHTR/hebOCR/etc. aren't ONNX/OV). Stays English. Future R&D = train a PP-OCR
  Hebrew rec model → ONNX → OV.
- Caveat: whisper-base Hebrew accuracy is weaker than English (small model); needs live mic test.
- Tests: `tests/test_hebrew.py` (commands, morphology, Whisper token, LLM prompt, RTL renderer,
  CLI/GUI flag). Verified the renderer visually (Hebrew prompt/banner/legend rendered to PNG).

## Voice & board-capture improvements (NEW)

- **Board capture mirror fix:** when NOT in `--mirror` mode, the captured frame is
  horizontally flipped before OCR (and in the saved report image) so whiteboard text
  reads correctly instead of mirror-writing (`_capture_board` → `cv2.flip(frame, 1)` when
  `not cfg.mirror`).
- **Dictation vs command Whisper configs:** dictation now uses a separate config —
  **no command-vocabulary bias** (so free speech isn't nudged toward shape words) and a
  longer token cap (220 vs 80). Commands keep the biased/short config. Plumbed via
  `VoiceRecorder.start(purpose=...)` ← `VoiceController.toggle(purpose=...)` ← classroom.
- **Mic feedback + timeout:** voice-unavailable is now shown on-screen ("voice off — no
  microphone"; Hebrew too) instead of silently hiding V/D/L. Mic-open probe timeout raised
  3 s → **6 s**, configurable via `--mic-timeout` + a GUI spinbox (VDI mics can be slow/absent).
- **Selectable Whisper model (all OpenVINO):** `--whisper-model {base,small}` + GUI dropdown
  (`cfg.whisper_model`; `voice.whisper_model_dir()`). `base` = `models/whisper-base-ov`
  (default, fast); `small` = `models/whisper-small-ov` (~0.5 GB, more accurate esp. Hebrew).
  Both are OpenVINO IR run via `ov_genai.WhisperPipeline` — verified small loads+infers on OV.
  whisper-small is downloaded into `models/` (ships alongside the exe; not bundled).

## Distributable + deliverable docs (NEW)

- **Distributable zip:** `D:\AirSketch_dist.zip` (~5.04 GB, 6092 files, ZIP_STORED,
  extracts into `AirSketch/`). Self-contained: full source + all runtime models + the
  one-click onedir exe (`dist/AirSketch/AirSketch.exe`) + `RUN_ME.txt`. Excludes
  `.venv`/`data`/`build`/`__pycache__`/`*.log`, the redundant raw HF caches
  (`models/models--*`), and the personal `teacher_voice.json`. Rebuild with
  `python build_dist.py` (writes to the parent dir; move to D: if C: is tight — only
  ~25 GB free on C:, D: has ~161 GB). Verified via `zipfile.testzip()`.
- **Deliverable-authoring docs (`docs/`):** `PROJECT_COMPENDIUM.md` (master content
  source), `PRESENTATION_GUIDE.md`, `BOOK_GUIDE.md`, `DELIVERABLES_README.md` — so a
  future session can build a **detailed academic presentation** and an **academic book**
  given just the repo + these files. Start at `docs/DELIVERABLES_README.md`. The prior
  deck (`AirSketch_MidProject_v2.pptx`) is the baseline the new deck must exceed.

## GUI launcher / exe (NEW)

`launcher.py` (project root) — a Tkinter feature-flags GUI that builds the
`airsketch.main` CLI and launches the app. Same file is launcher + app runner via a
`--launch-app` dispatch. `build_exe.bat` (needs `pip install -e .[exe]` → PyInstaller)
produces `dist/AirSketch/AirSketch.exe` — a **`--onedir`** folder build (~673 MB folder;
keep it together). **Onedir on purpose:** a `--onefile` build unpacks ~250 MB to
`%TEMP%\_MEIxxxxx` each launch, and endpoint AV on the VDI corrupted that mid-extract →
`pyi_rth_pkgres` / `base_library.zip not found` crash on the user's machine. Onedir has
no temp extraction (fixed it) and starts faster. Models NOT bundled — the app walks up
from the exe to find `models/` via `resolve_root()`. Icon: `airsketch.ico` (`make_icon.py`).
Verified frozen: `AirSketch.exe --check` imports all deps OK; `--launch-app --help`
reaches the real argparse. See `LAUNCHER.md`. Flag mapping is unit-tested
(`tests/test_launcher.py`). GUI closes on launch (single-shot).

---

## 10. Open items / next steps (priority)

1. **(DONE) OpenVINO hand tracking is live-validated and now the default.** Optional
   follow-ups: drop the `mediapipe` dep (it's only the fallback now) and/or pre-convert the
   two `.tflite` to IR with `ovc` for faster load / NPU. Tuning knobs are module-level
   constants in `hand_tracker_ov.py` (`HAND_ROI_SCALE`, `HAND_ROI_SHIFT_Y`, `PRESENCE_THRESH`,
   `LM_SMOOTH_ALPHA`); use `--hand-debug` to see palm-score/presence/state live.
2. **Tune teacher-voice** threshold + enrollment on a real mic.
3. **Validate NPU/GPU** for the hand models (both are fixed-shape → good NPU fits, no reshape
   needed); CPU was the only device in the dev env. Graceful fallback already wired.
4. Polish judging leniency + a clean classroom demo run-through.
5. (Stretch) wire `image_gen.py` (LCM).

---

## 11. Deliverables produced this session (locations)

- Dist: `C:\ssh-web-server-python\.claude\worktrees\AirSketch_dist.zip` (~3.85 GB, self-contained).
- Deck: `…\AirSketch\AirSketch_MidProject_v2.pptx` (updated — uniform, Hebrew notes, HW slide).
  `…\AirSketch_MidProject.pptx` is the earlier version (was locked/open when v2 was made).
- Plan: `…\AirSketch\OPENVINO_HAND_TRACKING_PLAN.md`.
- Test fixture: `…\AirSketch\sample_board.png` (for `python -m airsketch.board_capture`).

---

## 12. "To be verified" placeholders

- Presentation: course name + team members (deck slide 1).
- Target Intel device: exact Core Ultra / AI-PC SKU for the model→component plan (deck slide 6).
- Teacher-voice acceptance threshold on real hardware.
