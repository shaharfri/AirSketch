# AirSketch — Project Compendium (master content source)

> **Purpose.** This is the single, authoritative, content-complete reference for
> AirSketch. It exists so that a future Claude session — given the repository plus
> this file — can author **(a) a detailed academic presentation** and **(b) an
> academic book** about the project without having to re-derive anything.
>
> Read this together with `PRESENTATION_GUIDE.md` and `BOOK_GUIDE.md` (which say
> *how* to turn this content into each deliverable) and `DELIVERABLES_README.md`
> (the index). Ground-truth engineering detail also lives in `SESSION_HANDOFF.md`,
> `HANDOFF.md`, `OPENVINO_HAND_TRACKING_PLAN.md`, and `LAUNCHER.md`.
>
> Style note for downstream authors: prefer claims this document marks as
> **verified**; carry forward items marked **to-be-verified** as such. Do not
> overclaim (e.g. avoid "all models run on OpenVINO" without the nuance below).

---

## 0. One-paragraph abstract

AirSketch is a real-time, camera-based **air-drawing classroom application**: a
webcam tracks the user's index finger, the user "draws" shapes and objects in the
air, and a teacher/student **challenge game** scores the drawing (0–100 + 0–3
stars) with celebratory feedback. Around this core sit **voice commands**,
**speaker-gated teacher dictation**, **whiteboard capture** (photograph → OCR →
LLM summary), and an auto-generated **HTML lesson report**. The defining
engineering goal is that **all AI inference runs on the Intel OpenVINO runtime**
(CPU / GPU / NPU), making the system a showcase of on-device, accelerator-portable
AI. The project's headline technical contribution is a from-scratch re-implementation
of Google MediaPipe's hand-tracking pipeline (BlazePalm detector + hand-landmark
model) on OpenVINO, removing the last non-OpenVINO dependency. The app also supports
a **Hebrew/English** language toggle across every OpenVINO-native path, ships a
**GUI launcher** and a single-folder **Windows executable**, and is covered by
**175 automated tests**.

---

## 1. Motivation and goals

- **Pedagogical hook.** Drawing in the air is engaging for classrooms; gamifying it
  ("draw a triangle!", scored instantly) turns recognition models into a learning loop.
- **Technical north star.** Run **every** ML model on **OpenVINO**, device-selectable
  across Intel **CPU / GPU / NPU**. This makes the app a concrete demonstration of
  portable, on-device inference (no cloud, offline-capable).
- **Breadth as a feature.** The app deliberately spans many AI modalities — vision
  (hand tracking, sketch CNN, OCR, VLM), speech (STT, speaker ID), and language
  (LLM summarization) — all unified under one runtime.
- **Honest engineering.** A recurring theme (and a good narrative for the book/talk):
  verifying claims by *running* models, distinguishing *works* / *partial* /
  *planned*, and not overclaiming.

---

## 2. Lineage — a merge of two prototypes

- **AirDraw AR / AirNotes** (first prototype): multi-stroke "notebook", live
  snap-to-shape, joint-angle gesture detection, HTML export, early Qwen2-VL analysis.
- **Skysketch** (second prototype): a trained Quick-Draw **CNN classifier**, a clean
  OpenVINO `InferenceEngine`, a **pytest** suite, a training pipeline, optional
  Whisper voice, and (unused) LCM image generation.
- **AirSketch** = the merge: AirDraw's notebook/UX + Skysketch's CNN/tests/training/voice,
  then extended with the classroom game, board capture, teacher voice, OpenVINO hand
  tracking, Hebrew support, and packaging.

---

## 3. System architecture

Python package `airsketch` under `src/airsketch/`. Two entry points:
`python -m airsketch.main` (notebook) and `--classroom` (the game); plus the GUI
launcher (`launcher.py`).

**Runtime data flow (classroom):**
```
Camera ─▶ frame ─▶ HandTracker.process() ─▶ HandResult (21 landmarks, world landmarks)
                          │
                          ▼
              IndexPointingDetector (pen up/down)  ── pen down ▶ Notebook.append_to_stroke
                          │                                              │
                          ▼                                              ▼
                    Game state machine  ── submit ▶  Judge (primitive classifier / CNN)
                          │                                              │
                          ▼                                              ▼
                    Celebration + scoreboard                      ChallengeResult
   Voice (V): Whisper STT ─▶ parse_command ─▶ Intent ─▶ game actions
   Dictation (D): Whisper STT + WeSpeaker gate ─▶ lesson narration
   Board (B): photo ─▶ PP-OCR ─▶ (Qwen2.5-3B summary) ─▶ BoardNote
   On exit: export_lesson_report() ─▶ outputs/lesson_<ts>.html
```

**Module map (selected):**
- Core: `config.py` (`AppConfig` + enums; most defaults), `camera.py`,
  `video_source.py`, `hand_tracker.py` (interface + MediaPipe + factory),
  `hand_tracker_ov.py` (OpenVINO hand tracker), `gesture_detector.py`,
  `primitive_classifier.py`, `beautifier.py`, `stroke.py`, `notebook.py`,
  `shape_recognizer.py`, `exporter.py`, `utils.py` (drawing; Hebrew-aware text),
  `hebrew_text.py` (RTL renderer).
- Inference/OpenVINO: `inference_engine.py`, `sketch_classifier.py` (CNN),
  `sketch_cnn.py` (training-only model def), `voice.py` (Whisper),
  `diagram_analyzer.py` (analyzer chain + VLM + `pick_device`/`download_vlm_model`/
  `ensure_ov_tokenizer`), `ocr_reader.py` (PP-OCR), `board_capture.py`,
  `lesson_llm.py` (Qwen2.5-3B), `speaker_id.py` (WeSpeaker).
- Classroom (`src/airsketch/classroom/`): `app.py` (state machine + main loop +
  all rendering), `challenge_engine.py`, `judge.py`, `celebration.py`,
  `voice_commands.py`, `voice_controller.py`.

---

## 4. The AI stack — every model on OpenVINO

| Component | Model | Framework / format | OpenVINO runtime call | Device flag | Verified |
|---|---|---|---|---|---|
| Hand tracking (default) | BlazePalm + hand-landmark (MediaPipe `.tflite`) | TFLite read by OV TFLite frontend | `core.compile_model(core.read_model(.tflite))` | `--hand-device` | yes (CPU; live) |
| Sketch CNN | Quick-Draw CNN (custom) | OpenVINO IR (`.xml/.bin`) | `ov.Core().compile_model` | `--cnn-device` | yes (96% test acc) |
| Voice STT | Whisper base / small | OpenVINO IR (GenAI) | `ov_genai.WhisperPipeline` | `--voice` (device CPU) | yes |
| Board OCR | PP-OCRv5 detection + English recognition | ONNX read by OV | `core.compile_model(core.read_model(onnx))` | `--ocr-device` | yes |
| Board LLM | Qwen2.5-3B-Instruct INT4 | OpenVINO IR (GenAI) | `ov_genai.LLMPipeline` | `--llm-device` | yes |
| Speaker ID | WeSpeaker ResNet34 | ONNX read by OV | `core.compile_model(core.read_model(onnx))` | `--speaker-device` | yes |
| Notebook VLM | Qwen2-VL-2B-Instruct INT4 | OpenVINO IR (GenAI) | `ov_genai.VLMPipeline` | `--vlm-device` | yes (enrich-only; cannot read text) |

**Key framing:** *all seven model components run on the OpenVINO runtime.* Hand
tracking was the last hold-out (MediaPipe/TFLite-CPU) and is now ported; MediaPipe
remains only as an automatic fallback. Device selection is a live argument
(`AUTO|CPU|GPU|NPU`) via `pick_device()` with graceful fallback. NPU favors
fixed-shape models (CNN, speaker, the hand models); dynamic-shape models (OCR
detection, Whisper, LLM/VLM) may fall back to CPU/GPU on NPU.

---

## 5. Headline contribution — hand tracking on OpenVINO

**Problem.** MediaPipe Hands runs on TFLite/XNNPACK (CPU only) and is a third-party
black box — the one model not on OpenVINO. Goal: reproduce it on OpenVINO so the
*whole* app is accelerator-portable, with no MediaPipe dependency at runtime.

**Key enabler.** `models/hand_landmarker.task` is a zip bundling two TFLite
sub-models; **OpenVINO's `Core.read_model()` reads `.tflite` directly** (TFLite
frontend) — no external conversion needed (optional `ovc` → IR for NPU/static shapes).

**Verified model I/O:**
- `hand_detector.tflite` (BlazePalm): in `input_1 [1,192,192,3]` (NHWC RGB) →
  `Identity [1,2016,18]` (per-anchor 4 box + 7 keypoints×2 regressors) +
  `Identity_1 [1,2016,1]` (palm score logit).
- `hand_landmarks_detector.tflite`: in `input_1 [1,224,224,3]` →
  `Identity [1,63]` (21 landmarks ×xyz, crop space), `Identity_1 [1,1]` (presence,
  **already a probability**), `Identity_2 [1,1]` (handedness, probability),
  `Identity_3 [1,63]` (21×3 **world landmarks**, metric, wrist-relative).

**Pure-numpy/cv2 host pipeline (`hand_tracker_ov.py`, class `OpenVINOHandTracker`):**
1. **Letterbox** the frame to a square, resize to 192, normalize to **[0,1]** (x/255).
2. **BlazePalm SSD anchors** — `generate_anchors()`: `strides=[8,16,16,16]`,
   `anchor_offset=0.5`, merge same-stride layers, `fixed_anchor_size=True` (all
   anchors w=h=1.0). Yields exactly **2016 anchors** (24²·2 + 12²·6 = 1152 + 864).
3. **Decode** (`decode_boxes`): box center/size + 7 keypoints from per-anchor
   regressors relative to anchor centers, in normalized 192-space.
4. **Score** = sigmoid(`Identity_1`); threshold 0.5; **weighted NMS** (overlapping
   boxes merged by score-weighted average, not just suppressed).
5. **Rotated ROI** (`roi_from_palm`/`roi_affine`/`crop_roi`): rotation from keypoint
   0 (wrist) → keypoint 2 (middle-finger MCP); expand the palm box (~2.6×) into a
   rotated square shifted toward the fingers; `cv2.getAffineTransform` → warp to a
   224 crop. Landmarks map back to frame pixels via `cv2.invertAffineTransform`.
6. **Landmark model** on the crop → 21 px landmarks + world landmarks + presence.
7. **Detect↔track state machine**: once a hand is found, derive the next frame's ROI
   from the **previous landmarks** and **skip palm detection** while presence ≥ 0.5
   — re-detect only when presence drops. Light EMA smoothing reduces jitter.
8. Fills the same `HandResult` contract as MediaPipe (drop-in): `visible`, 21 pixel
   `landmarks`, `fingertip`=lm8, `thumb_tip`=lm4, `wrist`=lm0, `hand_size`,
   `confidence`=presence, `world_landmarks` (21×3 from `Identity_3`).

**Two real bugs found by running it (great teaching material):**
- **Input normalization is [0,1], not [-1,1].** The MediaPipe graph docs imply
  `[-1,1]`, but the converted `.tflite` expects `[0,1]`; feeding `[-1,1]` made the
  palm detector saturate to score 1.0 on uniform/blank input and emit off-frame
  "phantom" boxes. Verified on real hand stills: `[0,1]` → score 0.65–0.88 with valid
  in-frame boxes. Guarded by a blank-frame-not-visible regression test.
- **Presence/handedness are already probabilities — do not sigmoid them.** Applying
  sigmoid a second time pinned presence ≥ 0.5 for any input, so the tracker could
  never decide it had lost the hand → it got stuck on a bad ROI (the live "hand not
  recognized at all" symptom). Reading the raw probability fixed it.

**Validation & performance (verified):**
- End-to-end on real captured frames: the 21-landmark skeleton fits a real
  (horizontally-oriented) hand correctly, including rotation — confirming decode +
  rotated-ROI + inverse-mapping.
- **~108 FPS** CPU steady-state on the tracking path; ~75 FPS on the palm-detect
  path; the state machine does exactly one palm-detect then tracks.
- Default backend flipped to OpenVINO after live validation on the Check Point VDI
  (air-drawing scored real strokes). MediaPipe kept as automatic fallback.
- **To-be-verified:** GPU/NPU on real Intel hardware (dev env had CPU only; graceful
  fallback wired; both hand models are fixed-shape → good NPU candidates).

**Reference algorithm:** geaxgx `depthai_hand_tracker` (`mediapipe_utils.py`) and the
MediaPipe Hands model card.

---

## 6. The other components (concise but complete)

- **Sketch CNN** (`sketch_classifier.py`, OpenVINO IR): 12 Quick-Draw classes
  (triangle, square, circle, house, car, tree, star, cat, flower, sun, airplane,
  fish), **96% test accuracy**. Expects bright strokes on black background
  (Quick-Draw format); judged via a black-bg render of the user's strokes.
- **Gesture detection** (`gesture_detector.py`): `IndexPointingDetector` fuses two
  signals per finger — **joint angle** and **compactness ratio** — with hysteresis
  and a confirmation window; prefers orientation-invariant **3D world landmarks**,
  falls back to 2D. `PinchDetector` (thumb–index distance / hand size) is the
  alternative. This is why the OpenVINO tracker's `world_landmarks` matter.
- **Shape recognition / judging** (`primitive_classifier.py`, `judge.py`): geometry
  targets judged by the primitive classifier; object targets by the CNN; synonyms
  (square≈rectangle, circle≈ellipse); 0–100 score + 0–3 stars.
- **Challenge game** (`challenge_engine.py`, `app.py`): state machine
  READY → ANNOUNCE → DRAWING → RESULT; curriculum themes geometry/objects/mixed;
  **retry** after a miss (R or "again"); celebration (confetti/stars or shake/hint).
- **Voice** (`voice.py`, Whisper via `ov_genai.WhisperPipeline`): push-to-talk
  (camera paused during recording to avoid a camera+mic stall); a zero-filled
  `sd.rec` buffer + NaN/Inf scrub (see bug journey); separate command vs dictation
  configs (§7). Multilingual base model; selectable `base`/`small`.
- **Board capture** (`ocr_reader.py`, `board_capture.py`): PP-OCRv5 **detection**
  (DB post-processing) + English **recognition** (CTC), both ONNX on OpenVINO,
  hand-written numpy post-processing (no paddlepaddle). Optional Qwen2.5-3B summary.
- **Board understanding LLM** (`lesson_llm.py`): Qwen2.5-3B-Instruct INT4 on
  OpenVINO-GenAI; turns raw OCR into topic/summary/key-points/corrected JSON.
- **Speaker ID** (`speaker_id.py`): numpy Kaldi-style fbank + WeSpeaker ResNet34
  (ONNX on OV) → 256-d embedding; cosine threshold gates dictation to the enrolled
  teacher. **D** learns then dictates; **L** re-learns / switches voice.
- **Lesson report** (`exporter.export_lesson_report`): self-contained HTML
  (scoreboard + board notes with embedded images + teacher narration), written
  **incrementally** so a forced exit never loses it.
- **Notebook VLM** (`diagram_analyzer.py`): Qwen2-VL-2B on OV-GenAI; **enrich-only**
  — it cannot read text (proven), so board transcription uses PP-OCR, not the VLM.

---

## 7. Multilingual (Hebrew) support

Toggle `--lang {en,he}` (CLI + GUI dropdown; English default; `cfg.language`).
Implemented Hebrew **only where it stays on OpenVINO**:
- **Whisper STT**: `whisper-base-ov` is the *multilingual* base (99 languages,
  includes `<|he|>`); the previously forced `<|en|>` is now language-selected, with a
  Hebrew bias prompt. Covers voice triggers and dictation.
- **Voice-command parsing** (`voice_commands.py`): Hebrew shape/verb keywords
  (צייר משולש, הבא, שלח, נקה, שוב, צורות/חפצים/מעורב, קרא את הלוח…), plus **Hebrew
  morphology** — strips the attached definite article so "המשולש" matches "משולש".
  English parsing unchanged (regression-tested).
- **Board LLM**: Qwen2.5-3B receives a Hebrew system+user prompt and answers in Hebrew.
- **On-screen RTL text** (`hebrew_text.py`): OpenCV's Hershey fonts cannot render
  Hebrew or right-to-left, so Hebrew strings are rasterized via **Pillow** + a Windows
  Hebrew font and alpha-blitted onto the BGR frame. Hebrew is **non-cursive** (no
  glyph shaping), so an **in-house RTL reorderer** suffices — **no bidi dependency**.
  `utils.draw_text_with_shadow` / `utils.text_size` are Hebrew-aware; ASCII keeps the
  fast cv2 path. Classroom prompts/banners/legend/HUD are localized.
- **Speaker-gating** is language-agnostic (embeddings) — unchanged.
- **Board OCR Hebrew is deferred**: there is **no OpenVINO-runnable Hebrew OCR model**
  off-the-shelf (PaddleOCR has no Hebrew; HebHTR/hebOCR/Jochre are TF/C/not ONNX;
  Tesseract/EasyOCR are not OpenVINO). Future R&D = train a PP-OCR Hebrew recognition
  model → export ONNX → run on OV.
- **Caveat:** whisper-base Hebrew accuracy is weaker than English (small model) →
  hence the selectable `whisper-small` (more accurate, ~0.5 GB, still OpenVINO).

**Dictation vs command Whisper configs** (a quality fix): dictation uses a *neutral*
config (no command-vocabulary bias, longer token cap) so free speech isn't nudged
toward shape words or truncated; commands keep the biased/short config.

---

## 8. Packaging, UX, and deployment

- **GUI launcher** (`launcher.py`, Tkinter — no heavy dep, offline): a "feature flags"
  screen exposing every CLI option (mode, theme, **language**, camera/mirror/rotate,
  hand backend + device, CNN device, voice/teacher-voice, board/understand, VLM, snap,
  hand-debug, **mic timeout**, **Whisper model**) with a live command preview. One file
  serves as **both** launcher **and** app runner via a `--launch-app` dispatch.
- **Windows executable** (PyInstaller **`--onedir`** → `dist/AirSketch/AirSketch.exe`,
  custom icon `airsketch.ico` from `make_icon.py`). Onedir (not onefile) on purpose:
  onefile unpacks ~250 MB to `%TEMP%` per launch and the VDI's endpoint antivirus
  corrupted that extraction (`pyi_rth_pkgres` / `base_library.zip not found` crash);
  onedir has no temp extraction and starts faster. Models are **not** bundled (GBs) —
  the app finds `models/` by walking up from the exe (`resolve_root`).
- **Build pipeline** (`build_exe.bat`): `--collect-all` for openvino / openvino_genai /
  openvino_tokenizers / cv2 / mediapipe, `--collect-submodules airsketch`, excludes
  torch/torchvision/matplotlib/tensorflow.
- **No-console crash fix:** a windowed exe has `sys.stdout/err = None`; the classroom
  quit path called `.flush()` on them → crash on Q. Fixed by routing app output to a
  log file when those are None (`airsketch_run.log`) and making the flushes None-safe.

---

## 9. Engineering lessons / bug journeys (narrative gold for both deliverables)

1. **Verify by running, not by labels.** Every "it's on OpenVINO" claim was proven by
   loading/running the model; the hand-track normalization and presence bugs were only
   caught by feeding real images and reading actual scores.
2. **Hand-track normalization [-1,1] vs [0,1]** — wrong norm → phantom detections.
3. **Double-sigmoid on an already-probability output** → tracker stuck on a bad ROI.
4. **Whisper hallucinated garbage** on an uninitialized `sd.rec` buffer (NaN/Inf) →
   zero-filled buffer + scrub. Also: a missing `import math` (hidden by `os._exit`)
   masqueraded as a "voice hang."
5. **Qwen2-VL-2B cannot read text** → board capture uses PP-OCR, not the VLM.
6. **PyInstaller onefile + endpoint AV** → temp-extraction corruption → switch to onedir.
7. **Windowed exe `sys.stdout is None`** → None-safe flush + log redirection.
8. **A training-only module importing torch at top-level** crashed the build's isolated
   submodule scan (paging-file exhaustion) → lazy/guarded torch import.
9. **Zombie `python.exe` processes** accumulated and exhausted the Windows paging file,
   intermittently failing builds → kill leftovers before building.
10. **Camera + mic simultaneously stalls the camera** → push-to-talk pauses the camera.
11. **VDI specifics:** upside-down camera (`rotate_180` default), redirected mic may not
    open (configurable probe timeout + on-screen "voice off" feedback), cp1252 console.
12. **Board capture mirror:** in non-mirror mode the photo is horizontally flipped before
    OCR so whiteboard text isn't read as mirror-writing.

---

## 10. Testing & quality

- **175 pytest tests** (`PYTHONPATH=src python -m pytest tests/ -q`). Coverage spans
  primitive classifier, gesture detector, notebook, sketch classifier, classroom
  (+retry), voice commands (+retry/board/**Hebrew**), voice controller, board capture,
  OCR reader, lesson LLM, lesson report, speaker ID, **hand tracker OV**
  (anchors/decode/NMS/ROI math + real-model integration incl. a blank-frame-not-visible
  regression guard + device-fallback), **launcher** (flag builder + windowed-stdout
  guards), and **Hebrew** (commands/morphology, Whisper token, LLM prompt, RTL renderer).
- UI rendering is not unit-tested (no headless renderer for pixel QA), but the Hebrew
  renderer and the app icon were each verified by rendering to a PNG and visually
  inspecting it.

---

## 11. Results & evaluation (what to cite)

- CNN object recognition: **96%** test accuracy, 12 classes.
- OpenVINO hand tracking: **~108 FPS** CPU; parity with MediaPipe confirmed live for
  index-pointing pen up/down and the air-drawing flow.
- Whisper: functional STT EN/HE; accuracy improves with `small` over `base`
  (qualitative; quantify in future work).
- Board OCR: near-perfect English transcription on a clean test board (qualitative).
- All inference on OpenVINO CPU; device-portability designed in (GPU/NPU to be
  benchmarked on real Intel silicon — **future work**).

---

## 12. Limitations & future work

- **GPU/NPU benchmarking** on real Intel hardware (Core Ultra / AI PC); static-IR
  reshape for NPU on the fixed-shape models.
- **Hebrew board OCR** — train/convert an OpenVINO-runnable Hebrew recognition model.
- **Bigger/*quantified* STT** — evaluate `small`/`medium`; WER on EN/HE.
- **Teacher-voice threshold** tuning on real mics; multi-speaker.
- **Wire LCM image generation** (`image_gen.py`, present but unused).
- **Quantitative user study** of the classroom game (engagement, recognition accuracy).

---

## 13. Reproducibility

- Run from source: `pip install -e .` (+ `pip install sounddevice` for voice);
  `python -m airsketch.main --classroom` (OpenVINO hand tracking is default).
  Useful flags: `--lang he`, `--whisper-model small`, `--understand`, `--teacher-voice`,
  `--no-rotate --mirror` (normal PC), `--hand-device GPU`, `--mic-timeout 10`.
- GUI: `python launcher.py` (or the built `dist/AirSketch/AirSketch.exe`).
- Build exe: `pip install -e .[exe]` then `build_exe.bat`.
- Models live in `models/` (shipped in the distributable zip; not bundled in the exe).

---

## 14. Key terms / glossary

BlazePalm (MediaPipe palm detector), SSD anchors, weighted NMS, ROI (region of
interest), hand-landmark model, world landmarks, OpenVINO IR (`.xml/.bin`), OpenVINO
GenAI (`WhisperPipeline`/`LLMPipeline`/`VLMPipeline`), TFLite frontend, INT4
quantization, CTC decode, DB detection (PP-OCR), fbank, cosine similarity, RTL/bidi,
onedir/onefile (PyInstaller), `pyi_rth_pkgres`.

---

## 15. Development timeline (milestones)

1. Prototypes: AirDraw AR, Skysketch.
2. Merge → AirSketch; classroom Phase 1 (game), Phase 2 (voice), CNN trained (96%).
3. Phase 3 board capture (PP-OCR/OpenVINO) + Phase B understanding LLM (Qwen2.5-3B).
4. Teacher voice (WeSpeaker), incremental lesson report.
5. UI restyle; mid-project deck v2 (the prior presentation).
6. **OpenVINO hand-tracking port** — designed, implemented, two bugs fixed, validated
   live, made the default. (Last non-OpenVINO path closed.)
7. **GUI launcher + onedir exe + icon**; windowed-stdout and build-robustness fixes.
8. **Hebrew/English** support across all OpenVINO-native paths + RTL rendering.
9. Board mirror fix; dictation-config decoupling; **selectable Whisper base/small**;
   mic feedback + configurable timeout.
10. Distributable zip + this deliverable-authoring documentation set.

---

## 16. "What's new since the prior presentation" (for the *more-detailed* deck)

The prior deck (`AirSketch_MidProject_v2.pptx`, ~7 slides, low-text, Hebrew speaker
notes, model→component + target-HW slides) predates these — all should appear as
**new** content:
- OpenVINO **hand tracking** implemented & default (the headline; with the BlazePalm
  re-implementation, the two bugs, and ~108 FPS) → *now the whole app is on OpenVINO.*
- **Hebrew/English** language support (STT, commands, LLM, RTL UI) + the honest
  OCR-Hebrew gap.
- **GUI launcher** + one-click **Windows exe** (onedir) + icon.
- Voice/dictation quality improvements + **selectable Whisper model**.
- Board-capture mirror fix; mic feedback/timeout.
- Test suite grown to **175**.
- Expanded engineering-lessons narrative (bug journeys).
