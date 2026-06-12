# AirSketch — Academic Book Guide

> **For a future Claude session.** This file tells you *how* to write an academic
> book (technical monograph / thesis-style) about AirSketch. The *content* lives in
> `PROJECT_COMPENDIUM.md` (read it first, end-to-end). Use `SESSION_HANDOFF.md`,
> `HANDOFF.md`, `OPENVINO_HAND_TRACKING_PLAN.md`, and the source code for depth.

---

## 1. Goal, audience, scope

- **Goal:** a coherent, citable **academic book** documenting AirSketch — its
  motivation, design, the OpenVINO hand-tracking contribution, the full multimodal
  system, multilingual support, packaging, evaluation, and lessons.
- **Audience:** CS/EE students, instructors, and practitioners interested in on-device
  multimodal AI and applied computer vision. Self-contained: explain background.
- **Length target:** a short book / long thesis — roughly **60–120 pages** depending on
  figure density (the future session can scale chapters up/down; confirm with the user).
- **Tone:** academic but readable; first-person-plural ("we"); claims grounded in the
  Compendium's verified/to-be-verified distinctions.

## 2. Output format options (ask the user)

- **`.docx`** via the `anthropic-skills:docx` skill — best for a Word deliverable with
  ToC, headings, figures, captions. (Recommended default for an academic book.)
- **LaTeX → PDF** — best for a thesis look + bibliography (`.bib`); produce `.tex` and a
  build note (no guarantee a TeX toolchain is installed here — verify or deliver source).
- **Markdown master** — write `BOOK_DRAFT.md` first (single source), then render to
  docx/PDF. Recommended intermediate: draft in Markdown, then convert.

## 3. Recommended chapter outline (scope · depth · figures · Compendium ref)

**Front matter** — title page, abstract (adapt §0), ToC, list of figures/tables.

1. **Introduction** — motivation, the all-OpenVINO thesis, contributions list,
   book roadmap. · §0–§1
2. **Background & related work** — air-drawing/AR UIs; hand-pose estimation &
   **MediaPipe Hands / BlazePalm**; **OpenVINO** runtime & IR; **Whisper** STT;
   **PaddleOCR / PP-OCR**; **WeSpeaker** speaker verification; **Quick-Draw** dataset;
   quantization (INT4). Cite each. · §4–§6, §14
3. **System design & architecture** — package layout, data flow, entry points, the
   `HandResult`/analyzer contracts, configuration model. Figure: architecture diagram. · §3
4. **Hand tracking on OpenVINO (core chapter)** — the problem; TFLite-on-OpenVINO
   enabler; verified model I/O; the full host pipeline with **math**: SSD anchor
   generation (derive the 2016 count), box/keypoint decode equations, weighted NMS,
   rotated-ROI affine + inverse mapping, detect↔track state machine, EMA. The **two
   bugs** as a methodological case study (normalization; double-sigmoid). Validation &
   FPS. Figures: pipeline, anchor grid, ROI geometry, landmark-overlay before/after. · §5, §9
5. **The classroom application** — game state machine, gesture detection
   (angle+compactness fusion, world landmarks), judging/scoring, celebration, retry,
   the incremental lesson report. · §6
6. **Speech, board capture, and language understanding** — Whisper STT (command vs
   dictation configs), PP-OCR detection+recognition (DB + CTC, numpy post-proc),
   Qwen2.5-3B summarization, WeSpeaker speaker gating, the Qwen2-VL "cannot read text"
   finding. · §6
7. **Multilingual support (Hebrew)** — language toggle; multilingual Whisper; Hebrew
   command parsing + **morphology**; Hebrew LLM prompting; **RTL text rendering without
   a bidi dependency**; the **OpenVINO Hebrew-OCR gap** and why it's deferred. · §7
8. **Packaging & deployment** — GUI launcher (feature-flag UX), PyInstaller onedir exe,
   icon, model-path resolution, the windowed-stdout fix, build-robustness lessons
   (onefile-vs-onedir/AV, lazy torch, paging file). · §8, §9
9. **Engineering methodology & lessons** — *verify by running*; honest status
   taxonomy (works/partial/planned); the consolidated bug journeys as case studies;
   testing strategy (175 tests, the regression guards). · §9, §10
10. **Evaluation** — CNN 96%, hand-tracking FPS & parity, OCR/STT qualitative results,
    OpenVINO device portability; clearly separate measured vs to-be-measured. · §11
11. **Limitations & future work** — GPU/NPU benchmarking, Hebrew OCR model, STT WER,
    teacher-voice tuning, LCM image-gen, a user study. · §12
12. **Conclusion** — restate contributions; the all-OpenVINO outcome. · §0, §16

**Back matter / appendices:**
- A. **Reproducibility** — install, run, flags, build the exe, model inventory. · §13
- B. **API / module reference** — key classes & contracts (`HandResult`,
  `OpenVINOHandTracker`, analyzers, `VoiceController`). · §3
- C. **Test catalogue** — what each test file covers. · §10
- D. **Bibliography** — see §4 below.

## 4. References to include (resolve exact citations at write time)

MediaPipe Hands & BlazePalm (Google), OpenVINO toolkit & OpenVINO-GenAI (Intel),
OpenAI Whisper, PaddleOCR / PP-OCRv5 (Baidu), WeSpeaker (ResNet34 speaker
embeddings), Google Quick-Draw dataset, geaxgx `depthai_hand_tracker` (reference
implementation for anchors/decode/ROI), Qwen2.5 & Qwen2-VL (Alibaba), PyInstaller,
Tkinter, OpenCV, Pillow. Do **not** fabricate citation details — look up/confirm
each (year, venue, URL) when writing; mark any unverified as `[citation needed]`.

## 5. Figures to produce (and how)

- Architecture & hand-tracking pipeline diagrams (matplotlib/PIL or a draw tool).
- Anchor-grid illustration; rotated-ROI geometry; affine forward/inverse.
- Landmark-overlay images (regenerate by running `OpenVINOHandTracker` on a still and
  saving the overlay — done during development; see the session method).
- Result charts (FPS, CNN accuracy). Hebrew-UI and launcher screenshots.
- Use consistent captions and cross-reference from the text.

## 6. Writing process (for the future session)

1. Read `PROJECT_COMPENDIUM.md` fully; pull depth from source where a chapter needs it.
2. Confirm with the user: output format (docx/LaTeX/MD), page target, citation style,
   author/affiliation, English vs Hebrew (an academic book is likely English; confirm).
3. Draft `BOOK_DRAFT.md` chapter by chapter (Markdown master), citing Compendium
   sections; keep verified/to-be-verified honesty.
4. Generate figures; insert with captions.
5. Render to the chosen format (docx skill / LaTeX); build a bibliography.
6. Proofread for consistency with the code (don't drift from the actual implementation).
