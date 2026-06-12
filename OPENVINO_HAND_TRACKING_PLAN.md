# Plan — Replace MediaPipe hand tracking with an OpenVINO pipeline

> **Status: DONE — DEFAULT, validated live.** Shipped as
> `src/airsketch/hand_tracker_ov.py` (`OpenVINOHandTracker`); now the **default**
> hand backend (`hand_tracker_backend="openvino"`), with MediaPipe as the
> automatic fallback (`--hand-backend mediapipe`). Device via
> `--hand-device {AUTO,CPU,GPU,NPU}`, troubleshooting via `--hand-debug`.
> Confirmed live on the Check Point VDI: air-drawing scored real strokes with
> `[hand] OpenVINO hand tracker on CPU` (~108 FPS). Two bring-up bugs fixed:
> input normalization is [0,1] (not [-1,1]); the landmark presence/handedness
> outputs are already probabilities (don't sigmoid — double-sigmoid made the
> tracker stick on a bad ROI = the "hand not recognized" symptom). See
> "Implementation notes" at the bottom.
> **Goal:** make hand tracking run on the OpenVINO runtime (Intel CPU / GPU / NPU),
> closing the last non-OpenVINO inference path so the *whole* app honors the
> original "all inference on OpenVINO, Intel CPU-GPU-NPU" brief.

---

## 1. Why

Today every ML model in AirSketch runs on OpenVINO and is device-selectable
**except hand tracking**, which uses Google MediaPipe (`mediapipe.tasks` →
TFLite / XNNPACK, **CPU only**). See `src/airsketch/hand_tracker.py`
(`MediaPipeHandTracker`). That file even has commented-out OpenVINO scaffolding
(`# self._core = Core(); self._model = self._core.compile_model(...)`) anticipating
this swap.

Replacing it with OpenVINO lets hand tracking use GPU/NPU and removes the
`mediapipe` dependency.

## 2. Key finding — models are already in the repo and OpenVINO reads them directly

`models/hand_landmarker.task` is a **zip bundle** containing the two TFLite
sub-models. **Verified: `openvino.Core().read_model()` loads both TFLite files
directly** (OpenVINO has a TFLite frontend — no `ovc`/ONNX conversion strictly
required, though we can pre-convert to IR for speed/NPU; see §7).

Extract with:
```python
import zipfile; zipfile.ZipFile("models/hand_landmarker.task").extractall("models/hand_ov")
# -> hand_detector.tflite (2.3 MB), hand_landmarks_detector.tflite (5.5 MB)
```

### Verified I/O (via `ov.Core().read_model`)

**`hand_detector.tflite`** — BlazePalm-style SSD palm detector
- input  `input_1`  `[1, 192, 192, 3]`  (NHWC, RGB)
- output `Identity`   `[1, 2016, 18]`  — per-anchor box+keypoint regressors (4 box + 7 keypoints×2 = 18)
- output `Identity_1` `[1, 2016, 1]`   — per-anchor palm score (apply sigmoid)
- → **2016 anchors**; strides `[8,16,16,16]` over 192 input give feature maps
  24²+12²+12²+12² = 1008 cells × 2 anchors = 2016.

**`hand_landmarks_detector.tflite`** — 21-point hand landmark model
- input  `input_1`  `[1, 224, 224, 3]`  (NHWC, RGB)
- output `Identity`   `[1, 63]`  — 21 landmarks ×3 (x,y,z) in the **224 crop** space
- output `Identity_1` `[1, 1]`   — hand presence score (sigmoid)
- output `Identity_2` `[1, 1]`   — handedness (left/right)
- output `Identity_3` `[1, 63]`  — **world landmarks** (21×3, metric, wrist-relative)

> `Identity_3` (world landmarks) maps straight onto the `HandResult.world_landmarks`
> field that `IndexPointingDetector` prefers for orientation-invariant gestures —
> so we lose nothing vs. MediaPipe.

## 3. How MediaPipe Hands actually works (what we must reimplement)

Two-stage pipeline + a tracking shortcut:

1. **Palm detection** (run on the full frame, only when we don't already have a hand):
   - letterbox/resize frame → 192×192, RGB, normalize to `[-1, 1]` (`x/127.5 - 1`).
   - run detector → decode the 2016 anchors: `score = sigmoid(Identity_1)`,
     box+keypoints = `Identity` decoded against anchor centers/sizes.
   - filter by score (≈0.5), **weighted NMS** → best palm box + 7 keypoints.
   - from keypoints 0 (wrist) and 2 (middle-finger MCP) compute the **hand rotation angle**.

2. **ROI extraction**:
   - expand the palm box (~2.6×) into a square, rotate it by the angle.
   - affine-warp that rotated square out of the frame → 224×224 crop.

3. **Landmark detection** (run on the 224 crop):
   - normalize crop → run landmark model → 21 (x,y,z) in crop coords + presence.
   - **inverse-transform** the landmarks through the ROI affine back to **frame pixel coords**.

4. **Tracking shortcut** (the speed/stability trick): once a hand is found, derive the
   *next* frame's ROI from the **previous landmarks** (palm-bbox of landmarks, rotated),
   and **skip palm detection** while presence stays high. Re-run palm detection only when
   presence drops or no hand is tracked.

## 4. Target design — `OpenVINOHandTracker`

Add `src/airsketch/hand_tracker_ov.py` with a class that is a **drop-in** for
`MediaPipeHandTracker`: same `process(frame) -> HandResult` contract so nothing
downstream changes.

`HandResult` fields to populate (see `config.py`):
| field | source |
|---|---|
| `visible` | presence ≥ threshold |
| `landmarks` (21×2 px) | landmark `Identity` → inverse ROI transform → frame px |
| `fingertip` | `landmarks[8]` (index tip) |
| `thumb_tip` | `landmarks[4]` |
| `wrist` | `landmarks[0]` |
| `hand_size` | reference scale, e.g. ‖wrist − middle_MCP‖ in px (used for pinch scaling) |
| `confidence` | presence score |
| `world_landmarks` (21×3) | landmark `Identity_3` |

Construction mirrors the existing engines:
```python
core = ov.Core()
self._palm = core.compile_model(core.read_model(palm_path), device)
self._lm   = core.compile_model(core.read_model(lm_path),   device)
```
`device` from a new `cfg.hand_device` (`AUTO|CPU|GPU|NPU`).

## 5. Sub-tasks (the real work)

- **Anchor generation** for BlazePalm 192 (SSD anchors): `num_layers=4`,
  `strides=[8,16,16,16]`, `anchor_offset=0.5`, 2 anchors/cell → 2016. Must match
  MediaPipe's `SsdAnchorsCalculator` exactly or boxes are wrong. (Reference impl: geaxgx
  `depthai_hand_tracker`, file `mediapipe_utils.py`.)
- **Detection decode + weighted NMS** (numpy).
- **Rotated ROI** math: angle from keypoints, square expansion, `cv2.getAffineTransform` /
  `warpAffine` to 224×224, and the inverse map for landmarks.
- **Normalization** constants for each model (`/127.5 - 1` is the MediaPipe convention;
  confirm against the converted graph).
- **Tracking state machine** (detect ↔ track) — important for FPS and stability.
- **Smoothing** (optional): MediaPipe applies a landmark velocity filter; a light EMA on
  landmarks reduces jitter.

## 6. Pure-numpy host code (no new deps)

Pre/post-processing is numpy + cv2 (both already used). No new dependency is added —
we *remove* `mediapipe` once this is validated.

## 7. Model acquisition options (in priority order)

1. **In-repo TFLite, read directly by OpenVINO** (primary, zero download):
   extract the two `.tflite` from `models/hand_landmarker.task`, `core.read_model(...)`.
2. **Pre-convert to OpenVINO IR** for faster load and best GPU/NPU behaviour:
   `ovc hand_detector.tflite --output_model models/hand_ov/palm.xml` (and the landmark model).
   IR also lets us **reshape to static** shapes for NPU.
3. **Alternate sources** if the in-repo TFLite ever misbehaves: PINTO0309/PINTO_model_zoo
   publishes BlazePalm + hand-landmark in ONNX/IR; MediaPipe also ships the raw `.tflite`
   on its model storage. (Verify exact URLs/repo ids at implementation time — do not hardcode unverified links.)

## 8. Device / CPU-GPU-NPU notes

- Both models are **small and fixed-shape** → strong CPU/GPU performance and, unlike the
  dynamic-shape OCR/LLM models, **good NPU candidates** (NPU dislikes dynamic shapes).
- For NPU, convert to IR and `reshape` to the static input shapes (192³, 224³) before compile.
- Add `--hand-backend {mediapipe,openvino}` (default `mediapipe` until OV is validated) and
  `--hand-device {AUTO,CPU,GPU,NPU}`. Keep `MediaPipeHandTracker` as a fallback path.

## 9. Phased implementation

| Phase | Deliverable | Status |
|---|---|---|
| A | Extract models, confirm `read_model` + compile per device; pin I/O. | DONE (verified OV 2026.1 reads both .tflite; I/O exactly as documented). |
| B | Palm detector wrapper: preproc + anchors + decode + NMS → palm box + angle. Unit-test decode. | DONE (`generate_anchors`/`decode_boxes`/`weighted_nms`; 2016 anchors; unit-tested). |
| C | ROI affine (crop+rotate to 224) + inverse transform. | DONE (`roi_affine`/`crop_roi`/`project_points`; round-trip unit-tested). |
| D | Landmark wrapper → 21 px landmarks + world landmarks + presence, mapped to frame. | DONE (`_run_landmarks`; world landmarks from `Identity_3`). |
| E | Detect↔track state machine (skip palm detect while presence high). | DONE (verified: 5 frames → 1 palm-detect call, then tracks). |
| F | `OpenVINOHandTracker.process()` → `HandResult`; wire `--hand-backend/--hand-device`. | DONE (factory `create_hand_tracker`; flags in `main.py` + `classroom/app.py`). |
| G | Validate: gestures behave; FPS on CPU; spot-check GPU/NPU. | DONE + **live-validated on the VDI** (air-drawing scored real strokes; ~108 FPS CPU). Now the default backend. GPU/NPU not available in this env — graceful fallback wired. `mediapipe` kept as automatic fallback. |

## 10. Acceptance criteria

- Notebook + classroom run with `--hand-backend openvino` and draw correctly.
  *(Wired + CPU pipeline validated on real still images; live-camera run is the
  user's final confirmation.)*
- Index-pointing pen-up/down and pinch detection match MediaPipe qualitatively.
  *(Gesture detector consumes the same `landmarks` + `world_landmarks` contract,
  which the OV tracker fills from `Identity`/`Identity_3`; angle/compactness logic
  is scale/rotation-robust so behaviour should match — **user to confirm live**.)*
- Runs on CPU (parity) and at least one accelerator (GPU); NPU attempted with static IR.
  *(CPU: ~108 FPS steady-state. Both models are already fixed-shape — good NPU
  fits with no reshape needed. GPU/NPU compile attempted with CPU fallback; not
  testable here, only CPU present.)*
- `mediapipe` import no longer required when OV backend is selected.
  *(True at runtime: selecting `openvino` never imports `mediapipe`. The dep is
  intentionally KEPT in `pyproject.toml` because MediaPipe is still the default
  backend until live validation; drop it only after the user signs off.)*

## Implementation notes (what was verified, 2026-06-07)

- **Files:** `src/airsketch/hand_tracker_ov.py` (the tracker + all pure-numpy
  math), `tests/test_hand_tracker_ov.py` (26 tests), factory
  `create_hand_tracker()` in `hand_tracker.py`, config `hand_tracker_backend` +
  `hand_device`, CLI flags in both entry points.
- **Normalization (IMPORTANT, verified empirically):** BOTH sub-models expect
  **`[0,1]` (x/255)**, *not* the `[-1,1]` the MediaPipe graph docs imply. Feeding
  the palm detector `[-1,1]` made it saturate to score 1.0 on uniform/blank input
  and emit off-frame boxes (phantom hands). Confirmed on real hand stills: `[0,1]`
  gives score 0.65–0.88 with valid in-frame boxes; `[-1,1]` gives garbage. A
  blank-frame-not-visible regression test guards this.
- **End-to-end visual check:** ran the full pipeline on real captured app frames
  (predecessor `airdraw_ar/outputs`) and overlaid the skeleton — landmarks fit a
  real (horizontally-oriented) hand correctly, proving decode + rotated-ROI +
  inverse-mapping are right, including rotation handling.
- **Landmark coords:** the model may emit crop coords as pixels (0..224) or
  normalized (0..1) depending on the converted graph; the wrapper auto-detects and
  rescales.
- **Presence/handedness (IMPORTANT, verified empirically):** the landmark model's
  `Identity_1` (presence) and `Identity_2` (handedness) outputs are **already
  probabilities** (clear hand presence ~0.84, blank/noise ~0.006) — do NOT apply
  sigmoid. The earlier double-sigmoid pinned presence ≥ 0.5 for any input, so the
  tracker could never decide it had lost the hand and got stuck tracking a bad ROI
  (the live "hand not recognized at all" symptom). A blank-frame-not-visible test
  guards this too.
- **Perf:** ~108 FPS CPU steady-state (tracking path), well above realtime.
- **Live-validated on the VDI:** air-drawing produced real scored strokes
  (triangle 92%, rectangle 90%) with `[hand] OpenVINO hand tracker on CPU`. Now the
  default backend.
- **Still to verify (optional):** GPU/NPU on real Intel hardware (only CPU in dev env).

## 11. Risks & mitigations

- **Anchor/decoder mismatch** → garbage boxes. *Mitigation:* port exact params from a known
  reference (geaxgx) and unit-test decode.
- **ROI math errors** → landmarks offset/rotated. *Mitigation:* draw the ROI + landmarks
  overlay during bring-up; compare to MediaPipe side-by-side.
- **Perf regression** (two models/frame). *Mitigation:* the detect↔track shortcut; run palm
  detect at low cadence.
- **NPU dynamic-shape limits.** *Mitigation:* static IR via `ovc` + `reshape`.

## 12. Effort

Medium-large (comparable to the PP-OCR effort): ~the BlazePalm decode + rotated-ROI glue is
the bulk. Models and I/O are already verified, which removes the biggest unknown.

## 13. References

- Existing code: `src/airsketch/hand_tracker.py` (interface + commented OV scaffolding),
  `src/airsketch/gesture_detector.py` (consumer of landmarks/world_landmarks),
  `src/airsketch/inference_engine.py` (OpenVINO Core wrapper pattern to reuse).
- Algorithm reference: geaxgx `depthai_hand_tracker` (`mediapipe_utils.py` — anchors, decode, ROI).
- Model card: MediaPipe Hands (palm detection + hand landmark).
- OpenVINO TFLite frontend (`Core.read_model` on `.tflite`) and `ovc` for IR conversion.
