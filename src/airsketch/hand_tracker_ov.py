"""OpenVINO hand tracking — a drop-in replacement for MediaPipeHandTracker.

Runs Google's two MediaPipe Hands sub-models (BlazePalm detector + 21-point
hand-landmark model) on the OpenVINO runtime (Intel CPU / GPU / NPU) instead of
TFLite/XNNPACK. The two ``.tflite`` files live inside
``models/hand_landmarker.task`` (a zip) and are read directly by
``openvino.Core.read_model`` (OpenVINO has a TFLite frontend).

The host-side pre/post-processing (anchor generation, SSD box decode, weighted
NMS, rotated-ROI crop, inverse landmark mapping, detect<->track state machine)
is re-implemented here in pure numpy + cv2 — no ``mediapipe`` dependency.

Public surface mirrors ``MediaPipeHandTracker`` exactly:

    tracker = OpenVINOHandTracker(device="AUTO")
    result  = tracker.process(frame_bgr)   # -> HandResult
    tracker.release()

Algorithm reference for the anchors / decode / rotated-ROI math:
geaxgx ``depthai_hand_tracker`` (``mediapipe_utils.py``) and the MediaPipe Hands
model cards. Constants that the live camera is sensitive to (ROI scale/shift,
input normalization, presence threshold) are module-level so they can be tuned
without touching the algorithm.
"""
from __future__ import annotations

import math
import os
import zipfile
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from airsketch.config import HandResult
from airsketch.hand_tracker import HandTracker, _HAND_CONNECTIONS

# ---------------------------------------------------------------------------
# Model locations
# ---------------------------------------------------------------------------
_TASK_BUNDLE = os.path.join("models", "hand_landmarker.task")
_OV_DIR = os.path.join("models", "hand_ov")
_PALM_TFLITE = "hand_detector.tflite"
_LM_TFLITE = "hand_landmarks_detector.tflite"

# ---------------------------------------------------------------------------
# Pipeline constants (tunable — the live camera is sensitive to these)
# ---------------------------------------------------------------------------
PALM_INPUT = 192            # BlazePalm input side
LM_INPUT = 224              # landmark model input side
NUM_ANCHORS = 2016
PALM_STRIDES = (8, 16, 16, 16)
PALM_ANCHOR_OFFSET = 0.5

# Input normalization, verified empirically against the converted .tflite graphs
# on real hand images: BOTH sub-models expect [0, 1] (x/255). Feeding the palm
# detector [-1, 1] yields garbage (off-frame boxes, score saturated to 1.0 on
# blank input) — the conversion baked the float range into the graph.
PALM_NORM = (0.0, 1.0)      # (x/255)
LM_NORM = (0.0, 1.0)        # (x/255)

PALM_SCORE_THRESH = 0.5     # min sigmoid(score) for a palm detection
PALM_NMS_IOU = 0.3          # weighted-NMS overlap threshold
PRESENCE_THRESH = 0.5       # min sigmoid(presence) for "hand visible"

# Palm box -> hand ROI (MediaPipe RectTransformation for hand)
HAND_ROI_SCALE = 2.6        # expand the (tight) palm box to cover the whole hand
HAND_ROI_SHIFT_Y = -0.5     # shift the ROI toward the fingers (along the hand axis)

# Tracking ROI derived from the previous frame's landmarks (already span the
# hand, so they need less expansion than the tight palm box).
HAND_TRACK_SCALE = 2.0

LM_SMOOTH_ALPHA = 0.5       # EMA on pixel landmarks (0 = none, 1 = no smoothing)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # clip first (MediaPipe score_clipping_thresh = 100); compute in float64 so
    # exp() of the clipped extreme doesn't overflow float32.
    x = np.clip(np.asarray(x, dtype=np.float64), -100.0, 100.0)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# BlazePalm anchors  (MediaPipe SsdAnchorsCalculator, fixed_anchor_size=True)
# ---------------------------------------------------------------------------
def generate_anchors(
    input_size: int = PALM_INPUT,
    strides: Sequence[int] = PALM_STRIDES,
    anchor_offset: float = PALM_ANCHOR_OFFSET,
    interpolated_scale_aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Generate SSD anchors for the BlazePalm detector.

    Returns an ``(N, 4)`` float32 array of ``[x_center, y_center, w, h]`` in
    normalized [0, 1] coordinates. With MediaPipe's ``fixed_anchor_size=True``
    every anchor has ``w == h == 1.0`` — only the centers vary — so the scale
    parameters affect only the *count* of anchors per cell, not their size.

    For ``strides=[8,16,16,16]`` over a 192 input this yields exactly 2016
    anchors (24*24*2 + 12*12*6).
    """
    anchors: List[Tuple[float, float, float, float]] = []
    num_layers = len(strides)
    layer_id = 0
    while layer_id < num_layers:
        # Merge consecutive layers that share a stride (MediaPipe behaviour).
        last = layer_id
        n_per_cell = 0
        while last < num_layers and strides[last] == strides[layer_id]:
            n_per_cell += 1                       # aspect ratio 1.0
            if interpolated_scale_aspect_ratio > 0.0:
                n_per_cell += 1                   # interpolated scale anchor
            last += 1

        stride = strides[layer_id]
        feature_map = int(math.ceil(input_size / stride))
        for y in range(feature_map):
            for x in range(feature_map):
                x_center = (x + anchor_offset) / feature_map
                y_center = (y + anchor_offset) / feature_map
                for _ in range(n_per_cell):
                    anchors.append((x_center, y_center, 1.0, 1.0))
        layer_id = last

    return np.asarray(anchors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Detection decode + weighted NMS
# ---------------------------------------------------------------------------
def decode_boxes(
    raw_boxes: np.ndarray, anchors: np.ndarray, input_size: int = PALM_INPUT
) -> np.ndarray:
    """Decode the detector's per-anchor regressors into boxes + 7 keypoints.

    ``raw_boxes`` is ``(N, 18)`` = [cx, cy, w, h, kp0x, kp0y, ... kp6x, kp6y]
    (reverse_output_order=True → x before y). Anchors are ``(N, 4)``.

    Returns ``(N, 18)`` where columns are
    ``[xmin, ymin, xmax, ymax, kp0x, kp0y, ... kp6x, kp6y]`` in **normalized
    [0, 1]** coordinates of the square detector input.
    """
    scale = float(input_size)
    ax = anchors[:, 0]
    ay = anchors[:, 1]
    aw = anchors[:, 2]
    ah = anchors[:, 3]

    cx = raw_boxes[:, 0] / scale * aw + ax
    cy = raw_boxes[:, 1] / scale * ah + ay
    w = raw_boxes[:, 2] / scale * aw
    h = raw_boxes[:, 3] / scale * ah

    out = np.empty_like(raw_boxes)
    out[:, 0] = cx - w / 2.0      # xmin
    out[:, 1] = cy - h / 2.0      # ymin
    out[:, 2] = cx + w / 2.0      # xmax
    out[:, 3] = cy + h / 2.0      # ymax

    # 7 keypoints (x, y) each, relative to anchor center
    for i in range(7):
        kx = 4 + i * 2
        ky = kx + 1
        out[:, kx] = raw_boxes[:, kx] / scale * aw + ax
        out[:, ky] = raw_boxes[:, ky] / scale * ah + ay
    return out


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box [xmin,ymin,xmax,ymax] vs an (M,4) array."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def weighted_nms(
    decoded: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = PALM_NMS_IOU,
    max_detections: int = 1,
) -> List[Tuple[np.ndarray, float]]:
    """MediaPipe-style *weighted* NMS.

    Overlapping boxes (IoU >= threshold) are merged into one by a
    score-weighted average of their coordinates + keypoints, rather than simply
    dropping the non-max ones. Returns up to ``max_detections`` ``(row, score)``
    tuples, highest score first. ``row`` has the same 18-column layout as
    ``decoded``.
    """
    if decoded.shape[0] == 0:
        return []
    order = np.argsort(scores)[::-1]
    boxes = decoded[:, :4]
    results: List[Tuple[np.ndarray, float]] = []
    remaining = order.tolist()

    while remaining and len(results) < max_detections:
        best = remaining[0]
        ious = _iou(boxes[best], boxes[remaining])
        overlap_mask = ious >= iou_threshold
        overlap_idx = [remaining[i] for i in range(len(remaining)) if overlap_mask[i]]

        w = scores[overlap_idx]
        w_sum = float(w.sum())
        if w_sum > 0:
            merged = (decoded[overlap_idx] * w[:, None]).sum(axis=0) / w_sum
        else:
            merged = decoded[best]
        results.append((merged.astype(np.float32), float(scores[best])))

        remaining = [remaining[i] for i in range(len(remaining)) if not overlap_mask[i]]

    return results


# ---------------------------------------------------------------------------
# Rotated ROI math
# ---------------------------------------------------------------------------
class ROI:
    """A rotated square region of interest, in frame pixel coordinates."""

    __slots__ = ("cx", "cy", "size", "rotation")

    def __init__(self, cx: float, cy: float, size: float, rotation: float):
        self.cx = cx
        self.cy = cy
        self.size = size
        self.rotation = rotation


def _rotation_from_vector(x0: float, y0: float, x1: float, y1: float) -> float:
    """Rotation (radians) that makes the vector (p0 -> p1) point 'up' in the crop.

    Uses image coordinates (y grows downward). An upright hand (p1 directly
    above p0) yields rotation 0.
    """
    return math.pi / 2.0 - math.atan2(-(y1 - y0), x1 - x0)


def roi_from_palm(
    decoded_row: np.ndarray, frame_w: int, frame_h: int,
    scale: float = HAND_ROI_SCALE, shift_y: float = HAND_ROI_SHIFT_Y,
) -> ROI:
    """Build the hand ROI from one decoded palm detection (normalized coords).

    Keypoint 0 (wrist) and keypoint 2 (middle-finger MCP) give the rotation.
    """
    xmin, ymin, xmax, ymax = decoded_row[:4]
    bcx = (xmin + xmax) / 2.0 * frame_w
    bcy = (ymin + ymax) / 2.0 * frame_h
    bw = (xmax - xmin) * frame_w
    bh = (ymax - ymin) * frame_h

    kp0x, kp0y = decoded_row[4] * frame_w, decoded_row[5] * frame_h
    kp2x, kp2y = decoded_row[8] * frame_w, decoded_row[9] * frame_h
    rotation = _rotation_from_vector(kp0x, kp0y, kp2x, kp2y)

    long_side = max(bw, bh)
    size = long_side * scale
    # Shift the centre toward the fingers along the rotated 'up' axis.
    dx = -(shift_y * long_side) * math.sin(rotation)
    dy = (shift_y * long_side) * math.cos(rotation)
    return ROI(bcx + dx, bcy + dy, size, rotation)


def roi_from_landmarks(
    landmarks_px: Sequence[Tuple[float, float]],
    scale: float = HAND_TRACK_SCALE,
) -> ROI:
    """Derive the next-frame ROI from the current 21 pixel landmarks.

    Center = wrist..middle-MCP region; rotation from wrist(0) -> middle-MCP(9);
    size from the landmark bounding box.
    """
    pts = np.asarray(landmarks_px, dtype=np.float32)
    wrist = pts[0]
    middle_mcp = pts[9]
    rotation = _rotation_from_vector(
        float(wrist[0]), float(wrist[1]), float(middle_mcp[0]), float(middle_mcp[1]))

    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    long_side = max(xmax - xmin, ymax - ymin)
    return ROI(float(cx), float(cy), float(long_side * scale), rotation)


def roi_affine(roi: ROI, out_size: int = LM_INPUT) -> np.ndarray:
    """2x3 affine mapping FRAME pixels -> the ``out_size`` crop.

    Built from three rotated-square corners so the inverse is a plain
    ``cv2.invertAffineTransform``. The crop is oriented so an upright hand
    appears upright.
    """
    s = roi.size / 2.0
    cosr = math.cos(roi.rotation)
    sinr = math.sin(roi.rotation)

    def corner(lx: float, ly: float) -> Tuple[float, float]:
        return (roi.cx + lx * cosr - ly * sinr,
                roi.cy + lx * sinr + ly * cosr)

    tl = corner(-s, -s)
    tr = corner(s, -s)
    bl = corner(-s, s)
    src = np.array([tl, tr, bl], dtype=np.float32)
    dst = np.array([(0, 0), (out_size - 1, 0), (0, out_size - 1)], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def crop_roi(frame: np.ndarray, roi: ROI, out_size: int = LM_INPUT
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the rotated ROI out of ``frame`` into an ``out_size`` square.

    Returns ``(crop_bgr, M)`` where ``M`` maps frame px -> crop px.
    """
    M = roi_affine(roi, out_size)
    crop = cv2.warpAffine(frame, M, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return crop, M


def project_points(points_crop: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map crop-pixel points (N,2) back to frame pixels via the inverse of M."""
    inv = cv2.invertAffineTransform(M)
    pts = np.asarray(points_crop, dtype=np.float32)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    return homog @ inv.T


# ---------------------------------------------------------------------------
# OpenVINO hand tracker
# ---------------------------------------------------------------------------
def _ensure_tflite_models(task_path: str = _TASK_BUNDLE, out_dir: str = _OV_DIR
                          ) -> Tuple[str, str]:
    """Extract the two .tflite sub-models from the .task bundle if needed."""
    palm = os.path.join(out_dir, _PALM_TFLITE)
    lm = os.path.join(out_dir, _LM_TFLITE)
    if os.path.exists(palm) and os.path.exists(lm):
        return palm, lm
    if not os.path.exists(task_path):
        raise FileNotFoundError(
            f"Hand model bundle not found at {task_path}. It ships with the app; "
            f"on a fresh checkout run the MediaPipe backend once to download it.")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(task_path) as z:
        z.extractall(out_dir)
    if not (os.path.exists(palm) and os.path.exists(lm)):
        raise RuntimeError(
            f"{task_path} did not contain the expected .tflite sub-models.")
    return palm, lm


class OpenVINOHandTracker(HandTracker):
    """Hand tracker running BlazePalm + hand-landmark on OpenVINO.

    Drop-in for :class:`MediaPipeHandTracker`: same ``process(frame) ->
    HandResult`` contract. Populates 21 pixel landmarks, index/thumb/wrist,
    hand_size, confidence (presence) and 21x3 world landmarks.
    """

    WRIST_ID = 0
    THUMB_TIP_ID = 4
    INDEX_TIP_ID = 8
    MIDDLE_MCP_ID = 9

    def __init__(
        self,
        device: str = "AUTO",
        detection_confidence: float = PALM_SCORE_THRESH,
        presence_confidence: float = PRESENCE_THRESH,
        max_hands: int = 1,
        model_dir: str = "models",
        draw: bool = True,
        debug: bool = False,
    ):
        import openvino as ov
        from airsketch.diagram_analyzer import pick_device

        self._draw = draw
        self._debug = debug
        self._score_thresh = detection_confidence
        self._presence_thresh = presence_confidence
        # live diagnostics (shown by the --hand-debug overlay)
        self.last_palm_score = 0.0
        self.last_presence = 0.0
        self.last_state = "init"

        task_path = os.path.join(model_dir, "hand_landmarker.task")
        out_dir = os.path.join(model_dir, "hand_ov")
        palm_path, lm_path = _ensure_tflite_models(task_path, out_dir)

        resolved = pick_device(device)
        self.device = resolved
        core = ov.Core()

        def _compile(path: str):
            model = core.read_model(path)
            try:
                return core.compile_model(model, resolved)
            except Exception as e:
                # NPU/GPU can reject these graphs; fall back to CPU gracefully.
                print(f"[hand-ov] compile on {resolved} failed for "
                      f"{os.path.basename(path)} ({type(e).__name__}); using CPU.")
                self.device = "CPU"
                return core.compile_model(model, "CPU")

        self._palm = _compile(palm_path)
        self._lm = _compile(lm_path)
        # Output ports keyed by name (order is not guaranteed across versions).
        self._palm_out = {p.get_any_name(): p for p in self._palm.outputs}
        self._lm_out = {p.get_any_name(): p for p in self._lm.outputs}

        self._anchors = generate_anchors()
        # detect<->track state
        self._roi: Optional[ROI] = None
        self._tracking = False
        self._smoothed: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ infer
    def _run_palm(self, frame: np.ndarray) -> Optional[ROI]:
        """Letterbox the frame to a square, detect the palm, return a hand ROI."""
        h, w = frame.shape[:2]
        side = max(h, w)
        pad_x = (side - w) // 2
        pad_y = (side - h) // 2
        square = np.zeros((side, side, 3), dtype=frame.dtype)
        square[pad_y:pad_y + h, pad_x:pad_x + w] = frame

        blob = self._preprocess(square, PALM_INPUT, PALM_NORM)
        res = self._palm(blob)
        raw_boxes = res[self._palm_out["Identity"]][0]        # (2016, 18)
        raw_scores = res[self._palm_out["Identity_1"]][0, :, 0]  # (2016,)
        scores = _sigmoid(raw_scores)

        self.last_palm_score = float(scores.max()) if scores.size else 0.0
        keep = scores >= self._score_thresh
        if not np.any(keep):
            return None
        decoded = decode_boxes(raw_boxes[keep], self._anchors[keep])
        dets = weighted_nms(decoded, scores[keep], max_detections=1)
        if not dets:
            return None

        # ROI in *square* pixels, then shift by the letterbox padding to frame px.
        roi_sq = roi_from_palm(dets[0][0], side, side)
        return ROI(roi_sq.cx - pad_x, roi_sq.cy - pad_y, roi_sq.size, roi_sq.rotation)

    def _run_landmarks(self, frame: np.ndarray, roi: ROI):
        """Run the landmark model on the ROI crop. Returns
        ``(landmarks_px(21,2), world(21,3), presence, handedness)`` or None."""
        crop, M = crop_roi(frame, roi, LM_INPUT)
        blob = self._preprocess(crop, LM_INPUT, LM_NORM)
        res = self._lm(blob)

        lm = res[self._lm_out["Identity"]].reshape(21, 3)
        # Identity_1 (presence) and Identity_2 (handedness) are ALREADY
        # probabilities in [0,1] in this converted graph — verified empirically:
        # a clear hand -> presence ~0.84, blank/noise -> ~0.006. Do NOT sigmoid
        # them again (double-sigmoid pins presence >= 0.5 forever, so the tracker
        # can never decide it lost the hand and gets stuck on a bad ROI).
        presence = float(np.asarray(res[self._lm_out["Identity_1"]]).reshape(-1)[0])
        handed = float(np.asarray(res[self._lm_out["Identity_2"]]).reshape(-1)[0])
        world = res[self._lm_out["Identity_3"]].reshape(21, 3).astype(np.float32)

        xy = lm[:, :2].astype(np.float32)
        # The model emits crop coords either in pixels (0..224) or normalized
        # (0..1) depending on the converted graph — auto-detect and rescale.
        if float(np.nanmax(np.abs(xy))) <= 1.5:
            xy = xy * LM_INPUT
        landmarks_px = project_points(xy, M)
        return landmarks_px, world, presence, handed

    @staticmethod
    def _preprocess(img_bgr: np.ndarray, size: int, norm: Tuple[float, float]
                    ) -> np.ndarray:
        """BGR frame -> NHWC RGB float32 tensor normalized into ``norm`` range."""
        if img_bgr.shape[0] != size or img_bgr.shape[1] != size:
            img_bgr = cv2.resize(img_bgr, (size, size))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        lo, hi = norm
        # map [0,255] -> [lo,hi]
        rgb = rgb / 255.0 * (hi - lo) + lo
        return rgb[np.newaxis, ...]   # (1, size, size, 3)

    # ----------------------------------------------------------------- public
    def process(self, frame: np.ndarray) -> HandResult:
        # 1) Get an ROI: reuse the tracked one, else run palm detection.
        if not self._tracking or self._roi is None:
            self._roi = self._run_palm(frame)
            if self._roi is None:
                self.last_presence = 0.0
                self.last_state = "no_palm"
                self._reset_track()
                return self._finish(HandResult(visible=False), frame)
            self.last_state = "detected"
        else:
            self.last_state = "tracking"

        out = self._run_landmarks(frame, self._roi)
        if out is None:
            self._reset_track()
            return self._finish(HandResult(visible=False), frame)
        landmarks_px, world, presence, handed = out
        self.last_presence = presence

        if presence < self._presence_thresh:
            # Lost the hand. Drop tracking; next frame re-detects with the palm model.
            self.last_state = "low_presence"
            self._reset_track()
            return self._finish(HandResult(visible=False), frame)

        # 2) Smooth + update the tracking ROI from these landmarks.
        landmarks_px = self._smooth(landmarks_px)
        self._roi = roi_from_landmarks([tuple(p) for p in landmarks_px])
        self._tracking = True

        pixel_landmarks = [(float(x), float(y)) for x, y in landmarks_px]
        world_landmarks = [(float(a), float(b), float(c)) for a, b, c in world]

        if self._draw:
            self._draw_landmarks(frame, pixel_landmarks)

        index_tip = pixel_landmarks[self.INDEX_TIP_ID]
        thumb_tip = pixel_landmarks[self.THUMB_TIP_ID]
        wrist = pixel_landmarks[self.WRIST_ID]
        middle_mcp = pixel_landmarks[self.MIDDLE_MCP_ID]
        hand_size = math.hypot(wrist[0] - middle_mcp[0], wrist[1] - middle_mcp[1])
        if hand_size < 1.0:
            hand_size = 1.0

        result = HandResult(
            visible=True,
            fingertip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            hand_size=hand_size,
            confidence=presence,
            landmarks=pixel_landmarks,
            world_landmarks=world_landmarks,
        )
        return self._finish(result, frame)

    def _finish(self, result: HandResult, frame: np.ndarray) -> HandResult:
        if self._debug:
            self._draw_debug(frame, result)
        return result

    def _draw_debug(self, frame: np.ndarray, result: HandResult) -> None:
        """Top-left overlay: is palm detection firing? what's the presence/state?"""
        col_ok = (90, 230, 140)
        col_bad = (80, 90, 255)
        lines = [
            f"hand-ov [{self.device}] state={self.last_state}",
            f"palm score={self.last_palm_score:.2f} (thr {self._score_thresh:.2f})",
            f"presence ={self.last_presence:.2f} (thr {self._presence_thresh:.2f})",
            f"visible  ={result.visible}",
        ]
        y = 120
        cv2.rectangle(frame, (8, y - 16), (330, y + 64), (0, 0, 0), -1)
        for i, t in enumerate(lines):
            c = col_ok if (i == 3 and result.visible) else (
                col_bad if i == 3 else (230, 230, 235))
            cv2.putText(frame, t, (14, y + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, 1, cv2.LINE_AA)

    def _smooth(self, landmarks_px: np.ndarray) -> np.ndarray:
        if LM_SMOOTH_ALPHA >= 1.0 or self._smoothed is None:
            self._smoothed = landmarks_px.copy()
        else:
            a = LM_SMOOTH_ALPHA
            self._smoothed = a * landmarks_px + (1.0 - a) * self._smoothed
        return self._smoothed

    def _reset_track(self) -> None:
        self._tracking = False
        self._roi = None
        self._smoothed = None

    def _draw_landmarks(self, frame: np.ndarray, pixel_landmarks: list) -> None:
        pts = [(int(x), int(y)) for x, y in pixel_landmarks]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if i == self.INDEX_TIP_ID:
                color, r = (0, 255, 255), 5
            elif i == self.THUMB_TIP_ID:
                color, r = (255, 180, 0), 5
            else:
                color, r = (0, 180, 255), 3
            cv2.circle(frame, p, r, color, -1, cv2.LINE_AA)

    def release(self) -> None:
        self._palm = None
        self._lm = None
