"""PP-OCR text reader running on OpenVINO (no PaddlePaddle dependency).

Loads PaddleOCR's PP-OCRv5 detection + English recognition models, exported to
ONNX (which OpenVINO reads natively), and exposes `read()` -> text lines and
`read_text()` -> an assembled transcription.

The inference backend is the OpenVINO `Core`, so these ONNX models can later be
swapped for pre-converted OpenVINO IR (`.xml`) with NO caller changes — that is
the "future conversion to OpenVINO" path, already half-done since OV runs the
ONNX directly today.

Pipeline (classic PP-OCR):
    detection (DB)  ->  boxes  ->  per-box crop  ->  recognition (CRNN + CTC)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

_MODELS = Path(__file__).resolve().parent.parent.parent / "models" / "ppocr"
_DET = _MODELS / "detection__v5__det.onnx"
_REC = _MODELS / "languages__english__rec.onnx"
_DICT = _MODELS / "languages__english__dict.txt"

# DB detection normalization (PaddleOCR default — ImageNet stats after /255)
_DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_DET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
_REC_HEIGHT = 48


@dataclass
class TextLine:
    text: str
    confidence: float
    box: np.ndarray   # (4, 2) float32, original-image coords
    cx: float
    cy: float
    height: float


def assemble_text(lines: List["TextLine"]) -> str:
    """Order recognized lines top-to-bottom, left-to-right into a transcription.

    Boxes whose vertical centers are within ~0.6x the median line height are
    treated as the same row and joined left-to-right with spaces.
    """
    if not lines:
        return ""
    median_h = float(np.median([l.height for l in lines]))
    tol = max(median_h * 0.6, 8.0)
    ordered = sorted(lines, key=lambda l: l.cy)
    rows: List[List["TextLine"]] = []
    for l in ordered:
        if rows and abs(l.cy - rows[-1][0].cy) <= tol:
            rows[-1].append(l)
        else:
            rows.append([l])
    out = []
    for row in rows:
        row.sort(key=lambda l: l.cx)
        out.append(" ".join(l.text for l in row))
    return "\n".join(out)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _crop_rotated(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Perspective-rectify a (possibly rotated) text box into a horizontal crop."""
    box = _order_points(box.astype(np.float32))
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE,
                               flags=cv2.INTER_CUBIC)
    # Vertical text -> rotate to horizontal
    if h > 0 and w > 0 and (h / float(w)) >= 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


class PPOCROpenVINOReader:
    """PaddleOCR PP-OCR detection + recognition on the OpenVINO runtime."""

    name = "ppocr_openvino"

    def __init__(
        self,
        device: str = "CPU",
        det_limit_side: int = 960,
        det_thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.6,
        min_confidence: float = 0.5,
    ):
        import openvino as ov

        for p in (_DET, _REC, _DICT):
            if not p.exists():
                raise FileNotFoundError(f"PP-OCR model file missing: {p}")

        self._limit = det_limit_side
        self._det_thresh = det_thresh
        self._box_thresh = box_thresh
        self._unclip_ratio = unclip_ratio
        self._min_conf = min_confidence

        core = ov.Core()
        self._det = core.compile_model(core.read_model(str(_DET)), device)
        self._rec = core.compile_model(core.read_model(str(_REC)), device)
        self._det_out = self._det.output(0)
        self._rec_out = self._rec.output(0)

        # Character list: index 0 = CTC blank, 1..N = dict chars, N+1 = space.
        chars = _DICT.read_text(encoding="utf-8").splitlines()
        self._charset = ["<blank>"] + chars + [" "]

    # ----- detection -----

    def _detect(self, bgr: np.ndarray) -> List[np.ndarray]:
        h0, w0 = bgr.shape[:2]
        scale = min(self._limit / max(h0, w0), 1.0)
        nh = max(int(round(h0 * scale / 32)) * 32, 32)
        nw = max(int(round(w0 * scale / 32)) * 32, 32)

        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh)), cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - _DET_MEAN) / _DET_STD
        inp = norm.transpose(2, 0, 1)[None]            # NCHW

        prob = self._det([inp])[self._det_out][0, 0]    # (nh, nw) in [0,1]
        bitmap = (prob > self._det_thresh).astype(np.uint8)

        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rx, ry = w0 / float(nw), h0 / float(nh)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 9:
                continue
            score = self._box_score(prob, cnt)
            if score < self._box_thresh:
                continue
            box = self._unclip(cnt)
            if box is None:
                continue
            box[:, 0] = np.clip(box[:, 0] * rx, 0, w0 - 1)
            box[:, 1] = np.clip(box[:, 1] * ry, 0, h0 - 1)
            boxes.append(box.astype(np.float32))
        return boxes

    @staticmethod
    def _box_score(prob: np.ndarray, contour: np.ndarray) -> float:
        h, w = prob.shape
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 1)
        vals = prob[mask == 1]
        return float(vals.mean()) if vals.size else 0.0

    def _unclip(self, contour: np.ndarray) -> Optional[np.ndarray]:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), ang = rect
        if min(w, h) < 3:
            return None
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        if length <= 0:
            return None
        dist = area * self._unclip_ratio / length      # DB unclip distance
        rect = ((cx, cy), (w + 2 * dist, h + 2 * dist), ang)
        return cv2.boxPoints(rect)

    # ----- recognition -----

    def _recognize(self, crop: np.ndarray) -> tuple[str, float]:
        h, w = crop.shape[:2]
        if h < 1 or w < 1:
            return "", 0.0
        new_w = max(1, int(round(_REC_HEIGHT * w / float(h))))
        img = cv2.resize(crop, (new_w, _REC_HEIGHT))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb / 255.0 - 0.5) / 0.5                # -> [-1, 1]
        inp = norm.transpose(2, 0, 1)[None]
        logits = self._rec([inp])[self._rec_out][0]      # (T, num_classes)
        return self._ctc_decode(logits)

    def _ctc_decode(self, logits: np.ndarray) -> tuple[str, float]:
        idxs = logits.argmax(axis=1)
        confs = logits.max(axis=1)
        out, probs, prev = [], [], -1
        for i, ix in enumerate(idxs):
            if ix != 0 and ix != prev:                   # skip blank + repeats
                if ix < len(self._charset):
                    out.append(self._charset[ix])
                    probs.append(confs[i])
            prev = ix
        text = "".join(out).strip()
        conf = float(np.mean(probs)) if probs else 0.0
        return text, conf

    # ----- public API -----

    def read(self, bgr: np.ndarray) -> List[TextLine]:
        lines: List[TextLine] = []
        for box in self._detect(bgr):
            text, conf = self._recognize(_crop_rotated(bgr, box))
            if not text or conf < self._min_conf:
                continue
            cx = float(box[:, 0].mean())
            cy = float(box[:, 1].mean())
            height = float(np.linalg.norm(box[0] - box[3]))
            lines.append(TextLine(text, conf, box, cx, cy, max(height, 1.0)))
        return lines

    def read_text(self, bgr: np.ndarray) -> str:
        """Detect + recognize, then assemble lines top-to-bottom, left-to-right."""
        return assemble_text(self.read(bgr))
