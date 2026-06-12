import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np

from airsketch.config import RecognitionResult
from airsketch.utils import bounding_box, centroid, distance, normalize_points

# --- OpenVINO / ONNX integration point ---
# To replace geometric recognition with a learned classifier:
#   1. Subclass ShapeRecognizerBase
#   2. Load an ONNX or OpenVINO IR sketch-classification model
#   3. Rasterize the stroke points onto a small canvas (e.g. 64x64)
#   4. Run inference and map output logits to shape labels
# The rest of the application remains unchanged.


class ShapeRecognizerBase(ABC):
    """Interface for shape recognition backends."""

    @abstractmethod
    def recognize(
        self, points: List[Tuple[float, float]], is_closed_hint: bool | None = None
    ) -> RecognitionResult:
        ...


class ShapeRecognizer(ShapeRecognizerBase):
    """Geometric feature-based shape recognizer (no training required)."""

    MIN_POINTS = 10

    def recognize(
        self, points: List[Tuple[float, float]], is_closed_hint: bool | None = None
    ) -> RecognitionResult:
        if len(points) < self.MIN_POINTS:
            return RecognitionResult("unknown", 0.0, "Too few points to recognize")

        norm = normalize_points(points, canvas_size=200)
        pts_array = np.array(norm, dtype=np.float32)
        contour = pts_array.reshape(-1, 1, 2).astype(np.int32)

        features = self._extract_features(norm, contour, is_closed_hint)
        return self._classify(features)

    def _extract_features(
        self,
        points: List[Tuple[float, float]],
        contour: np.ndarray,
        is_closed_hint: bool | None,
    ) -> dict:
        x, y, w, h = bounding_box(points)
        aspect = w / max(h, 1e-6)
        perimeter_open = cv2.arcLength(contour, closed=False)
        perimeter_closed = cv2.arcLength(contour, closed=True)
        closure = distance(points[0], points[-1])
        closure_ratio = closure / max(perimeter_open, 1e-6)

        if is_closed_hint is None:
            is_closed = closure_ratio < 0.2
        else:
            is_closed = is_closed_hint

        perimeter = perimeter_closed if is_closed else perimeter_open

        # Multi-scale corner counting — pick the count that's most stable
        epsilons = [0.008, 0.015, 0.025, 0.04, 0.06]
        corner_counts = [
            len(cv2.approxPolyDP(contour, k * perimeter, closed=is_closed))
            for k in epsilons
        ]
        corners_finest = corner_counts[0]   # 0.008 — picks up star concavities
        corners_tight = corner_counts[2]    # 0.025
        corners_loose = corner_counts[3]    # 0.04
        corners_very_loose = corner_counts[4]  # 0.06

        area = cv2.contourArea(contour)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1e-6)

        cx, cy = centroid(points)
        dists = [distance(p, (cx, cy)) for p in points]
        dist_std = float(np.std(dists))
        dist_mean = float(np.mean(dists))
        dist_cv = dist_std / max(dist_mean, 1e-6)

        direction_changes = self._count_direction_changes(points)
        elongation = max(w, h) / max(min(w, h), 1e-6)

        return {
            "aspect": aspect,
            "closure_ratio": closure_ratio,
            "is_closed": is_closed,
            "circularity": circularity,
            "corners_finest": corners_finest,
            "corners_tight": corners_tight,
            "corners_loose": corners_loose,
            "corners_very_loose": corners_very_loose,
            "solidity": solidity,
            "dist_cv": dist_cv,
            "area": area,
            "perimeter": perimeter,
            "direction_changes": direction_changes,
            "elongation": elongation,
            "w": w,
            "h": h,
        }

    def _count_direction_changes(self, points: List[Tuple[float, float]]) -> int:
        if len(points) < 3:
            return 0
        changes = 0
        step = max(1, len(points) // 30)
        sampled = points[::step]
        for i in range(2, len(sampled)):
            dx1 = sampled[i - 1][0] - sampled[i - 2][0]
            dy1 = sampled[i - 1][1] - sampled[i - 2][1]
            dx2 = sampled[i][0] - sampled[i - 1][0]
            dy2 = sampled[i][1] - sampled[i - 1][1]
            cross = dx1 * dy2 - dy1 * dx2
            if abs(cross) > 15:
                changes += 1
        return changes

    def _classify(self, f: dict) -> RecognitionResult:
        is_closed = f["is_closed"]
        candidates: list[Tuple[str, float, str]] = []

        # ---- CLOSED-shape candidates ----
        if is_closed:
            # Circle: high circularity, uniform distance from centroid
            if f["circularity"] > 0.55 and f["dist_cv"] < 0.30:
                conf = 0.55 + 0.30 * f["circularity"] + 0.15 * (1 - f["dist_cv"])
                conf = min(0.99, conf)
                candidates.append((
                    "circle", conf,
                    f"circularity={f['circularity']:.2f}, dist_cv={f['dist_cv']:.2f}",
                ))
            elif f["corners_loose"] <= 3 and f["circularity"] > 0.45 and f["dist_cv"] < 0.4:
                conf = 0.50 + 0.20 * f["circularity"]
                candidates.append((
                    "circle", conf,
                    f"low-corner circle, circ={f['circularity']:.2f}",
                ))

            # Triangle: 3 corners (loose), or 3-4 (very loose) — high solidity
            if f["corners_loose"] == 3 and f["solidity"] > 0.7:
                conf = 0.75 + 0.20 * f["solidity"]
                candidates.append((
                    "triangle", conf,
                    f"corners_loose=3, solidity={f['solidity']:.2f}",
                ))
            elif f["corners_very_loose"] == 3 and f["solidity"] > 0.75:
                conf = 0.65 + 0.15 * f["solidity"]
                candidates.append((
                    "triangle", conf,
                    f"corners_very_loose=3, solidity={f['solidity']:.2f}",
                ))

            # Rectangle: 4 corners with decent aspect ratio and high solidity
            if f["corners_loose"] == 4 and f["solidity"] > 0.75:
                aspect_score = 1.0 - min(1.0, abs(math.log(f["aspect"])) / 2.0)
                conf = 0.65 + 0.20 * f["solidity"] + 0.10 * aspect_score
                conf = min(0.99, conf)
                candidates.append((
                    "rectangle", conf,
                    f"corners=4, aspect={f['aspect']:.2f}, solidity={f['solidity']:.2f}",
                ))
            elif f["corners_very_loose"] == 4 and f["solidity"] > 0.8:
                conf = 0.60 + 0.15 * f["solidity"]
                candidates.append((
                    "rectangle", conf,
                    f"corners_very_loose=4, solidity={f['solidity']:.2f}",
                ))

            # Heart: closed, dip at top, moderate circularity, 0.7-1.4 aspect
            if (0.30 < f["circularity"] < 0.65 and 0.7 < f["aspect"] < 1.4
                    and 0.6 < f["solidity"] < 0.92 and f["corners_loose"] in (4, 5, 6)):
                conf = 0.55 + 0.10 * (1 - abs(f["aspect"] - 1))
                candidates.append((
                    "heart", conf,
                    f"circ={f['circularity']:.2f}, solidity={f['solidity']:.2f}",
                ))

            # Star: many corners at fine scale AND clear concavities (low solidity)
            # AND big differential vs coarse scale AND reasonably circular bbox.
            # These together filter out hand-drawn noise from real star shapes.
            corner_diff = f["corners_finest"] - f["corners_very_loose"]
            bbox_aspect_ok = 0.6 < f["aspect"] < 1.7  # stars are roughly square
            if (f["corners_finest"] >= 10
                    and corner_diff >= 4
                    and f["solidity"] < 0.78
                    and bbox_aspect_ok):
                conf = min(0.90, 0.45 + 0.03 * f["corners_finest"]
                                 + 0.20 * (1 - f["solidity"]))
                candidates.append((
                    "star", conf,
                    f"corners_finest={f['corners_finest']}, "
                    f"diff={corner_diff}, solidity={f['solidity']:.2f}",
                ))

        # ---- OPEN-shape candidates ----
        else:
            # Line / arrow: elongated, few direction changes
            if f["direction_changes"] < 5 and f["elongation"] > 2.0:
                conf = min(0.92, 0.45 + 0.06 * f["elongation"])
                candidates.append((
                    "line", conf,
                    f"elongation={f['elongation']:.1f}, dir_changes={f['direction_changes']}",
                ))

            # Zigzag: many direction changes
            if f["direction_changes"] >= 6:
                conf = min(0.88, 0.40 + 0.04 * f["direction_changes"])
                candidates.append((
                    "star", conf,
                    f"zigzag, dir_changes={f['direction_changes']}",
                ))

        if not candidates:
            return RecognitionResult(
                "unknown", 0.30,
                f"No strong match (circ={f['circularity']:.2f}, "
                f"corners={f['corners_loose']}, closed={is_closed}, "
                f"solidity={f['solidity']:.2f})",
            )

        candidates.sort(key=lambda c: c[1], reverse=True)
        label, conf, explanation = candidates[0]
        return RecognitionResult(label, round(conf, 2), explanation)


# --- Future ONNX / OpenVINO classifier stub ---
# class OpenVINOShapeRecognizer(ShapeRecognizerBase):
#     def __init__(self, model_path: str, device: str = "CPU"):
#         from openvino.runtime import Core
#         self._core = Core()
#         self._model = self._core.compile_model(model_path, device)
#         self._labels = ["circle", "triangle", "rectangle", "star", "line", "heart", "unknown"]
#
#     def recognize(self, points: List[Tuple[float, float]]) -> RecognitionResult:
#         canvas = self._rasterize(points)
#         result = self._model([canvas])
#         probs = softmax(result[0])
#         idx = int(np.argmax(probs))
#         return RecognitionResult(self._labels[idx], float(probs[idx]), "OpenVINO classifier")
#
#     def _rasterize(self, points, size=64):
#         # Draw points on a 64x64 canvas, normalize, return tensor
#         ...
