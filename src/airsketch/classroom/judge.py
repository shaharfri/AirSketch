"""Judge a student's drawing against a challenge target.

Geometry targets are judged by the per-stroke primitive classifier.
Object targets (house, cat, ...) are judged by the Quick-Draw CNN.
Returns a 0-100 accuracy score and a 0-3 star rating.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import cv2
import numpy as np

from airsketch.classroom.challenge_engine import Challenge, ChallengeResult, OBJECT_TARGETS
from airsketch.primitive_classifier import PrimitiveClassifier, PrimitiveKind
from airsketch.stroke import Diagram


# Targets that are considered equivalent when matching.
_SYNONYMS = {
    "square": {"square", "rectangle"},
    "rectangle": {"rectangle", "square"},
    "circle": {"circle", "ellipse"},
    "ellipse": {"ellipse", "circle"},
}


def _matches(detected: str, target: str) -> bool:
    if detected == target:
        return True
    return detected in _SYNONYMS.get(target, {target})


def _stars_from_score(score: int) -> int:
    if score >= 85:
        return 3
    if score >= 70:
        return 2
    if score >= 55:
        return 1
    return 0


def render_for_cnn(diagram: Diagram, size: int = 256, thickness: int = 8) -> np.ndarray:
    """Render the diagram as bright strokes on a black background (Quick-Draw format)."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    all_pts = diagram.get_all_points()
    if not all_pts:
        return canvas
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, y_min = min(xs), min(ys)
    w = max(max(xs) - x_min, 1.0)
    h = max(max(ys) - y_min, 1.0)
    margin = size * 0.12
    usable = size - 2 * margin
    scale = usable / max(w, h)
    off_x = margin + (usable - w * scale) / 2 - x_min * scale
    off_y = margin + (usable - h * scale) / 2 - y_min * scale
    for stroke in diagram.strokes:
        if len(stroke.points) < 2:
            continue
        pts = np.array(
            [(int(p[0] * scale + off_x), int(p[1] * scale + off_y)) for p in stroke.points],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


class Judge:
    """Scores a finalized diagram against a challenge target."""

    def __init__(self, sketch_classifier=None):
        self._cnn = sketch_classifier
        self._prim = PrimitiveClassifier()

    def judge(self, diagram: Diagram, challenge: Challenge) -> ChallengeResult:
        if challenge.theme == "objects":
            return self._judge_object(diagram, challenge)
        return self._judge_geometry(diagram, challenge)

    # ---- Geometry: primitive classifier ----

    def _judge_geometry(self, diagram: Diagram, ch: Challenge) -> ChallengeResult:
        if not diagram.strokes:
            return self._empty_result(ch)

        # Classify each stroke; find the best primitive that matches the target.
        best_conf = 0.0
        best_label = "nothing"
        detected_counts: Counter = Counter()
        for stroke in diagram.strokes:
            if len(stroke.points) < 2:
                continue
            # Prefer the live-snapped kind if present
            kind = getattr(stroke, "snapped_to", None)
            conf = getattr(stroke, "snap_confidence", 0.0)
            if not kind:
                prim = self._prim.classify(stroke.points)
                kind, conf = prim.kind, prim.confidence
            detected_counts[kind] += 1
            if _matches(kind, ch.target) and conf > best_conf:
                best_conf = conf
                best_label = kind

        if best_label != "nothing" and _matches(best_label, ch.target):
            score = int(round(min(1.0, best_conf) * 100))
            return ChallengeResult(
                challenge=ch, detected=best_label, score=score,
                stars=_stars_from_score(score), passed=score >= 55,
                explanation=f"matched {best_label} @ {best_conf:.0%}",
            )

        # No match — report what we saw
        detected = detected_counts.most_common(1)[0][0] if detected_counts else "nothing"
        return ChallengeResult(
            challenge=ch, detected=detected, score=0, stars=0, passed=False,
            explanation=f"expected {ch.target}, saw {detected}",
        )

    # ---- Objects: CNN classifier ----

    def _judge_object(self, diagram: Diagram, ch: Challenge) -> ChallengeResult:
        if self._cnn is None:
            return ChallengeResult(
                challenge=ch, detected="(no CNN)", score=0, stars=0, passed=False,
                explanation="CNN classifier not loaded — train it to judge objects",
            )
        if not diagram.strokes:
            return self._empty_result(ch)

        canvas = render_for_cnn(diagram)
        try:
            label, conf = self._cnn.classify(canvas)
        except Exception as e:
            return ChallengeResult(
                challenge=ch, detected="(error)", score=0, stars=0, passed=False,
                explanation=f"CNN error: {type(e).__name__}",
            )

        if _matches(label, ch.target):
            score = int(round(min(1.0, conf) * 100))
            return ChallengeResult(
                challenge=ch, detected=label, score=score,
                stars=_stars_from_score(score), passed=score >= 55,
                explanation=f"CNN matched {label} @ {conf:.0%}",
            )
        return ChallengeResult(
            challenge=ch, detected=label, score=0, stars=0, passed=False,
            explanation=f"expected {ch.target}, CNN saw {label} @ {conf:.0%}",
        )

    @staticmethod
    def _empty_result(ch: Challenge) -> ChallengeResult:
        return ChallengeResult(
            challenge=ch, detected="nothing", score=0, stars=0, passed=False,
            explanation="no drawing detected",
        )
