"""Classify a single stroke into a geometric primitive + fit parameters.

Used by the beautifier to replace messy strokes with mathematically clean
versions (line, arrow, circle, ellipse, rectangle, triangle, polygon, curve).
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


class PrimitiveKind:
    LINE = "line"
    ARROW = "arrow"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    POLYGON = "polygon"
    CURVE = "curve"
    DOT = "dot"


@dataclass
class Primitive:
    """A clean geometric primitive fitted to a single stroke."""
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


# --- Helpers ---------------------------------------------------------------

def _bbox(pts: np.ndarray) -> Tuple[float, float, float, float]:
    if len(pts) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
    x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def _line_rms_deviation(pts: np.ndarray) -> float:
    """RMS perpendicular distance from points to the best-fit line."""
    if len(pts) < 2:
        return 0.0
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    nx, ny = -vy, vx  # normal vector
    deviations = np.abs((pts[:, 0] - x0) * nx + (pts[:, 1] - y0) * ny)
    return float(np.sqrt(np.mean(deviations ** 2)))


def _ellipse_residual(pts: np.ndarray, ellipse) -> float:
    """RMS distance from contour points to an ellipse outline."""
    (cx, cy), (a, b), theta = ellipse
    if a < 1 or b < 1:
        return 1e9
    theta_rad = math.radians(theta)
    cos_t, sin_t = math.cos(-theta_rad), math.sin(-theta_rad)
    rx, ry = a / 2.0, b / 2.0
    # Transform points to ellipse-aligned coordinates
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    ex = dx * cos_t - dy * sin_t
    ey = dx * sin_t + dy * cos_t
    # Normalize to unit circle space
    norm = np.sqrt((ex / rx) ** 2 + (ey / ry) ** 2)
    # Deviation from unit-distance in that space, scaled back to pixels
    residual = (norm - 1.0) * ((rx + ry) / 2.0)
    return float(np.sqrt(np.mean(residual ** 2)))


def _angle_at(p_prev: Point, p: Point, p_next: Point) -> float:
    v1x, v1y = p_prev[0] - p[0], p_prev[1] - p[1]
    v2x, v2y = p_next[0] - p[0], p_next[1] - p[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-3 or n2 < 1e-3:
        return 180.0
    c = (v1x * v2x + v1y * v2y) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _detect_arrowhead(pts: np.ndarray) -> bool:
    """Legacy: detect arrowhead in the last ~15% of stroke (kept for tests)."""
    return _find_arrow_shaft_end(pts, scale=1.0) >= 0


def _find_arrow_shaft_end(pts: np.ndarray, scale: float) -> int:
    """Find the index where the shaft ends and the arrowhead begins.

    An arrow has two parts:
      - shaft: a relatively straight leading segment
      - tail (arrowhead): at the end, with a sharp direction reversal that
        folds back toward / past the shaft direction

    Returns the shaft-end index, or -1 if no arrow structure detected.
    """
    n = len(pts)
    if n < 10:
        return -1

    # Reasonable shaft length fractions to try (longest first)
    best_split = -1
    best_quality = 0.0

    for shaft_frac in (0.90, 0.85, 0.80, 0.75, 0.70, 0.65):
        split = max(5, int(n * shaft_frac))
        if n - split < 3:
            continue

        shaft = pts[:split]
        tail = pts[split:]

        # 1) Shaft must fit a line tightly
        shaft_rms = _line_rms_deviation(shaft)
        if shaft_rms > 0.06 * max(scale, 1.0):
            continue

        # 2) Tail must contain AT LEAST TWO significant direction reversals
        # (a real arrowhead is a V — two sides; a single curl from your finger
        # curling at pen-up is just ONE direction change, so that's filtered out).
        sharp_turns = 0
        for i in range(1, len(tail) - 1):
            a = _angle_at(tail[i - 1], tail[i], tail[i + 1])
            if a < 90.0:   # interior angle < 90° = sharp turn
                sharp_turns += 1
        if sharp_turns < 2:
            continue

        # 3) Tail must fold back near the shaft end (arrowhead is small)
        shaft_end_pt = shaft[-1]
        tail_max_dist = float(
            np.max(np.linalg.norm(tail - shaft_end_pt, axis=1))
        )
        shaft_length = float(np.linalg.norm(shaft[-1] - shaft[0]))
        if shaft_length < 1e-3:
            continue
        tail_ratio = tail_max_dist / shaft_length
        # Arrowhead size: between 3% and 50% of shaft length is reasonable
        if tail_ratio < 0.03 or tail_ratio > 0.55:
            continue

        # 4) Quality score: prefer cleaner shaft and clear arrowhead structure
        shaft_quality = 1.0 - shaft_rms / (0.06 * max(scale, 1.0))
        quality = shaft_quality + 0.30 * sharp_turns
        if quality > best_quality:
            best_quality = quality
            best_split = split

    return best_split


# --- Main classifier -------------------------------------------------------

class PrimitiveClassifier:
    """Classify a stroke as one of several geometric primitives."""

    MIN_POINTS = 4
    DOT_MAX_BBOX = 8.0

    # Tolerances (as a fraction of stroke "scale" = max(bbox.w, bbox.h))
    LINE_RMS_REL = 0.05
    ELLIPSE_RMS_REL = 0.08
    CLOSURE_REL = 0.20

    def classify(self, points: List[Point]) -> Primitive:
        if not points:
            return Primitive(PrimitiveKind.DOT, {}, 0.0, (0, 0, 0, 0))

        pts = np.array(points, dtype=np.float32)
        bb = _bbox(pts)
        x, y, w, h = bb
        scale = max(w, h)

        # Degenerate cases
        if scale < self.DOT_MAX_BBOX or len(pts) < self.MIN_POINTS:
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            return Primitive(
                PrimitiveKind.DOT,
                {"center": (cx, cy), "radius": max(2.0, scale / 2)},
                confidence=0.7, bbox=bb,
            )

        closure_dist = float(np.linalg.norm(pts[0] - pts[-1]))
        perim_open = cv2.arcLength(pts.reshape(-1, 1, 2), False)
        closed = closure_dist < self.CLOSURE_REL * max(perim_open, 1.0)

        # --- Open primitives: arrow, line, curve ---
        if not closed:
            # 1) Try arrow FIRST: shaft-fit + arrowhead test.
            #    The whole-path line RMS would fail an arrow because the
            #    arrowhead deviates — so we check shaft-only.
            shaft_end_idx = _find_arrow_shaft_end(pts, scale)
            if shaft_end_idx > 0:
                start = (float(pts[0, 0]), float(pts[0, 1]))
                # End the arrow line at the shaft endpoint (cleaner render than tip)
                end = (float(pts[shaft_end_idx, 0]), float(pts[shaft_end_idx, 1]))
                return Primitive(
                    PrimitiveKind.ARROW,
                    {"start": start, "end": end},
                    confidence=0.85, bbox=bb,
                )

            # 2) Plain line
            line_rms = _line_rms_deviation(pts)
            if line_rms < self.LINE_RMS_REL * scale and len(pts) >= 4:
                start = (float(pts[0, 0]), float(pts[0, 1]))
                end = (float(pts[-1, 0]), float(pts[-1, 1]))
                conf = 0.85 - line_rms / max(scale, 1.0)
                return Primitive(
                    PrimitiveKind.LINE,
                    {"start": start, "end": end},
                    confidence=max(0.5, conf), bbox=bb,
                )

            # 3) Freeform curve
            return Primitive(
                PrimitiveKind.CURVE,
                {"points": [(float(p[0]), float(p[1])) for p in pts]},
                confidence=0.40, bbox=bb,
            )

        # --- Closed primitives: circle/ellipse, triangle, rectangle, polygon ---
        contour = pts.reshape(-1, 1, 2).astype(np.int32)
        perim_closed = cv2.arcLength(contour, True)

        # Try ellipse
        ellipse_info = None
        ellipse_residual = 1e9
        if len(pts) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                ellipse_residual = _ellipse_residual(pts, ellipse)
                if ellipse_residual < self.ELLIPSE_RMS_REL * scale:
                    (cx, cy), (a, b), theta = ellipse
                    aspect = max(a, b) / max(min(a, b), 1e-6)
                    ellipse_info = (cx, cy, a, b, theta, aspect)
            except cv2.error:
                pass

        # Polygon fit
        approx = cv2.approxPolyDP(contour, 0.04 * perim_closed, True)
        approx_pts = approx.reshape(-1, 2).astype(float).tolist()
        n_corners = len(approx_pts)

        # Pick the best match.
        # Rule of priority:
        #   - If ellipse residual is VERY small (< STRONG_ELLIPSE_REL), the
        #     stroke is unambiguously a smooth curve → ellipse/circle.
        #   - Otherwise if n_corners is exactly 3 or 4 the stroke has sharp
        #     corners → triangle/rectangle.
        #   - Otherwise fall back to ellipse / polygon / curve.
        candidates: List[Tuple[float, Primitive]] = []

        STRONG_ELLIPSE_REL = 0.03  # very tight ellipse fit
        unambiguous_ellipse = (
            ellipse_info is not None
            and ellipse_residual < STRONG_ELLIPSE_REL * scale
        )
        sharp_polygon = n_corners in (3, 4) and not unambiguous_ellipse

        if ellipse_info is not None and not sharp_polygon:
            cx, cy, a, b, theta, aspect = ellipse_info
            kind = PrimitiveKind.CIRCLE if aspect < 1.20 else PrimitiveKind.ELLIPSE
            # Give a very tight fit a confidence boost so it wins over the
            # ambiguous polygon vote.
            if unambiguous_ellipse:
                ellipse_conf = 0.96
            else:
                ellipse_conf = 0.88 - ellipse_residual / max(scale, 1.0)
            candidates.append((
                ellipse_conf,
                Primitive(
                    kind,
                    {
                        "center": (float(cx), float(cy)),
                        "axes": (float(a / 2.0), float(b / 2.0)),
                        "angle_deg": float(theta),
                    },
                    confidence=max(0.5, ellipse_conf), bbox=bb,
                ),
            ))

        # Only consider sharp polygons if the ellipse fit was NOT unambiguous.
        if unambiguous_ellipse:
            n_corners_check = -1  # disable polygon branches
        else:
            n_corners_check = n_corners

        if n_corners_check == 3:
            tri_conf = 0.92
            candidates.append((
                tri_conf,
                Primitive(
                    PrimitiveKind.TRIANGLE,
                    {"vertices": [tuple(p) for p in approx_pts]},
                    confidence=tri_conf, bbox=bb,
                ),
            ))
        elif n_corners_check == 4:
            rect = cv2.minAreaRect(contour)
            (rx, ry), (rw, rh), rang = rect
            rect_pts = cv2.boxPoints(rect)
            rect_conf = 0.90
            candidates.append((
                rect_conf,
                Primitive(
                    PrimitiveKind.RECTANGLE,
                    {
                        "vertices": [tuple(map(float, p)) for p in rect_pts],
                        "center": (float(rx), float(ry)),
                        "size": (float(rw), float(rh)),
                        "angle_deg": float(rang),
                    },
                    confidence=rect_conf, bbox=bb,
                ),
            ))
        elif 5 <= n_corners_check <= 10:
            # If the ellipse fit was very tight, this is probably a round shape
            # we should classify as ellipse — fall through to ellipse candidate.
            if ellipse_info is None:
                poly_conf = 0.65
                candidates.append((
                    poly_conf,
                    Primitive(
                        PrimitiveKind.POLYGON,
                        {"vertices": [tuple(p) for p in approx_pts]},
                        confidence=poly_conf, bbox=bb,
                    ),
                ))
            else:
                cx, cy, a, b, theta, aspect = ellipse_info
                kind = PrimitiveKind.CIRCLE if aspect < 1.20 else PrimitiveKind.ELLIPSE
                ellipse_conf = 0.90 - ellipse_residual / max(scale, 1.0)
                candidates.append((
                    ellipse_conf,
                    Primitive(
                        kind,
                        {
                            "center": (float(cx), float(cy)),
                            "axes": (float(a / 2.0), float(b / 2.0)),
                            "angle_deg": float(theta),
                        },
                        confidence=max(0.5, ellipse_conf), bbox=bb,
                    ),
                ))
                poly_conf = 0.55
                candidates.append((
                    poly_conf,
                    Primitive(
                        PrimitiveKind.POLYGON,
                        {"vertices": [tuple(p) for p in approx_pts]},
                        confidence=poly_conf, bbox=bb,
                    ),
                ))
        elif n_corners_check > 10 and ellipse_info is not None:
            # Many polygon corners + good ellipse fit = ellipse/circle
            cx, cy, a, b, theta, aspect = ellipse_info
            kind = PrimitiveKind.CIRCLE if aspect < 1.20 else PrimitiveKind.ELLIPSE
            ellipse_conf = 0.85 - ellipse_residual / max(scale, 1.0)
            candidates.append((
                ellipse_conf,
                Primitive(
                    kind,
                    {
                        "center": (float(cx), float(cy)),
                        "axes": (float(a / 2.0), float(b / 2.0)),
                        "angle_deg": float(theta),
                    },
                    confidence=max(0.5, ellipse_conf), bbox=bb,
                ),
            ))

        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            return candidates[0][1]

        # Fall back to curve
        return Primitive(
            PrimitiveKind.CURVE,
            {"points": [(float(p[0]), float(p[1])) for p in pts]},
            confidence=0.40, bbox=bb,
        )
