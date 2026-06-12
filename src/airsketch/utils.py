import math
from typing import List, Tuple

import cv2
import numpy as np


def smooth_points(
    points: List[Tuple[float, float]], window: int = 5
) -> List[Tuple[float, float]]:
    """Apply moving-average smoothing to a point trajectory."""
    if len(points) < window:
        return list(points)
    pts = np.array(points, dtype=np.float32)
    kernel = np.ones(window) / window
    sx = np.convolve(pts[:, 0], kernel, mode="valid")
    sy = np.convolve(pts[:, 1], kernel, mode="valid")
    return list(zip(sx.tolist(), sy.tolist()))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    pts = np.array(points, dtype=np.float32)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def bounding_box(
    points: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """Return (x_min, y_min, width, height)."""
    pts = np.array(points, dtype=np.float32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)


def draw_neon_line(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    base_thickness: int = 2,
    glow_layers: int = 4,
) -> None:
    """Draw a polyline with a neon glow effect using layered rendering."""
    if len(points) < 2:
        return
    pts_array = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    overlay = frame.copy()
    for i in range(glow_layers, 0, -1):
        alpha = 0.15 + 0.1 * (glow_layers - i)
        thickness = base_thickness + i * 4
        faded = tuple(max(0, int(c * (0.3 + 0.7 * (glow_layers - i) / glow_layers))) for c in color)
        cv2.polylines(overlay, [pts_array], False, faded, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        overlay = frame.copy()
    cv2.polylines(frame, [pts_array], False, color, base_thickness, cv2.LINE_AA)


def draw_text_with_shadow(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """Draw text with a dark shadow for readability.

    Hebrew (or any non-ASCII) text is rendered via Pillow (cv2's Hershey fonts
    can't draw it); ASCII text uses the fast cv2 path unchanged.
    """
    from airsketch import hebrew_text as _ht
    if _ht.has_hebrew(text) and _ht.draw_text(frame, text, pos, scale, color, thickness):
        return
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def text_size(text: str, scale: float, thickness: int = 1) -> Tuple[int, int]:
    """(width, height) of `text` — Hebrew-aware, for centering/layout.

    Mirrors ``cv2.getTextSize(...)[0]`` for ASCII text, and measures via the
    Hebrew renderer for non-ASCII text so RTL labels center correctly.
    """
    from airsketch import hebrew_text as _ht
    if _ht.has_hebrew(text):
        return _ht.measure(text, scale)
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]


def fill_rounded_rect(
    img: np.ndarray, x: int, y: int, w: int, h: int,
    color: Tuple[int, int, int], radius: int = 12,
) -> None:
    """Filled rounded rectangle (cv2 clips out-of-bounds automatically)."""
    r = max(0, min(radius, w // 2, h // 2))
    if r == 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        return
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r),
                   (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def stroke_rounded_rect(
    img: np.ndarray, x: int, y: int, w: int, h: int,
    color: Tuple[int, int, int], thickness: int = 1, radius: int = 12,
) -> None:
    """Outline of a rounded rectangle."""
    r = max(0, min(radius, w // 2, h // 2))
    if r == 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
        return
    cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    alpha: float = 0.55,
    bg: Tuple[int, int, int] = (22, 24, 32),
    border: Tuple[int, int, int] = (70, 76, 92),
    radius: int = 14,
    accent: Tuple[int, int, int] | None = None,
) -> None:
    """Draw a semi-transparent rounded panel with a subtle border.

    Blends only the panel's bounding region (cheaper than a full-frame blend) and
    is bounds-safe (clamps the ROI), so it can never index out of the frame.
    Pass `accent` to add a thin colored bar down the left edge.
    """
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame[y0:y1, x0:x1].copy()
    fill_rounded_rect(sub, x - x0, y - y0, w, h, bg, radius)
    cv2.addWeighted(sub, alpha, frame[y0:y1, x0:x1], 1 - alpha, 0, frame[y0:y1, x0:x1])
    stroke_rounded_rect(frame, x, y, w, h, border, 1, radius)
    if accent is not None:
        ar = max(0, min(radius, h // 2))
        cv2.line(frame, (x + 2, y + ar), (x + 2, y + h - ar), accent, 3, cv2.LINE_AA)


def draw_progress_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    progress: float,
    color: Tuple[int, int, int] = (0, 200, 255),
) -> None:
    """Draw a progress bar (progress in [0, 1])."""
    p = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 70), 1, cv2.LINE_AA)
    fill_w = int(w * p)
    if fill_w > 0:
        cv2.rectangle(frame, (x + 1, y + 1), (x + fill_w - 1, y + h - 1), color, -1, cv2.LINE_AA)


def normalize_points(
    points: List[Tuple[float, float]], canvas_size: int = 200
) -> List[Tuple[float, float]]:
    """Normalize points into a square canvas preserving aspect ratio."""
    if len(points) < 2:
        return list(points)
    pts = np.array(points, dtype=np.float32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min
    scale = max(w, h)
    if scale < 1e-6:
        return list(points)
    margin = canvas_size * 0.1
    effective = canvas_size - 2 * margin
    pts[:, 0] = (pts[:, 0] - x_min) / scale * effective + margin + (effective - w / scale * effective) / 2
    pts[:, 1] = (pts[:, 1] - y_min) / scale * effective + margin + (effective - h / scale * effective) / 2
    return [(float(p[0]), float(p[1])) for p in pts]
