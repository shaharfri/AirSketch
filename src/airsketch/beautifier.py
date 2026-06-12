"""Render a clean canvas from primitive-classified strokes."""
import math
from typing import List, Tuple

import cv2
import numpy as np

from airsketch.primitive_classifier import Primitive, PrimitiveClassifier, PrimitiveKind
from airsketch.stroke import Diagram


# ----------------------------------------------------------------------------
# High-level entry point
# ----------------------------------------------------------------------------

def beautify_diagram(
    diagram: Diagram,
    canvas_size: int = 512,
    white_bg: bool = True,
    classifier: PrimitiveClassifier | None = None,
) -> Tuple[np.ndarray, List[Primitive]]:
    """Render the diagram on a clean canvas.

    Strokes are already live-snapped to clean primitives at pen-up time, so
    we just render their points directly to a centered/scaled canvas. The
    primitive list is still computed (for the LocalAnalyzer's title) — for
    snapped strokes we use the stroke's recorded `snapped_to` kind, for
    free-form strokes we re-classify.
    """
    cls = classifier or PrimitiveClassifier()
    primitives: List[Primitive] = []
    for stroke in diagram.strokes:
        if len(stroke.points) < 2:
            continue
        snapped_kind = getattr(stroke, "snapped_to", None)
        snap_conf = getattr(stroke, "snap_confidence", 0.0)
        if snapped_kind:
            x_min = min(p[0] for p in stroke.points)
            y_min = min(p[1] for p in stroke.points)
            x_max = max(p[0] for p in stroke.points)
            y_max = max(p[1] for p in stroke.points)
            primitives.append(Primitive(
                kind=snapped_kind,
                params={},
                confidence=snap_conf,
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
            ))
        else:
            primitives.append(cls.classify(stroke.points))

    canvas = _make_canvas(canvas_size, white_bg)

    all_pts = diagram.get_all_points()
    if not all_pts:
        return canvas, primitives

    src_bbox = _bbox(all_pts)
    transform = _make_transform(src_bbox, canvas_size, margin_ratio=0.10)

    # Render each stroke directly from its (already-snapped) points.
    for stroke in diagram.strokes:
        if len(stroke.points) < 2:
            continue
        is_curve = (getattr(stroke, "snapped_to", None) == PrimitiveKind.CURVE) or (
            getattr(stroke, "snapped_to", None) is None
        )
        thickness = CURVE_THICK if is_curve else LINE_THICK
        if is_curve:
            pts_to_draw = _smooth_polyline(list(stroke.points), window=5)
        else:
            pts_to_draw = stroke.points
        pts_arr = np.array(
            [transform["p"](*p) for p in pts_to_draw],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_arr], False, INK, thickness, cv2.LINE_AA)

    return canvas, primitives


# ----------------------------------------------------------------------------
# Coordinate transform
# ----------------------------------------------------------------------------

def _bbox(points: List[Tuple[float, float]]):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _make_transform(
    src_bbox: Tuple[float, float, float, float],
    canvas_size: int,
    margin_ratio: float = 0.08,
):
    """Return a callable (x, y) -> (cx, cy) mapping source bbox to centered canvas."""
    sx, sy, sw, sh = src_bbox
    sw = max(sw, 1.0)
    sh = max(sh, 1.0)
    margin = canvas_size * margin_ratio
    usable = canvas_size - 2 * margin
    scale = usable / max(sw, sh)
    offset_x = margin + (usable - sw * scale) / 2 - sx * scale
    offset_y = margin + (usable - sh * scale) / 2 - sy * scale

    def tx(x: float, y: float) -> Tuple[int, int]:
        return int(x * scale + offset_x), int(y * scale + offset_y)

    def txf(x: float, y: float) -> Tuple[float, float]:
        return x * scale + offset_x, y * scale + offset_y

    return {"scale": scale, "ox": offset_x, "oy": offset_y, "p": tx, "pf": txf}


def _make_canvas(size: int, white_bg: bool) -> np.ndarray:
    val = 250 if white_bg else 24
    canvas = np.full((size, size, 3), val, dtype=np.uint8)
    if white_bg:
        # Subtle paper texture: a single off-white wash
        canvas[:] = (252, 251, 248)
    return canvas


# ----------------------------------------------------------------------------
# Primitive renderers
# ----------------------------------------------------------------------------

INK = (40, 40, 50)
ACCENT = (180, 120, 60)
LINE_THICK = 3
CURVE_THICK = 2


def _draw_primitive(canvas: np.ndarray, prim: Primitive, transform: dict, white_bg: bool) -> None:
    kind = prim.kind
    p = transform["p"]
    pf = transform["pf"]
    scale = transform["scale"]

    if kind == PrimitiveKind.DOT:
        c = prim.params["center"]
        cv2.circle(canvas, p(*c), max(3, int(prim.params.get("radius", 3) * scale)),
                   INK, -1, cv2.LINE_AA)

    elif kind == PrimitiveKind.LINE:
        s = prim.params["start"]
        e = prim.params["end"]
        cv2.line(canvas, p(*s), p(*e), INK, LINE_THICK, cv2.LINE_AA)

    elif kind == PrimitiveKind.ARROW:
        s = prim.params["start"]
        e = prim.params["end"]
        sxy, exy = p(*s), p(*e)
        cv2.arrowedLine(canvas, sxy, exy, INK, LINE_THICK, cv2.LINE_AA, tipLength=0.12)

    elif kind == PrimitiveKind.CIRCLE:
        cx, cy = prim.params["center"]
        ax, ay = prim.params["axes"]
        r = int(((ax + ay) / 2.0) * scale)
        cv2.circle(canvas, p(cx, cy), max(3, r), INK, LINE_THICK, cv2.LINE_AA)

    elif kind == PrimitiveKind.ELLIPSE:
        cx, cy = prim.params["center"]
        ax, ay = prim.params["axes"]
        angle = prim.params["angle_deg"]
        center = p(cx, cy)
        axes = (max(3, int(ax * scale)), max(3, int(ay * scale)))
        cv2.ellipse(canvas, center, axes, angle, 0, 360, INK, LINE_THICK, cv2.LINE_AA)

    elif kind == PrimitiveKind.TRIANGLE:
        verts = prim.params["vertices"]
        pts = np.array([p(*v) for v in verts], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, INK, LINE_THICK, cv2.LINE_AA)

    elif kind == PrimitiveKind.RECTANGLE:
        verts = prim.params["vertices"]
        pts = np.array([p(*v) for v in verts], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, INK, LINE_THICK, cv2.LINE_AA)

    elif kind == PrimitiveKind.POLYGON:
        verts = prim.params["vertices"]
        pts = np.array([p(*v) for v in verts], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, INK, LINE_THICK, cv2.LINE_AA)

    elif kind == PrimitiveKind.CURVE:
        # Smooth the input curve
        raw = prim.params["points"]
        smooth = _smooth_polyline(raw, window=5)
        pts = np.array([p(*v) for v in smooth], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, INK, CURVE_THICK, cv2.LINE_AA)


def points_for_primitive(prim: Primitive, density: int = 60) -> list:
    """Generate a list of (x, y) points that *render* the primitive cleanly.

    Used by the live snap-to-shape: when the user lifts the pen, we replace the
    raw stroke points with these so the polyline renderer draws a clean shape.
    """
    kind = prim.kind
    p = prim.params

    if kind == PrimitiveKind.DOT:
        cx, cy = p.get("center", (0.0, 0.0))
        return [(cx, cy)]

    if kind == PrimitiveKind.LINE:
        sx, sy = p["start"]
        ex, ey = p["end"]
        n = max(2, density // 3)
        return [
            (sx + (ex - sx) * t / (n - 1), sy + (ey - sy) * t / (n - 1))
            for t in range(n)
        ]

    if kind == PrimitiveKind.ARROW:
        sx, sy = p["start"]
        ex, ey = p["end"]
        n_shaft = max(2, density // 2)
        pts = [
            (sx + (ex - sx) * t / (n_shaft - 1), sy + (ey - sy) * t / (n_shaft - 1))
            for t in range(n_shaft)
        ]
        # Add a V arrowhead at the end
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        if length < 1.0:
            return pts
        ux, uy = dx / length, dy / length
        head_len = max(8.0, length * 0.18)
        head_wid = head_len * 0.55
        # back-of-head shifts toward start, then sideways
        backx, backy = ex - ux * head_len, ey - uy * head_len
        nx, ny = -uy, ux  # perpendicular
        side1 = (backx + nx * head_wid, backy + ny * head_wid)
        side2 = (backx - nx * head_wid, backy - ny * head_wid)
        # Draw arrowhead: side1 -> tip -> side2 (a V)
        pts.extend([side1, (ex, ey), side2])
        return pts

    if kind in (PrimitiveKind.CIRCLE, PrimitiveKind.ELLIPSE):
        cx, cy = p["center"]
        ax, ay = p["axes"]
        theta = math.radians(p.get("angle_deg", 0.0))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        n = max(24, density)
        out = []
        for i in range(n + 1):  # +1 to close the loop
            ang = 2 * math.pi * i / n
            ex = ax * math.cos(ang)
            ey = ay * math.sin(ang)
            x = cx + ex * cos_t - ey * sin_t
            y = cy + ex * sin_t + ey * cos_t
            out.append((x, y))
        return out

    if kind in (PrimitiveKind.RECTANGLE, PrimitiveKind.TRIANGLE, PrimitiveKind.POLYGON):
        verts = p["vertices"]
        per_edge = max(4, density // max(1, len(verts)))
        out: list = []
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            for t in range(per_edge):
                f = t / per_edge
                out.append((v1[0] + f * (v2[0] - v1[0]), v1[1] + f * (v2[1] - v1[1])))
        out.append(verts[0])  # close the loop
        return out

    if kind == PrimitiveKind.CURVE:
        # Lightly smooth the raw curve
        return _smooth_polyline(p.get("points", []), window=5)

    return p.get("points", [])


def _smooth_polyline(points, window: int = 5):
    if len(points) < window:
        return list(points)
    arr = np.array(points, dtype=np.float32)
    k = np.ones(window) / window
    sx = np.convolve(arr[:, 0], k, mode="same")
    sy = np.convolve(arr[:, 1], k, mode="same")
    # Fix endpoints (convolution edge effects)
    sx[0] = arr[0, 0]; sx[-1] = arr[-1, 0]
    sy[0] = arr[0, 1]; sy[-1] = arr[-1, 1]
    return list(zip(sx.tolist(), sy.tolist()))
