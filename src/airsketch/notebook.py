"""Notebook session manager.

Owns the list of diagrams, the currently-drawing stroke, the diagram-segmentation
timer, and the async ThreadPoolExecutor used for VLM inference.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np

from airsketch.beautifier import points_for_primitive
from airsketch.diagram_analyzer import DiagramAnalyzer
from airsketch.primitive_classifier import Primitive, PrimitiveClassifier, PrimitiveKind
from airsketch.shape_recognizer import ShapeRecognizer
from airsketch.stroke import Diagram, DiagramAnalysis, DiagramStatus, Stroke


class Notebook:
    """Multi-diagram drawing session with async analysis."""

    def __init__(
        self,
        analyzer: DiagramAnalyzer,
        recognizer: ShapeRecognizer,
        pause_seconds: float = 3.0,
        canvas_render_size: int = 512,
        thumbnail_size: int = 140,
        tail_trim: int = 4,
        live_snap_enabled: bool = True,
        live_snap_min_confidence: float = 0.78,
        live_snap_arrow_min_confidence: float = 0.85,
    ):
        self.diagrams: List[Diagram] = []
        self.current: Diagram = Diagram()
        self.current_stroke: Optional[Stroke] = None

        self._analyzer = analyzer
        self._recognizer = recognizer
        self._primitive_cls = PrimitiveClassifier()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")
        self._pause_seconds = pause_seconds
        self._canvas_render_size = canvas_render_size
        self._thumbnail_size = thumbnail_size

        self._tail_trim = tail_trim
        self._live_snap_enabled = live_snap_enabled
        self._live_snap_min_confidence = live_snap_min_confidence
        self._live_snap_arrow_min_confidence = live_snap_arrow_min_confidence

        self._last_activity = time.time()

    # ----- Stroke lifecycle -----

    def begin_stroke(self, point: Tuple[float, float]) -> None:
        self.current_stroke = Stroke()
        self.current_stroke.add_point(point)
        self._last_activity = time.time()

    def append_to_stroke(self, point: Tuple[float, float]) -> None:
        if self.current_stroke is None:
            self.begin_stroke(point)
            return
        self.current_stroke.add_point(point)
        self._last_activity = time.time()

    def end_stroke(self) -> None:
        if self.current_stroke is None:
            return

        stroke = self.current_stroke
        self.current_stroke = None
        self._last_activity = time.time()

        if len(stroke.points) < 2:
            return

        stroke.finalize()

        # Live snap-to-shape: classify (trying both trimmed and full versions)
        # and replace raw points with a clean primitive when confident enough.
        # The trim handles finger-curling artifacts; the "try both" lets a real
        # arrow keep its arrowhead.
        if self._live_snap_enabled:
            self._classify_and_snap(stroke)
        elif self._tail_trim > 0 and len(stroke.points) > self._tail_trim + 4:
            # Snap disabled — still do a simple trim
            stroke.points = stroke.points[: -self._tail_trim]

        self.current.add_stroke(stroke)

    def _classify_and_snap(self, stroke: Stroke) -> None:
        """Classify the just-completed stroke against both the full path and a
        tail-trimmed version. Snap to whichever interpretation is most confident,
        applying per-primitive thresholds."""
        full_pts = list(stroke.points)

        candidates: list[tuple[Primitive, list]] = []
        try:
            candidates.append((self._primitive_cls.classify(full_pts), full_pts))
        except Exception:
            return

        # Trimmed version handles fist-closing curls. Skip if it would
        # leave too little stroke.
        if self._tail_trim > 0 and len(full_pts) > self._tail_trim + 4:
            trimmed_pts = full_pts[: -self._tail_trim]
            try:
                candidates.append((self._primitive_cls.classify(trimmed_pts), trimmed_pts))
            except Exception:
                pass

        # Pick the best interpretation. Bias arrow slightly: if any candidate
        # found a confident arrow, prefer it (don't let the trimmed version
        # downgrade a real arrow to a line by chopping off the head).
        arrow_candidates = [
            (p, pts) for p, pts in candidates
            if p.kind == PrimitiveKind.ARROW
            and p.confidence >= self._live_snap_arrow_min_confidence
        ]
        if arrow_candidates:
            best_prim, best_pts = max(arrow_candidates, key=lambda c: c[0].confidence)
        else:
            # No confident arrow → pick the highest-confidence non-arrow candidate.
            # If confidences tie, prefer the trimmed one (last in list) — it's cleaner.
            n = len(candidates)
            best_prim, best_pts = max(
                enumerate(candidates),
                key=lambda ic: (ic[1][0].confidence, ic[0]),
            )[1]

        # Apply threshold based on what we're snapping to
        if best_prim.kind == PrimitiveKind.ARROW:
            threshold = self._live_snap_arrow_min_confidence
        elif best_prim.kind in (PrimitiveKind.CURVE, PrimitiveKind.DOT):
            # Never snap curves/dots — keep the trimmed raw stroke
            if best_pts is not full_pts:
                stroke.points = best_pts
            return
        else:
            threshold = self._live_snap_min_confidence

        if best_prim.confidence < threshold:
            # Don't snap, but still use the trimmed version if it's cleaner
            if best_pts is not full_pts:
                stroke.points = best_pts
            return

        clean_pts = points_for_primitive(best_prim, density=60)
        if len(clean_pts) < 2:
            return

        stroke.raw_points = full_pts
        stroke.points = clean_pts
        stroke.snapped_to = best_prim.kind
        stroke.snap_confidence = best_prim.confidence

    # ----- Diagram lifecycle -----

    def check_pause_timeout(self) -> bool:
        """Auto-finalize current diagram if it's been idle long enough. Returns True if finalized."""
        if self.current.is_empty:
            return False
        if self.current_stroke is not None:
            return False
        if (time.time() - self._last_activity) > self._pause_seconds:
            self.finalize_current_diagram()
            return True
        return False

    def finalize_current_diagram(self) -> Optional[Diagram]:
        """Finalize the current diagram and queue VLM analysis.

        No "second convert" — the strokes are already live-snapped at pen-up,
        so the rendered canvas IS the clean version. We render once into
        `canvas`/`thumbnail` and alias `clean_canvas`/`clean_thumbnail` to them.
        """
        if self.current.is_empty:
            return None

        diagram = self.current
        diagram.finalized_at = time.time()

        # Single render of the live-snapped strokes. This is what the user
        # saw on screen, just framed into a square canvas for sidebar/export.
        diagram.canvas = self._render_canvas(
            diagram, self._canvas_render_size, white_bg=True
        )
        diagram.thumbnail = self._render_canvas(
            diagram, self._thumbnail_size, white_bg=True
        )
        # No second processing pass — alias clean_* to the same render.
        diagram.clean_canvas = diagram.canvas
        diagram.clean_thumbnail = diagram.thumbnail

        # Extract primitives directly from the snapped strokes (no
        # re-classification). For free-form / unsnapped strokes we still
        # run the classifier so we have *something* to feed the analyzer.
        diagram.primitives = self._extract_primitives(diagram)

        diagram.shapes_detected = self._detect_shapes(diagram)
        diagram.status = DiagramStatus.PENDING
        diagram.future = self._executor.submit(self._run_analysis, diagram)

        self.diagrams.append(diagram)
        self.current = Diagram()
        self._last_activity = time.time()
        return diagram

    def _extract_primitives(self, diagram: Diagram) -> List[Primitive]:
        """Build the primitive list from each stroke's live-snapped kind.

        Strokes that snapped during drawing are returned with their recorded
        kind (no re-classification). Unsnapped strokes are classified now.
        """
        out: List[Primitive] = []
        for stroke in diagram.strokes:
            if len(stroke.points) < 2:
                continue
            snapped_kind = getattr(stroke, "snapped_to", None)
            snap_conf = getattr(stroke, "snap_confidence", 0.0)
            if snapped_kind:
                xs = [p[0] for p in stroke.points]
                ys = [p[1] for p in stroke.points]
                bb = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                out.append(Primitive(
                    kind=snapped_kind,
                    params={},
                    confidence=snap_conf,
                    bbox=bb,
                ))
            else:
                out.append(self._primitive_cls.classify(stroke.points))
        return out

    def _run_analysis(self, diagram: Diagram) -> DiagramAnalysis | None:
        diagram.status = DiagramStatus.ANALYZING
        try:
            # Prefer the clean canvas — easier for the VLM to interpret
            img = diagram.clean_canvas if diagram.clean_canvas is not None else diagram.canvas
            analysis = self._analyzer.analyze(
                img, diagram.shapes_detected, primitives=diagram.primitives,
            )
            diagram.analysis = analysis
            diagram.status = DiagramStatus.DONE
            return analysis
        except Exception as e:
            diagram.error = f"{type(e).__name__}: {e}"
            diagram.status = DiagramStatus.FAILED
            return None

    def reanalyze_last(self) -> bool:
        """Re-run the analyzer on the most recent finalized diagram."""
        if not self.diagrams:
            return False
        d = self.diagrams[-1]
        d.status = DiagramStatus.PENDING
        d.analysis = None
        d.error = None
        d.future = self._executor.submit(self._run_analysis, d)
        return True

    # ----- Shape detection over a full diagram -----

    def _detect_shapes(self, diagram: Diagram) -> list:
        """Run the geometric recognizer on each stroke (per-stroke for now)."""
        results = []
        for stroke in diagram.strokes:
            if len(stroke.points) >= self._recognizer.MIN_POINTS:
                r = self._recognizer.recognize(stroke.points)
                results.append(r)
        return results

    # ----- Rendering -----

    def _render_canvas(
        self, diagram: Diagram, size: int, white_bg: bool = True
    ) -> np.ndarray:
        """Render all strokes onto a fixed-size square canvas."""
        canvas = np.full(
            (size, size, 3),
            255 if white_bg else 24,
            dtype=np.uint8,
        )

        all_pts = diagram.get_all_points()
        if not all_pts:
            return canvas

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)

        margin = size * 0.08
        usable = size - 2 * margin
        scale = usable / max(w, h)
        # Center
        offset_x = margin + (usable - w * scale) / 2 - x_min * scale
        offset_y = margin + (usable - h * scale) / 2 - y_min * scale

        line_color = (40, 40, 40) if white_bg else (0, 255, 200)
        thickness = max(2, int(size / 220))

        for stroke in diagram.strokes:
            if len(stroke.points) < 2:
                continue
            pts = np.array(
                [
                    (int(p[0] * scale + offset_x), int(p[1] * scale + offset_y))
                    for p in stroke.points
                ],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, line_color, thickness, cv2.LINE_AA)

        return canvas

    # ----- Edit helpers -----

    def clear_current(self) -> None:
        self.current = Diagram()
        self.current_stroke = None
        self._last_activity = time.time()

    def undo_last_stroke(self) -> bool:
        if self.current.strokes:
            self.current.strokes.pop()
            self._last_activity = time.time()
            return True
        return False

    # ----- Convenience -----

    @property
    def diagram_count(self) -> int:
        return len(self.diagrams)

    @property
    def has_pending_analysis(self) -> bool:
        return any(
            d.status in (DiagramStatus.PENDING, DiagramStatus.ANALYZING)
            for d in self.diagrams
        )

    @property
    def is_drawing(self) -> bool:
        return self.current_stroke is not None

    def shutdown(self, wait: bool = True) -> None:
        try:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
        except TypeError:
            self._executor.shutdown(wait=wait)
