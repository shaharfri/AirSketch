"""Data models for strokes, diagrams, and VLM analyses."""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


class DiagramStatus:
    DRAWING = "drawing"
    PENDING = "pending"       # finalized, queued
    ANALYZING = "analyzing"   # analyzer running
    DONE = "done"
    FAILED = "failed"


@dataclass
class Stroke:
    """One continuous pen-down trajectory."""
    points: List[Tuple[float, float]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    # Set when live snap-to-shape replaced points with a clean primitive
    snapped_to: Optional[str] = None
    snap_confidence: float = 0.0
    raw_points: List[Tuple[float, float]] = field(default_factory=list)

    def add_point(self, p: Tuple[float, float]) -> None:
        self.points.append(p)

    def finalize(self) -> None:
        self.ended_at = time.time()

    @property
    def is_finalized(self) -> bool:
        return self.ended_at is not None

    @property
    def point_count(self) -> int:
        return len(self.points)


@dataclass
class DiagramAnalysis:
    """Result returned by a DiagramAnalyzer."""
    title: str = "Untitled"
    description: str = ""
    topic: str = "sketch"
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""
    analyzer_name: str = ""


@dataclass
class Diagram:
    """A collection of strokes that together form one diagram."""
    strokes: List[Stroke] = field(default_factory=list)
    status: str = DiagramStatus.DRAWING
    analysis: Optional[DiagramAnalysis] = None
    error: Optional[str] = None
    canvas: Optional[np.ndarray] = None        # raw render for VLM (white bg)
    thumbnail: Optional[np.ndarray] = None     # small raw render for sidebar
    clean_canvas: Optional[np.ndarray] = None  # beautified render (Layer A)
    clean_thumbnail: Optional[np.ndarray] = None
    primitives: list = field(default_factory=list)  # list[Primitive]
    created_at: float = field(default_factory=time.time)
    finalized_at: Optional[float] = None
    shapes_detected: list = field(default_factory=list)
    future: Optional[object] = None            # concurrent.futures.Future

    def add_stroke(self, stroke: Stroke) -> None:
        self.strokes.append(stroke)

    @property
    def is_empty(self) -> bool:
        return len(self.strokes) == 0

    @property
    def total_points(self) -> int:
        return sum(len(s.points) for s in self.strokes)

    def get_all_points(self) -> List[Tuple[float, float]]:
        """Flatten all stroke points into a single list."""
        pts: List[Tuple[float, float]] = []
        for s in self.strokes:
            pts.extend(s.points)
        return pts

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, width, height) over all stroke points."""
        all_pts = self.get_all_points()
        if not all_pts:
            return 0.0, 0.0, 0.0, 0.0
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
