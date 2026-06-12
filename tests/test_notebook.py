"""Tests for the multi-stroke Notebook + LocalAnalyzer flow."""
import math
import time

import numpy as np
import pytest

from airsketch.diagram_analyzer import LocalAnalyzer
from airsketch.notebook import Notebook
from airsketch.shape_recognizer import ShapeRecognizer
from airsketch.stroke import DiagramStatus


@pytest.fixture
def nb():
    return Notebook(
        analyzer=LocalAnalyzer(),
        recognizer=ShapeRecognizer(),
        pause_seconds=10.0,            # disable auto-timeout in tests
        canvas_render_size=300,
        thumbnail_size=80,
        tail_trim=4,
        live_snap_enabled=True,
    )


def _circle(cx=150, cy=150, r=50, n=40):
    return [(cx + r * math.cos(t), cy + r * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, n)]


def _line(n=20):
    return [(20.0 + i * 5, 150.0) for i in range(n)]


def _drive_stroke(nb_, points):
    nb_.begin_stroke(points[0])
    for p in points[1:]:
        nb_.append_to_stroke(p)
    nb_.end_stroke()


class TestStrokes:
    def test_begin_end_records_stroke(self, nb):
        _drive_stroke(nb, _line())
        assert len(nb.current.strokes) == 1

    def test_tail_trim_drops_last_points(self):
        n = Notebook(analyzer=LocalAnalyzer(), recognizer=ShapeRecognizer(),
                     tail_trim=4, live_snap_enabled=False)
        _drive_stroke(n, _line(20))
        # snap is off, so we keep the trimmed raw points
        assert len(n.current.strokes[0].points) == 16

    def test_no_tail_trim(self):
        n = Notebook(analyzer=LocalAnalyzer(), recognizer=ShapeRecognizer(),
                     tail_trim=0, live_snap_enabled=False)
        _drive_stroke(n, _line(20))
        assert len(n.current.strokes[0].points) == 20


class TestLiveSnap:
    def test_circle_snaps_to_circle(self, nb):
        _drive_stroke(nb, _circle())
        s = nb.current.strokes[0]
        assert s.snapped_to == "circle"
        assert s.snap_confidence > 0.7

    def test_line_snaps_to_line(self, nb):
        _drive_stroke(nb, _line(25))
        s = nb.current.strokes[0]
        assert s.snapped_to == "line"


class TestDiagram:
    def test_finalize_runs_local_analysis(self, nb):
        _drive_stroke(nb, _circle())
        d = nb.finalize_current_diagram()
        # Wait for async analysis
        time.sleep(0.3)
        assert d.status == DiagramStatus.DONE
        assert d.analysis is not None
        assert "circle" in d.analysis.title.lower()

    def test_diagram_count(self, nb):
        _drive_stroke(nb, _circle())
        nb.finalize_current_diagram()
        _drive_stroke(nb, _line(25))
        nb.finalize_current_diagram()
        assert nb.diagram_count == 2

    def test_empty_finalize_returns_none(self, nb):
        result = nb.finalize_current_diagram()
        assert result is None

    def test_canvas_aliased_to_clean_canvas(self, nb):
        """The 'second convert' was removed; clean_canvas IS canvas."""
        _drive_stroke(nb, _circle())
        d = nb.finalize_current_diagram()
        assert d.clean_canvas is d.canvas
        assert d.clean_thumbnail is d.thumbnail
