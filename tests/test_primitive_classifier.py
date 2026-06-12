"""Tests for the per-stroke primitive classifier."""
import math
import random

import numpy as np
import pytest

from airsketch.primitive_classifier import PrimitiveClassifier, PrimitiveKind


@pytest.fixture
def cls():
    return PrimitiveClassifier()


# ---- Helper shape generators ----

def line(n=30, dx=5.0, dy=0.0):
    return [(10.0 + i * dx, 50.0 + i * dy) for i in range(n)]


def circle(cx=100, cy=100, r=50, n=60):
    return [(cx + r * math.cos(t), cy + r * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, n)]


def triangle():
    verts = [(100, 30), (170, 150), (30, 150), (100, 30)]
    pts = []
    for i in range(3):
        for j in range(20):
            f = j / 20
            pts.append((verts[i][0] + f * (verts[i + 1][0] - verts[i][0]),
                        verts[i][1] + f * (verts[i + 1][1] - verts[i][1])))
    return pts


def rectangle():
    corners = [(50, 50), (150, 50), (150, 120), (50, 120), (50, 50)]
    pts = []
    for i in range(4):
        for j in range(20):
            f = j / 20
            pts.append((corners[i][0] + f * (corners[i + 1][0] - corners[i][0]),
                        corners[i][1] + f * (corners[i + 1][1] - corners[i][1])))
    return pts


def arrow():
    p = line()
    tip = p[-1]
    p += [(tip[0] - 15, tip[1] - 10), (tip[0], tip[1]), (tip[0] - 15, tip[1] + 10)]
    return p


# ---- Tests ----

class TestPrimitiveClassifier:
    def test_line(self, cls):
        assert cls.classify(line()).kind == PrimitiveKind.LINE

    def test_arrow(self, cls):
        assert cls.classify(arrow()).kind == PrimitiveKind.ARROW

    def test_circle(self, cls):
        assert cls.classify(circle()).kind == PrimitiveKind.CIRCLE

    def test_triangle(self, cls):
        assert cls.classify(triangle()).kind == PrimitiveKind.TRIANGLE

    def test_rectangle(self, cls):
        assert cls.classify(rectangle()).kind == PrimitiveKind.RECTANGLE

    def test_too_few_points_is_dot(self, cls):
        assert cls.classify([(10, 10), (10.1, 10.1)]).kind == PrimitiveKind.DOT

    def test_squiggle_is_curve(self, cls):
        sq = [(20 + i * 3, 60 + 15 * math.sin(i * 0.4)) for i in range(40)]
        assert cls.classify(sq).kind == PrimitiveKind.CURVE

    def test_wobbly_circle_still_circle(self, cls):
        random.seed(42)
        wobble = [(100 + 50 * math.cos(t) + random.uniform(-3, 3),
                   100 + 50 * math.sin(t) + random.uniform(-3, 3))
                  for t in np.linspace(0, 2 * math.pi, 60)]
        assert cls.classify(wobble).kind == PrimitiveKind.CIRCLE

    def test_returns_confidence(self, cls):
        result = cls.classify(circle())
        assert 0.0 <= result.confidence <= 1.0

    def test_returns_bbox(self, cls):
        result = cls.classify(rectangle())
        assert len(result.bbox) == 4
        x, y, w, h = result.bbox
        assert w > 0 and h > 0
