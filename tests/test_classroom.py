"""Tests for the classroom challenge engine + judge."""
import math

import numpy as np
import pytest

from airsketch.classroom.challenge_engine import (
    Challenge,
    ChallengeEngine,
    ChallengeResult,
    GEOMETRY_TARGETS,
    OBJECT_TARGETS,
)
from airsketch.classroom.judge import Judge, render_for_cnn, _stars_from_score, _matches
from airsketch.stroke import Diagram, Stroke


# ---- helpers ----

def _circle_diagram(cx=150, cy=150, r=50, n=40):
    d = Diagram()
    pts = [(cx + r * math.cos(t), cy + r * math.sin(t))
           for t in np.linspace(0, 2 * math.pi, n)]
    s = Stroke(points=pts)
    s.snapped_to = "circle"
    s.snap_confidence = 0.95
    s.finalize()
    d.add_stroke(s)
    return d


def _triangle_diagram():
    verts = [(150, 50), (220, 170), (80, 170), (150, 50)]
    pts = []
    for i in range(3):
        for j in range(20):
            f = j / 20
            pts.append((verts[i][0] + f * (verts[i + 1][0] - verts[i][0]),
                        verts[i][1] + f * (verts[i + 1][1] - verts[i][1])))
    d = Diagram()
    s = Stroke(points=pts)
    s.snapped_to = "triangle"
    s.snap_confidence = 0.9
    s.finalize()
    d.add_stroke(s)
    return d


# ---- engine ----

class TestChallengeEngine:
    def test_geometry_pool(self):
        eng = ChallengeEngine(theme="geometry", seed=1)
        ch = eng.next_challenge()
        assert ch.target in GEOMETRY_TARGETS
        assert ch.theme == "geometry"

    def test_objects_pool(self):
        eng = ChallengeEngine(theme="objects", seed=1)
        ch = eng.next_challenge()
        assert ch.target in OBJECT_TARGETS
        assert ch.theme == "objects"

    def test_round_increments(self):
        eng = ChallengeEngine(seed=1)
        eng.next_challenge()
        eng.next_challenge()
        assert eng.round_count == 2

    def test_avoid_repeats(self):
        eng = ChallengeEngine(theme="geometry", seed=3, avoid_repeats=True)
        prev = None
        for _ in range(10):
            ch = eng.next_challenge()
            assert ch.target != prev
            prev = ch.target

    def test_prompt_generated(self):
        ch = Challenge(target="triangle", theme="geometry", round_index=1)
        assert "TRIANGLE" in ch.prompt

    def test_retry_reissues_same_target_and_round(self):
        eng = ChallengeEngine(theme="geometry", seed=1)
        ch = eng.next_challenge()
        eng.record_result(ChallengeResult(
            challenge=ch, detected="circle", score=0, stars=0, passed=False))
        retry = eng.retry_last()
        assert retry is not None
        assert retry.target == ch.target
        assert retry.round_index == ch.round_index   # reuses the round
        assert eng.round_count == 1                  # round count unchanged
        assert eng.history == []                     # the miss was dropped

    def test_retry_does_not_count_the_miss(self):
        eng = ChallengeEngine(theme="geometry", seed=1)
        ch = eng.next_challenge()
        eng.record_result(ChallengeResult(
            challenge=ch, detected="dot", score=0, stars=0, passed=False))
        retry = eng.retry_last()
        eng.record_result(ChallengeResult(
            challenge=retry, detected=retry.target, score=92, stars=3, passed=True))
        assert eng.passed_count == 1
        assert eng.total_score == 92                 # only the successful retry counts

    def test_retry_with_no_history_returns_none(self):
        eng = ChallengeEngine(seed=1)
        assert eng.retry_last() is None


# ---- judge ----

class TestJudge:
    def test_correct_geometry_passes(self):
        judge = Judge(sketch_classifier=None)
        ch = Challenge(target="triangle", theme="geometry", round_index=1)
        result = judge.judge(_triangle_diagram(), ch)
        assert result.passed
        assert result.detected == "triangle"
        assert result.score > 55
        assert result.stars >= 1

    def test_wrong_geometry_fails(self):
        judge = Judge(sketch_classifier=None)
        ch = Challenge(target="triangle", theme="geometry", round_index=1)
        result = judge.judge(_circle_diagram(), ch)
        assert not result.passed
        assert result.score == 0

    def test_circle_matches_circle(self):
        judge = Judge(sketch_classifier=None)
        ch = Challenge(target="circle", theme="geometry", round_index=1)
        result = judge.judge(_circle_diagram(), ch)
        assert result.passed

    def test_object_without_cnn_fails_gracefully(self):
        judge = Judge(sketch_classifier=None)
        ch = Challenge(target="house", theme="objects", round_index=1)
        result = judge.judge(_triangle_diagram(), ch)
        assert not result.passed
        assert "CNN" in result.explanation

    def test_empty_diagram(self):
        judge = Judge(sketch_classifier=None)
        ch = Challenge(target="circle", theme="geometry", round_index=1)
        result = judge.judge(Diagram(), ch)
        assert not result.passed


class TestHelpers:
    def test_stars_thresholds(self):
        assert _stars_from_score(90) == 3
        assert _stars_from_score(75) == 2
        assert _stars_from_score(60) == 1
        assert _stars_from_score(40) == 0

    def test_synonyms(self):
        assert _matches("square", "rectangle")
        assert _matches("rectangle", "square")
        assert _matches("circle", "circle")
        assert not _matches("triangle", "circle")

    def test_render_for_cnn_black_bg(self):
        canvas = render_for_cnn(_circle_diagram(), size=128)
        assert canvas.shape == (128, 128, 3)
        # Black background with bright strokes
        assert canvas.min() == 0
        assert canvas.max() > 200
