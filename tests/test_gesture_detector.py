"""Tests for the angle-based IndexPointingDetector."""
import math

import pytest

from airsketch.gesture_detector import (
    IndexPointingDetector,
    PinchDetector,
    finger_angle,
)


def _make_landmarks(idx_curl_deg=0, mid_curl_deg=180, ring_curl_deg=60, pinky_curl_deg=60):
    """Synthesize 21 landmarks at canonical hand positions.

    Each finger's joint angles are controlled by the *curl* parameter:
        180° = fully extended
         60° = curled (fist)
    """
    def finger_chain(mcp_x: float, angle_deg: float):
        mcp = (mcp_x, 250.0)
        pip = (mcp_x, 200.0)
        # Bend at PIP by `180 - angle_deg`. At 180 the chain is straight (up).
        rad = math.radians(180 - angle_deg)
        dip = (pip[0] + 30 * math.sin(rad), pip[1] - 30 * math.cos(rad))
        # Same bend at DIP for cleaner classification
        dip_to_tip = math.radians((180 - angle_deg) * 2)
        tip = (dip[0] + 30 * math.sin(dip_to_tip), dip[1] - 30 * math.cos(dip_to_tip))
        return [mcp, pip, dip, tip]

    lms = [(100.0, 300.0)]  # wrist
    lms += [(80, 280), (70, 260), (60, 240), (50, 220)]  # thumb
    lms += finger_chain(100, idx_curl_deg)
    lms += finger_chain(115, mid_curl_deg)
    lms += finger_chain(130, ring_curl_deg)
    lms += finger_chain(145, pinky_curl_deg)
    return lms


class TestFingerAngle:
    def test_extended_is_high(self):
        lms = _make_landmarks(idx_curl_deg=180)
        angle = finger_angle(lms, 5, 6, 7, 8)
        assert angle > 150

    def test_curled_is_low(self):
        lms = _make_landmarks(idx_curl_deg=60)
        angle = finger_angle(lms, 5, 6, 7, 8)
        assert angle < 130


class TestIndexPointing:
    def test_pointing_pose_activates(self):
        d = IndexPointingDetector(confirm_frames=3)
        lms = _make_landmarks(idx_curl_deg=180, mid_curl_deg=60)
        for _ in range(4):
            d.update(lms)
        assert d.is_active

    def test_open_palm_does_not_activate(self):
        d = IndexPointingDetector(confirm_frames=3)
        lms = _make_landmarks(idx_curl_deg=180, mid_curl_deg=180,
                              ring_curl_deg=180, pinky_curl_deg=180)
        for _ in range(4):
            d.update(lms)
        assert not d.is_active

    def test_fist_does_not_activate(self):
        d = IndexPointingDetector(confirm_frames=3)
        lms = _make_landmarks(idx_curl_deg=60, mid_curl_deg=60)
        for _ in range(4):
            d.update(lms)
        assert not d.is_active

    def test_none_landmarks_deactivates(self):
        d = IndexPointingDetector(confirm_frames=3)
        for _ in range(4):
            d.update(_make_landmarks(idx_curl_deg=180, mid_curl_deg=60))
        assert d.is_active
        d.update(None)
        assert not d.is_active

    def test_diagnostics_present(self):
        d = IndexPointingDetector()
        d.update(_make_landmarks())
        diag = d.diagnostics
        # Either flat ('index_a') or nested ('index': {'angle': ...}) format ok
        assert "index_a" in diag or "index" in diag


class TestPinch:
    def test_close_thumb_index_is_pinch(self):
        d = PinchDetector(pinch_threshold=0.30, confirm_frames=2)
        # Need confirm_frames consistent updates before transition flips
        for _ in range(4):
            d.update((10, 10), (12, 12), 100.0)
        assert d.is_pinching

    def test_apart_is_no_pinch(self):
        d = PinchDetector(pinch_threshold=0.30, confirm_frames=2)
        for _ in range(3):
            assert d.update((10, 10), (90, 90), 100.0) == False
        assert not d.is_pinching

    def test_none_resets(self):
        d = PinchDetector()
        for _ in range(3):
            d.update((10, 10), (12, 12), 100.0)
        assert d.is_pinching
        d.update(None, (12, 12), 100.0)
        assert not d.is_pinching
