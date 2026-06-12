"""Unit tests for the OpenVINO hand-tracking math (no model needed).

These cover the pieces the live camera is most sensitive to and which the plan
flagged as the top risks: BlazePalm anchor generation, SSD box decode, weighted
NMS, and the rotated-ROI affine + its inverse landmark mapping.
"""
import math
import os

import numpy as np
import pytest

from airsketch.hand_tracker_ov import (
    NUM_ANCHORS,
    PALM_INPUT,
    ROI,
    _rotation_from_vector,
    _sigmoid,
    crop_roi,
    decode_boxes,
    generate_anchors,
    project_points,
    roi_affine,
    roi_from_landmarks,
    roi_from_palm,
    weighted_nms,
)


# ----------------------------------------------------------------- anchors
class TestAnchors:
    def test_count_is_2016(self):
        anchors = generate_anchors()
        assert anchors.shape == (NUM_ANCHORS, 4)

    def test_layer_split_matches_feature_maps(self):
        # stride-8 layer: 24*24*2 = 1152 ; stride-16 group: 12*12*6 = 864
        anchors = generate_anchors()
        assert anchors.shape[0] == 1152 + 864

    def test_centers_normalized_and_fixed_size(self):
        anchors = generate_anchors()
        assert anchors[:, 0].min() >= 0.0 and anchors[:, 0].max() <= 1.0
        assert anchors[:, 1].min() >= 0.0 and anchors[:, 1].max() <= 1.0
        # fixed_anchor_size=True -> all w == h == 1.0
        assert np.allclose(anchors[:, 2], 1.0)
        assert np.allclose(anchors[:, 3], 1.0)

    def test_first_anchor_is_top_left_cell_center(self):
        anchors = generate_anchors()
        # first cell of the 24x24 (stride 8) map: center (0.5/24, 0.5/24)
        assert anchors[0, 0] == pytest.approx(0.5 / 24, abs=1e-6)
        assert anchors[0, 1] == pytest.approx(0.5 / 24, abs=1e-6)


# ----------------------------------------------------------------- decode
class TestDecode:
    def test_zero_regressors_recover_anchor_box(self):
        # With raw regressors all zero, the decoded box collapses to a zero-size
        # box centred on the anchor (w/h come from raw, which is 0).
        anchors = np.array([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
        raw = np.zeros((1, 18), dtype=np.float32)
        dec = decode_boxes(raw, anchors)
        # xmin==xmax==ymin==ymax==0.5 (center), keypoints all at the centre too
        assert dec[0, 0] == pytest.approx(0.5)
        assert dec[0, 2] == pytest.approx(0.5)
        assert dec[0, 4] == pytest.approx(0.5)  # kp0 x
        assert dec[0, 5] == pytest.approx(0.5)  # kp0 y

    def test_known_box_decode(self):
        # anchor at centre, a box of width/height = half the input.
        anchors = np.array([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
        raw = np.zeros((1, 18), dtype=np.float32)
        raw[0, 2] = PALM_INPUT / 2.0   # w -> 0.5 normalized
        raw[0, 3] = PALM_INPUT / 2.0   # h -> 0.5 normalized
        dec = decode_boxes(raw, anchors)
        assert dec[0, 0] == pytest.approx(0.25)  # xmin = 0.5 - 0.25
        assert dec[0, 2] == pytest.approx(0.75)  # xmax = 0.5 + 0.25

    def test_keypoint_offset(self):
        anchors = np.array([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
        raw = np.zeros((1, 18), dtype=np.float32)
        raw[0, 4] = PALM_INPUT * 0.25  # kp0 x offset -> +0.25
        dec = decode_boxes(raw, anchors)
        assert dec[0, 4] == pytest.approx(0.75)


# ----------------------------------------------------------------- NMS
class TestWeightedNMS:
    def test_merges_overlapping(self):
        # two near-identical boxes -> one merged detection
        a = np.zeros((2, 18), dtype=np.float32)
        a[0, :4] = [0.10, 0.10, 0.50, 0.50]
        a[1, :4] = [0.12, 0.12, 0.52, 0.52]
        scores = np.array([0.9, 0.8], dtype=np.float32)
        out = weighted_nms(a, scores, max_detections=5)
        assert len(out) == 1
        merged = out[0][0]
        # merged xmin is the score-weighted avg, between the two inputs
        assert 0.10 <= merged[0] <= 0.12

    def test_keeps_disjoint(self):
        a = np.zeros((2, 18), dtype=np.float32)
        a[0, :4] = [0.0, 0.0, 0.2, 0.2]
        a[1, :4] = [0.7, 0.7, 0.9, 0.9]
        scores = np.array([0.9, 0.85], dtype=np.float32)
        out = weighted_nms(a, scores, max_detections=5)
        assert len(out) == 2

    def test_highest_score_first(self):
        a = np.zeros((2, 18), dtype=np.float32)
        a[0, :4] = [0.0, 0.0, 0.2, 0.2]
        a[1, :4] = [0.7, 0.7, 0.9, 0.9]
        scores = np.array([0.4, 0.95], dtype=np.float32)
        out = weighted_nms(a, scores, max_detections=5)
        assert out[0][1] == pytest.approx(0.95)

    def test_empty(self):
        assert weighted_nms(np.zeros((0, 18), np.float32), np.zeros((0,), np.float32)) == []


# ----------------------------------------------------------------- sigmoid
def test_sigmoid_clips_and_bounds():
    out = _sigmoid(np.array([-1e9, 0.0, 1e9], dtype=np.float32))
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- rotation
class TestRotation:
    def test_upright_is_zero(self):
        # p1 directly above p0 (image y grows down) -> upright -> rotation 0
        assert _rotation_from_vector(100, 200, 100, 100) == pytest.approx(0.0, abs=1e-6)

    def test_pointing_right_is_quarter_turn(self):
        rot = _rotation_from_vector(100, 100, 200, 100)
        assert rot == pytest.approx(math.pi / 2, abs=1e-6)


# ----------------------------------------------------------------- ROI affine
class TestROIAffine:
    def test_center_maps_to_crop_center(self):
        roi = ROI(cx=320, cy=240, size=200, rotation=0.0)
        M = roi_affine(roi, out_size=224)
        center = project_points_forward(M, [(320, 240)])[0]
        assert center[0] == pytest.approx(111.5, abs=1.0)  # (224-1)/2
        assert center[1] == pytest.approx(111.5, abs=1.0)

    def test_inverse_round_trip(self):
        roi = ROI(cx=300, cy=260, size=180, rotation=0.6)
        M = roi_affine(roi, out_size=224)
        # pick some crop-space points, map back to frame, then forward again
        crop_pts = np.array([[0, 0], [223, 0], [112, 112], [50, 200]], np.float32)
        frame_pts = project_points(crop_pts, M)
        back = project_points_forward(M, frame_pts)
        assert np.allclose(back, crop_pts, atol=1e-3)

    def test_upright_crop_is_axis_aligned(self):
        # rotation 0 -> the frame->crop map is a pure scale+translation
        roi = ROI(cx=320, cy=240, size=224, rotation=0.0)
        M = roi_affine(roi, out_size=224)
        # off-diagonal terms ~ 0
        assert abs(M[0, 1]) < 1e-6
        assert abs(M[1, 0]) < 1e-6


def project_points_forward(M, pts):
    """Apply the forward affine M (frame -> crop) to (N,2) points."""
    pts = np.asarray(pts, dtype=np.float32)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    return homog @ M.T


# ----------------------------------------------------------------- crop
def test_crop_shape_and_inverse_consistency():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = ROI(cx=320, cy=240, size=200, rotation=0.3)
    crop, M = crop_roi(frame, roi, out_size=224)
    assert crop.shape == (224, 224, 3)
    # a landmark at crop centre maps near the ROI centre in the frame
    framed = project_points(np.array([[111.5, 111.5]], np.float32), M)[0]
    assert framed[0] == pytest.approx(320, abs=1.5)
    assert framed[1] == pytest.approx(240, abs=1.5)


# ----------------------------------------------------------------- tracking ROI
class TestROIFromLandmarks:
    def _upright_hand(self):
        # 21 landmarks roughly forming an upright hand: wrist low, fingers up.
        # Wrist sits directly below the middle-finger MCP (landmark 9, x=335)
        # so the wrist->MCP vector points straight up -> rotation ~ 0.
        lm = [(335, 400)]                      # 0 wrist
        lm += [(300, 380), (290, 360), (285, 345), (280, 330)]   # thumb
        for base_x in (320, 335, 350, 365):    # index, middle, ring, pinky MCP..tip
            lm += [(base_x, 360), (base_x, 330), (base_x, 305), (base_x, 285)]
        return lm

    def test_rotation_near_zero_for_upright(self):
        roi = roi_from_landmarks(self._upright_hand())
        assert abs(roi.rotation) < 0.3

    def test_size_positive_and_covers_hand(self):
        roi = roi_from_landmarks(self._upright_hand())
        assert roi.size > 0
        # center inside the frame region of the hand
        assert 280 < roi.cx < 380
        assert 280 < roi.cy < 410


# ----------------------------------------------------------------- ROI from palm
# ----------------------------------------------------------------- preprocess
def test_preprocess_normalizes_to_unit_range():
    from airsketch.hand_tracker_ov import OpenVINOHandTracker
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    blob = OpenVINOHandTracker._preprocess(img, size=224, norm=(0.0, 1.0))
    assert blob.shape == (1, 224, 224, 3)
    assert blob.dtype == np.float32
    assert blob.max() == pytest.approx(1.0)
    black = OpenVINOHandTracker._preprocess(np.zeros((10, 10, 3), np.uint8), 224, (0.0, 1.0))
    assert black.min() == pytest.approx(0.0)


# ----------------------------------------------------------------- integration
# These load the real OpenVINO models (which ship with the app). Skipped when
# OpenVINO or the model bundle isn't available, so the math tests above still run.
def _models_available() -> bool:
    try:
        import openvino  # noqa: F401
    except Exception:
        return False
    return os.path.exists(os.path.join("models", "hand_landmarker.task")) or (
        os.path.exists(os.path.join("models", "hand_ov", "hand_detector.tflite")))


needs_models = pytest.mark.skipif(
    not _models_available(), reason="OpenVINO and/or hand model bundle not available")


@needs_models
class TestIntegration:
    @pytest.fixture(scope="class")
    def tracker(self):
        from airsketch.hand_tracker_ov import OpenVINOHandTracker
        t = OpenVINOHandTracker(device="CPU", draw=False)
        yield t
        t.release()

    def test_blank_frame_not_visible(self, tracker):
        """Regression guard: a blank frame must NOT report a hand.

        Catches the wrong-normalization bug ([-1,1] made the palm detector
        saturate to score 1.0 on uniform input -> phantom hands)."""
        tracker._reset_track()
        r = tracker.process(np.zeros((480, 640, 3), np.uint8))
        assert r.visible is False

    def test_noise_frame_not_visible(self, tracker):
        tracker._reset_track()
        rng = np.random.default_rng(0)
        r = tracker.process(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
        assert r.visible is False

    def test_result_contract_shape(self, tracker):
        """A processed frame returns a well-formed HandResult either way."""
        tracker._reset_track()
        r = tracker.process(np.zeros((480, 640, 3), np.uint8))
        assert hasattr(r, "visible")
        if r.visible:
            assert r.landmarks is not None and len(r.landmarks) == 21
            assert r.world_landmarks is not None and len(r.world_landmarks) == 21

    def test_gpu_request_falls_back_to_cpu(self):
        """Requesting an unavailable device must not crash (graceful fallback)."""
        from airsketch.hand_tracker_ov import OpenVINOHandTracker
        t = OpenVINOHandTracker(device="GPU", draw=False)
        assert t.device in ("GPU", "CPU")  # CPU here; GPU only if one exists
        t.release()


def test_roi_from_palm_basic():
    # decoded row: box centred, wrist below middle-MCP (upright)
    row = np.zeros(18, dtype=np.float32)
    row[:4] = [0.4, 0.4, 0.6, 0.6]   # box -> center (0.5,0.5) size 0.2
    row[4], row[5] = 0.5, 0.55       # kp0 (wrist) lower
    row[8], row[9] = 0.5, 0.45       # kp2 (middle MCP) higher -> upright
    roi = roi_from_palm(row, frame_w=640, frame_h=640)
    assert abs(roi.rotation) < 0.2
    assert roi.size > 0
    # expanded beyond the raw box (0.2*640 = 128 px)
    assert roi.size > 128
