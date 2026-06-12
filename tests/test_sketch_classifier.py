"""Tests for sketch classifier preprocessing and classification logic."""

import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock

from airsketch.sketch_classifier import SketchClassifier, _softmax


class TestPreprocess:
    def test_empty_canvas_returns_zeros(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        result = SketchClassifier._preprocess(canvas)
        assert result.shape == (1, 1, 28, 28)
        assert np.all(result == 0)

    def test_output_shape(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(canvas, (320, 240), 50, (255, 255, 255), 3)
        result = SketchClassifier._preprocess(canvas)
        assert result.shape == (1, 1, 28, 28)
        assert result.dtype == np.float32

    def test_normalized_range(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (100, 100), (300, 300), (0, 255, 0), 3)
        result = SketchClassifier._preprocess(canvas)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_centers_small_drawing(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a tiny square in the top-left corner
        cv2.rectangle(canvas, (10, 10), (40, 40), (255, 255, 255), 2)
        result = SketchClassifier._preprocess(canvas)
        # After crop+center+resize, the drawing should occupy most of the 28x28 image
        # Check that there's content (non-zero) somewhere in the output
        assert np.any(result > 0)
        # The edges of the input canvas should NOT affect centering —
        # content should be roughly centered, not in one corner
        top_half = result[0, 0, :14, :].sum()
        bottom_half = result[0, 0, 14:, :].sum()
        # Both halves should have some content (drawing is centered)
        assert top_half > 0 and bottom_half > 0

    def test_handles_colored_drawing(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        # Green drawing (like our app's default)
        cv2.circle(canvas, (320, 240), 80, (0, 255, 0), 4)
        result = SketchClassifier._preprocess(canvas)
        assert np.any(result > 0)


class TestSoftmax:
    def test_sums_to_one(self):
        logits = np.array([2.0, 1.0, 0.1, -1.0])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_argmax_preserved(self):
        logits = np.array([0.5, 3.0, 1.0])
        probs = _softmax(logits)
        assert np.argmax(probs) == 1

    def test_handles_large_values(self):
        logits = np.array([1000.0, 1001.0, 999.0])
        probs = _softmax(logits)
        assert not np.any(np.isnan(probs))
        assert abs(probs.sum() - 1.0) < 1e-6


class TestClassify:
    @patch("airsketch.sketch_classifier.InferenceEngine")
    def test_returns_label_and_confidence(self, mock_engine_cls):
        mock_engine = MagicMock()
        # Simulate high confidence for "house" (index 3)
        logits = np.zeros((1, 12), dtype=np.float32)
        logits[0, 3] = 5.0  # house
        mock_engine.infer.return_value = logits
        mock_engine_cls.return_value = mock_engine

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.read = lambda: '["triangle","square","circle","house","car","tree","star","cat","flower","sun","airplane","fish"]'

            # Manually construct classifier with mocked engine
            classifier = SketchClassifier.__new__(SketchClassifier)
            classifier._engine = mock_engine
            classifier._labels = ["triangle", "square", "circle", "house", "car",
                                   "tree", "star", "cat", "flower", "sun", "airplane", "fish"]
            classifier._threshold = 0.5

        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(canvas, (320, 240), 50, (255, 255, 255), 3)

        label, conf = classifier.classify(canvas)
        assert label == "house"
        assert conf > 0.5

    @patch("airsketch.sketch_classifier.InferenceEngine")
    def test_below_threshold_returns_unknown(self, mock_engine_cls):
        mock_engine = MagicMock()
        # Uniform logits → no class is confident
        logits = np.ones((1, 12), dtype=np.float32) * 0.1
        mock_engine.infer.return_value = logits
        mock_engine_cls.return_value = mock_engine

        classifier = SketchClassifier.__new__(SketchClassifier)
        classifier._engine = mock_engine
        classifier._labels = ["triangle", "square", "circle", "house", "car",
                               "tree", "star", "cat", "flower", "sun", "airplane", "fish"]
        classifier._threshold = 0.5

        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(canvas, (320, 240), 50, (255, 255, 255), 3)

        label, conf = classifier.classify(canvas)
        assert label == "unknown"
