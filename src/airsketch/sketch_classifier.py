"""CNN-based sketch classification using OpenVINO inference."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from airsketch.inference_engine import InferenceEngine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL = str(_PROJECT_ROOT / "models" / "sketch_classifier.xml")
_DEFAULT_LABELS = str(_PROJECT_ROOT / "models" / "class_names.json")


class SketchClassifier:
    """Classify hand-drawn sketches on a canvas image via OpenVINO.

    Usage:
        classifier = SketchClassifier()
        label, confidence = classifier.classify(canvas_image)
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        labels_path: str = _DEFAULT_LABELS,
        device: str = "AUTO",
        confidence_threshold: float = 0.5,
    ):
        self._engine = InferenceEngine(model_path, device)
        with open(labels_path) as f:
            self._labels: list[str] = json.load(f)
        self._threshold = confidence_threshold

    def classify(self, canvas: np.ndarray) -> tuple[str, float]:
        """Classify a canvas image.

        Args:
            canvas: BGR canvas image (any size, colored drawing on black).

        Returns:
            (label, confidence). Returns ("unknown", 0.0) if below threshold.
        """
        tensor = self._preprocess(canvas)
        logits = self._engine.infer(tensor)
        probs = _softmax(logits[0])
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if conf < self._threshold:
            return ("unknown", conf)
        return (self._labels[idx], conf)

    @property
    def labels(self) -> list[str]:
        return self._labels

    @staticmethod
    def _preprocess(canvas: np.ndarray) -> np.ndarray:
        """Convert canvas to model input: 1x1x28x28 float32."""
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Normalize to full 0-255 range regardless of drawing color
        max_val = gray.max()
        if max_val > 0:
            gray = (gray.astype(np.float32) / max_val * 255).astype(np.uint8)

        # Find bounding box of drawn content
        coords = cv2.findNonZero(gray)
        if coords is None:
            return np.zeros((1, 1, 28, 28), dtype=np.float32)

        x, y, w, h = cv2.boundingRect(coords)

        # Add 10% padding
        pad = max(w, h) // 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        cropped = gray[y1:y2, x1:x2]

        # Make square (pad shorter dimension with black)
        h_c, w_c = cropped.shape
        size = max(h_c, w_c)
        square = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - h_c) // 2
        x_off = (size - w_c) // 2
        square[y_off:y_off + h_c, x_off:x_off + w_c] = cropped

        # Resize to 28x28
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] float32, add batch and channel dims
        tensor = resized.astype(np.float32) / 255.0
        return tensor.reshape(1, 1, 28, 28)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
