import math
import os
import time
import urllib.request
from abc import ABC, abstractmethod

import cv2
import numpy as np

from airsketch.config import HandResult

# --- OpenVINO integration point ---
# To replace MediaPipe with OpenVINO hand-pose estimation:
#   1. Subclass HandTracker
#   2. Load an OpenVINO IR model (.xml/.bin) for hand landmark detection
#   3. Preprocess the frame, run inference via openvino.runtime.Core, post-process
#   4. Map the resulting landmarks to HandResult
# The rest of the application remains unchanged.


class HandTracker(ABC):
    """Interface for hand tracking backends."""

    @abstractmethod
    def process(self, frame: np.ndarray) -> HandResult:
        ...

    @abstractmethod
    def release(self) -> None:
        ...


# MediaPipe Tasks API hand-landmark model
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_FILENAME = "hand_landmarker.task"

# Hand-skeleton connections (landmark index pairs) for drawing
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm
]


def _ensure_model(model_dir: str = "models") -> str:
    """Download the hand-landmarker .task file on first run."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, _MODEL_FILENAME)
    if not os.path.exists(path):
        print(f"Downloading hand-landmark model to {path} ...")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("Model downloaded.")
    return path


class MediaPipeHandTracker(HandTracker):
    """Hand tracker using MediaPipe Tasks (new API).

    Returns full landmark list, index-tip, thumb-tip, wrist, and a hand-size
    reference distance suitable for normalizing pinch detection.
    """

    WRIST_ID = 0
    THUMB_TIP_ID = 4
    INDEX_TIP_ID = 8
    MIDDLE_MCP_ID = 9   # middle-finger base — stable reference for hand size

    def __init__(
        self,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
        max_hands: int = 1,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = _ensure_model()

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=tracking_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process(self, frame: np.ndarray) -> HandResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self._start_time) * 1000)

        try:
            result = self._detector.detect_for_video(mp_image, ts_ms)
        except Exception:
            return HandResult(visible=False)

        if not result.hand_landmarks:
            return HandResult(visible=False)

        landmarks = result.hand_landmarks[0]
        h, w, _ = frame.shape

        pixel_landmarks = [(lm.x * w, lm.y * h) for lm in landmarks]

        # 3D world landmarks (meters, hand-centered) — orientation invariant
        world_landmarks: list[tuple[float, float, float]] | None = None
        if (hasattr(result, "hand_world_landmarks")
                and result.hand_world_landmarks
                and result.hand_world_landmarks[0]):
            world_landmarks = [
                (lm.x, lm.y, lm.z) for lm in result.hand_world_landmarks[0]
            ]

        self._draw_landmarks(frame, pixel_landmarks)

        index_tip = pixel_landmarks[self.INDEX_TIP_ID]
        thumb_tip = pixel_landmarks[self.THUMB_TIP_ID]
        wrist = pixel_landmarks[self.WRIST_ID]
        middle_mcp = pixel_landmarks[self.MIDDLE_MCP_ID]

        # Hand size: wrist → middle-finger base, robust to finger pose
        hand_size = math.hypot(
            wrist[0] - middle_mcp[0], wrist[1] - middle_mcp[1]
        )
        if hand_size < 1.0:
            hand_size = 1.0

        confidence = 0.9
        if result.handedness and result.handedness[0]:
            confidence = float(result.handedness[0][0].score)

        return HandResult(
            visible=True,
            fingertip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            hand_size=hand_size,
            confidence=confidence,
            landmarks=pixel_landmarks,
            world_landmarks=world_landmarks,
        )

    def _draw_landmarks(
        self, frame: np.ndarray, pixel_landmarks: list
    ) -> None:
        pts = [(int(x), int(y)) for x, y in pixel_landmarks]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if i == self.INDEX_TIP_ID:
                color, r = (0, 255, 255), 5
            elif i == self.THUMB_TIP_ID:
                color, r = (255, 180, 0), 5
            else:
                color, r = (0, 180, 255), 3
            cv2.circle(frame, p, r, color, -1, cv2.LINE_AA)

    def release(self) -> None:
        self._detector.close()


def create_hand_tracker(cfg) -> "HandTracker":
    """Build the hand tracker selected by ``cfg.hand_tracker_backend``.

    "openvino" -> :class:`OpenVINOHandTracker` (BlazePalm + landmark on the
    OpenVINO runtime, device from ``cfg.hand_device``). Falls back to MediaPipe
    if the OpenVINO backend can't be constructed (e.g. models missing).
    "mediapipe" (default) -> :class:`MediaPipeHandTracker`.

    Imports are lazy so neither ``openvino`` nor ``mediapipe`` is loaded unless
    the corresponding backend is actually chosen.
    """
    backend = str(getattr(cfg, "hand_tracker_backend", "mediapipe")).lower()
    if backend == "openvino":
        try:
            from airsketch.hand_tracker_ov import OpenVINOHandTracker
            tracker = OpenVINOHandTracker(
                device=getattr(cfg, "hand_device", "AUTO"),
                debug=getattr(cfg, "hand_debug", False))
            print(f"[hand]     OpenVINO hand tracker on {tracker.device}"
                  + ("  [debug overlay ON]" if getattr(cfg, "hand_debug", False) else ""))
            return tracker
        except Exception as e:
            print(f"[hand]     OpenVINO hand tracker unavailable "
                  f"({type(e).__name__}: {e}); falling back to MediaPipe.")
    tracker = MediaPipeHandTracker(
        detection_confidence=getattr(cfg, "hand_detection_confidence", 0.7),
        tracking_confidence=getattr(cfg, "hand_tracking_confidence", 0.6),
    )
    print("[hand]     MediaPipe hand tracker (TFLite/CPU)")
    return tracker
