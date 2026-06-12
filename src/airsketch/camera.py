"""Webcam / video-file capture with context-manager API.

Combines Skysketch's clean Camera class with AirDraw's rotation/mirror flags
that proved necessary for VDI environments where the redirected camera may be
inverted or mirrored.
"""
from __future__ import annotations

import cv2
import numpy as np


class Camera:
    """Context-managed camera/video source.

    Usage:
        with Camera(device_index=0) as cam:
            frame = cam.read()
            cv2.imshow("preview", frame)
    """

    def __init__(
        self,
        device_index: int | str = 0,
        width: int = 1280,
        height: int = 720,
        rotate_180: bool = False,
        mirror: bool = False,
    ):
        self._source = device_index
        self._width = width
        self._height = height
        self._rotate_180 = rotate_180
        self._mirror = mirror
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        self.open()
        if not self.is_open:
            raise RuntimeError(f"Could not open camera/video: {self._source}")
        return self

    def __exit__(self, *exc):
        self.release()

    def open(self) -> bool:
        if isinstance(self._source, int):
            # Try backends in order: default, MSMF, DSHOW
            for backend in (cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW):
                cap = cv2.VideoCapture(self._source, backend)
                if cap.isOpened():
                    self._cap = cap
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    break
                cap.release()
        else:
            self._cap = cv2.VideoCapture(self._source)
        return self.is_open

    def read(self) -> np.ndarray:
        """Read one frame (BGR). Raises RuntimeError on EOF / failure."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera not opened")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Frame read failed (end of stream?)")
        if self._rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if self._mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
                import gc
                gc.collect()

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        if self._cap is None:
            return self._width
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self._cap is None:
            return self._height
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
