import cv2
import numpy as np


class VideoSource:
    """Abstraction over webcam or video file input."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        rotate_180: bool = False,
        mirror: bool = True,
    ):
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self._width = width
        self._height = height
        self._rotate_180 = rotate_180
        self._mirror = mirror

    def open(self) -> bool:
        if isinstance(self._source, int):
            # Try backends in order: default, MSMF, DSHOW
            for backend in (cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW):
                cap = cv2.VideoCapture(self._source, backend)
                if cap.isOpened():
                    self._cap = cap
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    break
                cap.release()
        else:
            self._cap = cv2.VideoCapture(self._source)
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if ret:
            if self._rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if self._mirror:
                frame = cv2.flip(frame, 1)
        return ret, frame

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                # Drop the reference so the COM/DirectShow handle is freed
                self._cap = None
                import gc
                gc.collect()

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def source_label(self) -> str:
        if isinstance(self._source, int):
            return f"Camera {self._source}"
        return str(self._source)
