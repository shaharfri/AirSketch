"""Visual effects for the drawing canvas — all pure OpenCV, toggle on/off.

Usage:
    fx = EffectsChain()
    fx.toggle()                    # cycle through effects
    canvas_with_fx = fx.apply(canvas, cursor, speed)
"""

from __future__ import annotations

import time
from enum import Enum

import cv2
import numpy as np


class EffectMode(Enum):
    NONE = "none"
    GLOW = "glow"
    SPEED_COLOR = "speed_color"
    GLOW_AND_COLOR = "glow+color"


class EffectsChain:
    """Manages and applies visual effects to the canvas."""

    _MODES = list(EffectMode)

    def __init__(self, mode: EffectMode = EffectMode.NONE):
        self._mode_idx = self._MODES.index(mode)
        self._prev_cursor: tuple[int, int] | None = None
        self._prev_time: float = 0.0

    @property
    def mode(self) -> EffectMode:
        return self._MODES[self._mode_idx]

    @property
    def mode_name(self) -> str:
        return self.mode.value

    def toggle(self) -> str:
        """Cycle to next effect mode. Returns new mode name."""
        self._mode_idx = (self._mode_idx + 1) % len(self._MODES)
        return self.mode_name

    def compute_speed(self, cursor: tuple[int, int] | None) -> float:
        """Calculate cursor speed in pixels/frame."""
        if cursor is None or self._prev_cursor is None:
            self._prev_cursor = cursor
            return 0.0
        dx = cursor[0] - self._prev_cursor[0]
        dy = cursor[1] - self._prev_cursor[1]
        speed = (dx * dx + dy * dy) ** 0.5
        self._prev_cursor = cursor
        return speed

    def get_color_for_speed(self, speed: float) -> tuple[int, int, int]:
        """Map speed to BGR color via HSV hue rotation."""
        # Slow = green (60), fast = red (0) via blue (120)
        hue = int(max(0, min(179, 60 + speed * 0.8)))
        hsv = np.array([[[hue, 255, 230]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])

    def apply(self, canvas: np.ndarray, cursor: tuple[int, int] | None = None) -> np.ndarray:
        """Apply current effect to canvas. Returns modified copy."""
        mode = self.mode
        if mode == EffectMode.NONE:
            return canvas

        result = canvas.copy()

        if mode in (EffectMode.GLOW, EffectMode.GLOW_AND_COLOR):
            result = _apply_glow(result)

        return result

    def apply_to_line_color(self, base_color: tuple[int, int, int],
                            cursor: tuple[int, int] | None) -> tuple[int, int, int]:
        """Get the line color for the current frame (may vary by speed)."""
        mode = self.mode
        if mode in (EffectMode.SPEED_COLOR, EffectMode.GLOW_AND_COLOR):
            speed = self.compute_speed(cursor)
            return self.get_color_for_speed(speed)
        else:
            self.compute_speed(cursor)  # keep tracking for smooth transition
            return base_color


def _apply_glow(canvas: np.ndarray) -> np.ndarray:
    """Add neon glow effect to drawn lines."""
    # Create blur of the drawn content
    blurred = cv2.GaussianBlur(canvas, (15, 15), 5.0)
    # Additive blend — glow underneath the sharp lines
    glowed = cv2.add(blurred, canvas)
    return glowed
