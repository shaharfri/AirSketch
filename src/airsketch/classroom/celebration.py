"""Success / failure celebration animations (pure OpenCV)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    age: int = 0
    color: Tuple[int, int, int] = (0, 255, 255)
    size: int = 4


_CONFETTI_COLORS = [
    (80, 220, 255), (120, 255, 120), (255, 180, 80),
    (255, 120, 220), (80, 180, 255), (255, 255, 120),
]


class Celebration:
    """Plays a success or failure animation for a fixed number of frames.

    Usage:
        cel = Celebration()
        cel.start_success(stars=3)     # or cel.start_failure(target="triangle")
        while cel.is_active:
            cel.render(frame)
    """

    SUCCESS_FRAMES = 75
    FAILURE_FRAMES = 60

    def __init__(self):
        self._mode: Optional[str] = None
        self._frame = 0
        self._duration = 0
        self._particles: List[_Particle] = []
        self._stars = 0
        self._target = ""
        self._score = 0

    # ---- triggers ----

    def start_success(self, stars: int = 3, score: int = 100) -> None:
        self._mode = "success"
        self._frame = 0
        self._duration = self.SUCCESS_FRAMES
        self._stars = stars
        self._score = score
        self._particles.clear()

    def start_failure(self, target: str = "", score: int = 0) -> None:
        self._mode = "failure"
        self._frame = 0
        self._duration = self.FAILURE_FRAMES
        self._target = target
        self._score = score
        self._particles.clear()

    @property
    def is_active(self) -> bool:
        return self._mode is not None and self._frame < self._duration

    # ---- rendering ----

    def render(self, frame: np.ndarray) -> None:
        if not self.is_active:
            self._mode = None
            return
        if self._mode == "success":
            self._render_success(frame)
        else:
            self._render_failure(frame)
        self._frame += 1

    def _spawn_confetti(self, frame: np.ndarray, count: int) -> None:
        h, w = frame.shape[:2]
        for _ in range(count):
            self._particles.append(_Particle(
                x=random.uniform(w * 0.2, w * 0.8),
                y=random.uniform(-20, h * 0.2),
                vx=random.uniform(-2, 2),
                vy=random.uniform(2, 7),
                life=random.randint(40, 70),
                color=random.choice(_CONFETTI_COLORS),
                size=random.randint(3, 7),
            ))

    def _update_particles(self, frame: np.ndarray) -> None:
        alive = []
        for p in self._particles:
            p.x += p.vx
            p.y += p.vy
            p.vy += 0.12
            p.age += 1
            if p.age < p.life:
                fade = 1.0 - p.age / p.life
                col = tuple(int(c * fade) for c in p.color)
                # confetti as small rotated rectangles
                cv2.circle(frame, (int(p.x), int(p.y)), max(1, int(p.size * fade)),
                           col, -1, cv2.LINE_AA)
                alive.append(p)
        self._particles = alive

    def _render_success(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        t = self._frame / self._duration

        # Green vignette flash that fades
        if self._frame < 12:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (40, 200, 60), -1)
            a = 0.30 * (1 - self._frame / 12)
            cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)

        # Confetti
        if self._frame % 4 == 0 and self._frame < self._duration * 0.6:
            self._spawn_confetti(frame, 18)
        self._update_particles(frame)

        # Banner
        cx = w // 2
        scale = 1.6 + 0.25 * math.sin(self._frame * 0.4)
        text = "CORRECT!"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)[0]
        tx = cx - ts[0] // 2
        ty = int(h * 0.32)
        cv2.putText(frame, text, (tx + 2, ty + 2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, scale, (60, 255, 120), 3, cv2.LINE_AA)

        # Stars
        self._draw_stars(frame, cx, int(h * 0.45), self._stars, reveal=t)

        # Score
        s = f"Score: {self._score}"
        ss = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.putText(frame, s, (cx - ss[0] // 2, int(h * 0.56)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 255), 2, cv2.LINE_AA)

    def _render_failure(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        cx = w // 2

        # Red flash
        if self._frame < 10:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (60, 60, 220), -1)
            a = 0.25 * (1 - self._frame / 10)
            cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)

        # Horizontal shake on the banner
        shake = int(8 * math.sin(self._frame * 1.1)) if self._frame < 25 else 0
        text = "TRY AGAIN!"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)[0]
        tx = cx - ts[0] // 2 + shake
        ty = int(h * 0.32)
        cv2.putText(frame, text, (tx + 2, ty + 2), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.4, (90, 120, 255), 3, cv2.LINE_AA)

        # Hint: show the target shape outline
        if self._target:
            hint = f"Hint: draw a {self._target}"
            hs = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, hint, (cx - hs[0] // 2, int(h * 0.46)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 240), 2, cv2.LINE_AA)
            self._draw_target_hint(frame, cx, int(h * 0.62), self._target)

    @staticmethod
    def _draw_stars(frame, cx, cy, stars, reveal=1.0):
        spacing = 70
        start_x = cx - spacing
        for i in range(3):
            sx = start_x + i * spacing
            filled = i < stars and reveal > (i + 1) / 4
            color = (60, 220, 255) if filled else (90, 90, 100)
            _star_poly(frame, sx, cy, 26 if filled else 20, color, filled)

    @staticmethod
    def _draw_target_hint(frame, cx, cy, target):
        color = (180, 180, 200)
        if target in ("circle", "ellipse"):
            cv2.circle(frame, (cx, cy), 35, color, 2, cv2.LINE_AA)
        elif target == "triangle":
            pts = np.array([(cx, cy - 35), (cx - 35, cy + 30), (cx + 35, cy + 30)], np.int32)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        elif target in ("square", "rectangle"):
            cv2.rectangle(frame, (cx - 40, cy - 30), (cx + 40, cy + 30), color, 2, cv2.LINE_AA)
        elif target == "star":
            _star_poly(frame, cx, cy, 35, color, False)
        elif target in ("line", "arrow"):
            cv2.arrowedLine(frame, (cx - 45, cy), (cx + 45, cy), color, 2, cv2.LINE_AA, tipLength=0.25)


def _star_poly(frame, cx, cy, r, color, filled):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((int(cx + rr * math.cos(ang)), int(cy + rr * math.sin(ang))))
    arr = np.array(pts, np.int32)
    if filled:
        cv2.fillPoly(frame, [arr], color, cv2.LINE_AA)
    else:
        cv2.polylines(frame, [arr], True, color, 2, cv2.LINE_AA)
