"""Render Hebrew (and other non-ASCII) text onto OpenCV BGR frames.

OpenCV's ``cv2.putText`` uses Hershey vector fonts that have **no Hebrew glyphs
and no right-to-left support**, so Hebrew comes out as missing boxes in logical
(reversed) order. This module renders such text with Pillow + a system TrueType
font that has Hebrew glyphs, applying a lightweight RTL reordering.

Hebrew is **non-cursive** (letters don't join/shape like Arabic), so correct
display only needs *reordering* to visual order — no glyph reshaping and no
external bidi dependency. The reorderer handles embedded Latin/number runs
(e.g. "סבב 3") by keeping those runs left-to-right.

Text is rasterized to a small RGBA sprite and alpha-blended onto the frame, so
there's no costly full-frame PIL round-trip per call.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False

# Candidate Hebrew-capable fonts (Windows first, then common Linux paths).
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/david.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]

_HEBREW_LO, _HEBREW_HI = "֐", "׿"   # Hebrew Unicode block


def has_hebrew(text: str) -> bool:
    return any(_HEBREW_LO <= c <= _HEBREW_HI for c in (text or ""))


@lru_cache(maxsize=1)
def _font_path() -> Optional[str]:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


@lru_cache(maxsize=64)
def _get_font(px: int):
    if not _HAS_PIL:
        return None
    path = _font_path()
    if path is None:
        return None
    try:
        return ImageFont.truetype(path, px)
    except Exception:
        return None


def font_px(scale: float) -> int:
    """Approximate the pixel size matching a cv2 Hershey `scale`."""
    return max(11, int(round(scale * 34)))


def to_visual(text: str) -> str:
    """Reorder a base-RTL logical string into visual order for a LTR rasterizer.

    Standard trick: reverse the whole string (RTL), then flip back any runs of
    ASCII alphanumerics (+ common adjacent punctuation) so Latin/numbers read
    left-to-right. Hebrew needs no glyph shaping, only this reordering.
    """
    rev = text[::-1]
    out = []
    i, n = 0, len(rev)
    while i < n:
        c = rev[i]
        if c.isascii() and c.isalnum():
            j = i
            while j < n and rev[j].isascii() and (rev[j].isalnum() or rev[j] in ".:%/+-"):
                j += 1
            out.append(rev[i:j][::-1])   # un-reverse this LTR run
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def measure(text: str, scale: float) -> Tuple[int, int]:
    """(width, height) in px of `text` at `scale`, matching what draw() renders.
    Falls back to a rough estimate if PIL/font is unavailable."""
    font = _get_font(font_px(scale))
    if font is None:
        return (len(text) * int(scale * 18), int(scale * 30))
    l, t, r, b = font.getbbox(to_visual(text))
    return (max(1, r - l), max(1, b - t))


def _alpha_blit(frame_bgr: np.ndarray, rgba: np.ndarray, x0: int, y0: int) -> None:
    """Alpha-blend an RGBA sprite onto a BGR frame at (x0, y0), clipped to bounds."""
    fh, fw = frame_bgr.shape[:2]
    sh, sw = rgba.shape[:2]
    # clip
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(fw, x0 + sw), min(fh, y0 + sh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx0, sy0 = dx0 - x0, dy0 - y0
    sprite = rgba[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
    rgb = sprite[:, :, :3].astype(np.float32)
    bgr = rgb[:, :, ::-1]   # RGB -> BGR
    region = frame_bgr[dy0:dy1, dx0:dx1].astype(np.float32)
    frame_bgr[dy0:dy1, dx0:dx1] = (alpha * bgr + (1 - alpha) * region).astype(np.uint8)


def draw_text(frame_bgr: np.ndarray, text: str, pos: Tuple[int, int],
              scale: float = 0.6, color: Tuple[int, int, int] = (255, 255, 255),
              thickness: int = 1) -> bool:
    """Draw `text` with a dark shadow at `pos` (cv2-style baseline-left origin).

    Returns True if rendered via PIL, False if it couldn't (no PIL/font) — so the
    caller can fall back to cv2.putText.
    """
    font = _get_font(font_px(scale))
    if font is None:
        return False
    visual = to_visual(text)
    ascent, descent = font.getmetrics()
    l, t, r, b = font.getbbox(visual)
    w = max(1, r - l)
    pad = 3
    img = Image.new("RGBA", (w + 2 * pad, ascent + descent + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rgb = (int(color[2]), int(color[1]), int(color[0]))   # BGR -> RGB
    d.text((pad - l + 1, pad + 1), visual, font=font, fill=(0, 0, 0, 255))   # shadow
    d.text((pad - l, pad), visual, font=font, fill=(rgb[0], rgb[1], rgb[2], 255))
    arr = np.array(img)
    x0 = int(pos[0])
    y0 = int(pos[1]) - ascent - pad   # align baseline at pos[1]
    _alpha_blit(frame_bgr, arr, x0, y0)
    return True
