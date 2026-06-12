"""Compositing video frame with canvas overlay, state badges, and transition UI."""

from __future__ import annotations

import cv2
import numpy as np

# Color palette (BGR format for OpenCV)
_BLUE = (255, 158, 74)      # #4A9EFF
_GREEN = (118, 230, 0)      # #00E676
_PURPLE = (252, 134, 187)   # #BB86FC
_ORANGE = (0, 165, 255)     # #FFA500
_WHITE = (255, 255, 255)
_DARK = (30, 30, 30)

_RED = (0, 0, 255)

_STATE_STYLES = {
    "idle": ("READY", _BLUE),
    "sketching": ("DRAWING", _GREEN),
    "analyzing": ("RESULT", _PURPLE),
}


def compose(
    frame: np.ndarray,
    canvas: np.ndarray,
    state: str = "",
    status_text: str = "",
    landmark_point: tuple[int, int] | None = None,
    hold_progress: float = 0.0,
    transition_text: str = "",
    transition_hint: str = "",
    llm_text: str = "",
    recording: bool = False,
    effect_mode: str = "",
    gen_status: str = "",
) -> np.ndarray:
    """Overlay canvas drawing on the camera frame with full UI."""
    display = frame.copy()

    # Blend canvas onto frame
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
    if np.any(mask):
        # In analyzing state, dim the drawing slightly
        alpha = 0.6 if state == "analyzing" else 0.8
        blended = cv2.addWeighted(frame, 1.0 - alpha, canvas, alpha, 0)
        display[mask] = blended[mask]

    # Fingertip cursor (double ring for visibility)
    if landmark_point is not None:
        cv2.circle(display, landmark_point, 10, (0, 180, 180), 2)
        cv2.circle(display, landmark_point, 5, (0, 255, 255), -1)

    # Transition banner (takes priority when active)
    if hold_progress > 0.0 and transition_text:
        _draw_transition_banner(display, hold_progress, transition_text, transition_hint)
    elif status_text:
        # Status text with dark background pill for readability
        _draw_text_with_bg(display, status_text, (10, 55), 0.7,
                           _WHITE if state != "analyzing" else _PURPLE)

    # LLM response text (below status, during analyzing)
    if llm_text:
        _draw_text_with_bg(display, llm_text, (10, 85), 0.5, (200, 200, 200), max_width=600)

    # State badge (always visible, top-left)
    if state:
        _draw_state_badge(display, state)

    # Voice recording indicator (top-right)
    if recording:
        _draw_recording_indicator(display)

    # Image generation status (bottom-center)
    if gen_status:
        _draw_text_with_bg(display, gen_status, (10, display.shape[0] - 20), 0.55, _ORANGE)

    # Effect mode indicator (top-right, below recording)
    if effect_mode:
        h = display.shape[0]
        y_pos = 30 if not recording else 55
        w = display.shape[1]
        _draw_text_with_bg(display, f"FX: {effect_mode}", (w - 150, y_pos), 0.45, _GREEN)

    return display


def _draw_recording_indicator(display: np.ndarray) -> None:
    """Draw a red pulsing REC dot in the top-right corner."""
    h, w = display.shape[:2]
    x, y = w - 80, 20
    cv2.circle(display, (x, y), 8, _RED, -1)
    cv2.putText(display, "REC", (x + 14, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _RED, 1, cv2.LINE_AA)


def _draw_text_with_bg(
    display: np.ndarray, text: str, pos: tuple[int, int],
    scale: float, color: tuple[int, int, int], max_width: int = 0,
) -> None:
    """Draw text with a semi-transparent dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2 if scale >= 0.7 else 1

    # Truncate if max_width specified
    if max_width > 0:
        while True:
            size = cv2.getTextSize(text, font, scale, thickness)[0]
            if size[0] <= max_width or len(text) <= 3:
                break
            text = text[:-4] + "..."

    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x, y = pos
    pad = 6

    # Dark background pill
    overlay = display.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - text_size[1] - pad),
                  (x + text_size[0] + pad, y + pad),
                  _DARK, -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Text
    cv2.putText(display, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_state_badge(display: np.ndarray, state: str) -> None:
    """Draw a colored pill badge showing current state."""
    label, color = _STATE_STYLES.get(state, ("---", (128, 128, 128)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    text_size = cv2.getTextSize(label, font, scale, thickness)[0]

    pad_x, pad_y = 14, 8
    w = text_size[0] + pad_x * 2
    h = text_size[1] + pad_y * 2
    x, y = 10, 10

    # Shadow
    overlay = display.copy()
    cv2.rectangle(overlay, (x + 2, y + 2), (x + w + 2, y + h + 2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

    # Filled badge
    overlay = display.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

    # Text centered in pill
    tx = x + pad_x
    ty = y + pad_y + text_size[1]
    cv2.putText(display, label, (tx, ty), font, scale, _WHITE, thickness, cv2.LINE_AA)


def _draw_transition_banner(
    display: np.ndarray, progress: float, message: str, hint: str
) -> None:
    """Draw centered transition banner with circular progress arc."""
    h, w = display.shape[:2]

    # Semi-transparent dark backdrop
    overlay = display.copy()
    banner_h = 110
    y1 = (h - banner_h) // 2
    y2 = y1 + banner_h
    cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    # Circular progress arc
    center_x = w // 2
    center_y = h // 2
    arc_x = center_x - 70
    _draw_arc_progress(display, (arc_x, center_y), 25, progress)

    # Transition message
    text_x = arc_x + 40
    cv2.putText(display, message, (text_x, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, _WHITE, 2, cv2.LINE_AA)

    # Hint text (centered below)
    if hint:
        hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        hint_x = (w - hint_size[0]) // 2
        cv2.putText(display, hint, (hint_x, center_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Next state indicator
    next_state = ""
    if "Starting" in message:
        next_state = "-> Start Sketching"
    elif "Analyzing" in message:
        next_state = "-> Analyze Drawing"
    if next_state:
        ns_size = cv2.getTextSize(next_state, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        ns_x = (w - ns_size[0]) // 2
        cv2.putText(display, next_state, (ns_x, center_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREEN, 1, cv2.LINE_AA)


def _draw_arc_progress(
    display: np.ndarray, center: tuple[int, int], radius: int, progress: float
) -> None:
    """Draw a circular arc progress indicator."""
    cv2.circle(display, center, radius, (60, 60, 60), 3)

    if progress > 0:
        angle_end = int(360 * min(progress, 1.0))
        color = _GREEN if progress >= 1.0 else _ORANGE
        cv2.ellipse(display, center, (radius, radius), -90, 0, angle_end, color, 4)

    cv2.circle(display, center, 4, _WHITE, -1)
