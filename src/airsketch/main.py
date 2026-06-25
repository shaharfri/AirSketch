"""AirSketch — merged air-drawing app.

Real-time air whiteboard for video calls / classrooms:
 - Pinch (thumb + index together) to draw
 - Multiple strokes per diagram
 - Auto-finalize after a brief pause, or press N
 - Each diagram is analyzed by Qwen2-VL via OpenVINO (async, non-blocking)
 - Export a self-contained HTML notebook at the end
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from airsketch.config import AppConfig, AppMode
from airsketch.diagram_analyzer import create_analyzer
from airsketch.exporter import export_html, export_json
from airsketch.gesture_detector import IndexPointingDetector, PinchDetector
from airsketch.hand_tracker import create_hand_tracker
from airsketch.notebook import Notebook
from airsketch.shape_recognizer import ShapeRecognizer
from airsketch.stroke import DiagramStatus
from airsketch.utils import draw_neon_line, draw_panel, draw_text_with_shadow
from airsketch.video_source import VideoSource

# --- Status colours --------------------------------------------------------

STATUS_COLORS = {
    DiagramStatus.DRAWING:   (140, 140, 150),
    DiagramStatus.PENDING:   (255, 200, 60),
    DiagramStatus.ANALYZING: (0, 200, 255),
    DiagramStatus.DONE:      (0, 230, 120),
    DiagramStatus.FAILED:    (60, 90, 255),
}

STATUS_LABEL = {
    DiagramStatus.DRAWING:   "drawing",
    DiagramStatus.PENDING:   "pending",
    DiagramStatus.ANALYZING: "analyzing",
    DiagramStatus.DONE:      "done",
    DiagramStatus.FAILED:    "failed",
}


# --- CLI / config ----------------------------------------------------------

def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(
        prog="airsketch",
        description="AirSketch — air-drawing notebook with primitive + CNN + Qwen-VL analysis"
    )
    p.add_argument("--classroom", action="store_true",
                   help="Launch the classroom challenge game instead of the notebook")
    p.add_argument("--lang", choices=("en", "he"), default=None,
                   help="Language for voice (Whisper), command keywords, and the "
                        "board-understanding LLM: en (default) or he (Hebrew). "
                        "Board OCR stays English (no OpenVINO Hebrew model).")
    p.add_argument("--theme", choices=("geometry", "objects", "mixed"), default="geometry",
                   help="Classroom challenge theme (with --classroom)")
    p.add_argument("--voice", action="store_true",
                   help="Enable voice control in classroom mode (needs mic + Whisper)")
    p.add_argument("--board", action="store_true",
                   help="Enable whiteboard capture in classroom mode (PP-OCR on OpenVINO)")
    p.add_argument("--ocr-device", default=None,
                   help="OpenVINO device for board OCR: CPU | GPU | NPU | AUTO")
    p.add_argument("--understand", action="store_true",
                   help="Add the LLM that summarizes board text (implies --board; ~1.8 GB)")
    p.add_argument("--llm-device", default=None,
                   help="OpenVINO device for the understanding LLM: CPU | GPU | NPU | AUTO")
    p.add_argument("--voice-device", default=None,
                   help="OpenVINO device for voice STT (Whisper): CPU | GPU | NPU | AUTO")
    p.add_argument("--speaker-device", default=None,
                   help="OpenVINO device for speaker ID (WeSpeaker): CPU | GPU | NPU | AUTO")
    p.add_argument("--teacher-voice", action="store_true",
                   help="Speaker recognition: enroll (E) the teacher's voice, gate dictation to them (implies --voice)")
    p.add_argument("--speaker-threshold", type=float, default=None,
                   help="Cosine acceptance threshold for the teacher's voice (default 0.5)")
    p.add_argument("--mic-timeout", type=float, default=None,
                   help="Seconds to wait for the mic to open at startup (default 6). "
                        "Raise it (e.g. 10) if voice is unavailable on a slow/VDI mic.")
    p.add_argument("--whisper-model", choices=("base", "small"), default=None,
                   help="Whisper model (OpenVINO): 'base' (default, fast) or 'small' "
                        "(more accurate, esp. Hebrew; needs models/whisper-small-ov)")
    p.add_argument("--video", type=str, default=None,
                   help="Path to a video file (default: webcam)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--no-rotate", action="store_true", help="Disable 180-degree rotation")
    p.add_argument("--mirror", action="store_true", help="Enable horizontal mirror")
    p.add_argument("--vlm", action="store_true",
                   help="Enable Qwen2-VL titling (downloads ~1.7 GB on first run)")
    p.add_argument("--vlm-offline", action="store_true",
                   help="Enable Qwen2-VL only if already cached locally — never download")
    p.add_argument("--no-vlm", action="store_true",
                   help="Force-disable Qwen2-VL (default behavior; use LocalAnalyzer)")
    p.add_argument("--vlm-device", default=None,
                   help="OpenVINO device: AUTO | CPU | GPU | NPU")
    p.add_argument("--vlm-model", default=None,
                   help="HuggingFace repo id of the VLM model")
    p.add_argument("--hand-backend", choices=("mediapipe", "openvino"), default=None,
                   help="Hand tracking backend: 'openvino' (default; BlazePalm + landmark "
                        "on the OpenVINO runtime) or 'mediapipe' (TFLite/CPU fallback)")
    p.add_argument("--hand-device", default=None,
                   help="OpenVINO device for hand tracking: AUTO | CPU | GPU | NPU "
                        "(only with --hand-backend openvino)")
    p.add_argument("--hand-debug", action="store_true",
                   help="Overlay live OpenVINO hand-tracking diagnostics "
                        "(palm score / presence / state) for troubleshooting")
    p.add_argument("--no-cnn", action="store_true",
                   help="Disable Quick-Draw CNN classifier even if model is present")
    p.add_argument("--cnn-device", default=None,
                   help="OpenVINO device for CNN: AUTO | CPU | GPU | NPU")
    p.add_argument("--cnn-model", default=None,
                   help="Path to OpenVINO IR sketch-classifier model")
    p.add_argument("--pause-seconds", type=float, default=None,
                   help="Idle seconds before auto-finalizing a diagram")
    p.add_argument("--gesture", choices=("point", "pinch"), default=None,
                   help="Pen-down gesture: 'point' (index-only) or 'pinch'")
    p.add_argument("--snap", action="store_true",
                   help="Enable live snap-to-shape on pen-up (off by default)")
    p.add_argument("--no-snap", action="store_true",
                   help="Force-disable live snap-to-shape (default behavior)")
    p.add_argument("--no-tail-trim", action="store_true",
                   help="Don't trim trailing transition points at pen-up")
    args = p.parse_args()

    cfg = AppConfig()
    if args.lang:
        cfg.language = args.lang
    if args.video:
        cfg.video_path = args.video
    else:
        cfg.camera_index = args.camera
    if args.no_rotate:
        cfg.rotate_180 = False
    if args.mirror:
        cfg.mirror = True
    if args.vlm:
        cfg.vlm_enabled = True
        cfg.vlm_offline_only = False
    if args.vlm_offline:
        cfg.vlm_enabled = True
        cfg.vlm_offline_only = True
    if args.no_vlm:
        cfg.vlm_enabled = False
    if args.vlm_device:
        cfg.vlm_device = args.vlm_device.upper()
    if args.vlm_model:
        cfg.vlm_model_id = args.vlm_model
    if args.hand_backend:
        cfg.hand_tracker_backend = args.hand_backend
    if args.hand_device:
        cfg.hand_device = args.hand_device.upper()
    if args.hand_debug:
        cfg.hand_debug = True
    if args.no_cnn:
        cfg.cnn_enabled = False
    if args.cnn_device:
        cfg.cnn_device = args.cnn_device.upper()
    if args.cnn_model:
        cfg.cnn_model_path = args.cnn_model
    if args.pause_seconds is not None:
        cfg.diagram_pause_seconds = args.pause_seconds
    if args.gesture:
        cfg.gesture_mode = args.gesture
    if args.snap:
        cfg.live_snap_enabled = True
    if args.no_snap:
        cfg.live_snap_enabled = False
    if args.no_tail_trim:
        cfg.stroke_tail_trim = 0
    if args.board:
        cfg.board_enabled = True
    if args.ocr_device:
        cfg.ocr_device = args.ocr_device.upper()
    if args.understand:
        cfg.board_enabled = True            # LLM needs the OCR text
        cfg.board_llm_enabled = True
    if args.llm_device:
        cfg.llm_device = args.llm_device.upper()
    if args.voice_device:
        cfg.voice_device = args.voice_device.upper()
    if args.speaker_device:
        cfg.speaker_device = args.speaker_device.upper()
    if args.teacher_voice:
        cfg.teacher_voice_enabled = True    # speaker ID needs the mic + Whisper
    if args.speaker_threshold is not None:
        cfg.speaker_threshold = args.speaker_threshold
    if args.mic_timeout is not None:
        cfg.mic_probe_timeout = args.mic_timeout
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    # Stash classroom selection on the config (dynamic attrs)
    cfg.launch_classroom = args.classroom
    cfg.classroom_theme = args.theme
    cfg.classroom_voice = args.voice or args.teacher_voice
    return cfg


# --- Rendering helpers -----------------------------------------------------

def draw_status_panel(
    frame: np.ndarray,
    cfg: AppConfig,
    fps: float,
    pen_down: bool,
    notebook: Notebook,
    analyzer_name: str,
    device_label: str,
) -> None:
    h, w = frame.shape[:2]
    panel_w = 280
    draw_panel(frame, 10, 10, panel_w, 132, alpha=0.6)

    # Mode dot
    if pen_down:
        dot, mode_text = (0, 230, 120), "PEN DOWN"
    elif notebook.is_drawing:
        dot, mode_text = (0, 200, 255), "STROKE"
    else:
        dot, mode_text = (140, 140, 150), "READY"
    cv2.circle(frame, (28, 36), 7, dot, -1, cv2.LINE_AA)
    cv2.circle(frame, (28, 36), 8, (255, 255, 255), 1, cv2.LINE_AA)
    draw_text_with_shadow(frame, mode_text, (45, 41), 0.6, dot, thickness=2)

    draw_text_with_shadow(frame, f"FPS: {fps:.0f}", (22, 66), 0.5, (210, 210, 220))
    draw_text_with_shadow(
        frame, f"Diagrams: {notebook.diagram_count}", (22, 88),
        0.5, (210, 210, 220),
    )
    draw_text_with_shadow(
        frame, f"Strokes (current): {len(notebook.current.strokes)}",
        (22, 110), 0.45, (180, 180, 190),
    )
    draw_text_with_shadow(
        frame, f"VLM: {analyzer_name} on {device_label}", (22, 130),
        0.42, (140, 200, 255),
    )

    # Backend tag (top-right)
    backend = cfg.backend_label
    bs = cv2.getTextSize(backend, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    bp_w = bs[0] + 24
    draw_panel(frame, w - bp_w - 10 - cfg.sidebar_width, 10, bp_w, 30, alpha=0.45)
    draw_text_with_shadow(
        frame, backend, (w - bp_w + 2 - cfg.sidebar_width, 30),
        0.42, (180, 180, 190),
    )


def draw_sidebar(frame: np.ndarray, cfg: AppConfig, notebook: Notebook) -> None:
    """Render the right-edge sidebar with diagram thumbnails + titles."""
    h, w = frame.shape[:2]
    sw = cfg.sidebar_width
    sx0 = w - sw

    # Background panel for the whole sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (sx0, 0), (w, h), (12, 14, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.line(frame, (sx0, 0), (sx0, h), (60, 60, 70), 1, cv2.LINE_AA)

    draw_text_with_shadow(
        frame, "NOTEBOOK", (sx0 + 14, 28), 0.55, (0, 230, 200), thickness=2,
    )
    draw_text_with_shadow(
        frame, f"{notebook.diagram_count} diagram(s)",
        (sx0 + 14, 50), 0.42, (160, 160, 175),
    )

    if not notebook.diagrams:
        draw_text_with_shadow(
            frame, "Pinch to start drawing.", (sx0 + 14, 90),
            0.46, (140, 140, 160),
        )
        draw_text_with_shadow(
            frame, "Release & pause to", (sx0 + 14, 112), 0.42, (120, 120, 140),
        )
        draw_text_with_shadow(
            frame, "auto-finalize.", (sx0 + 14, 130), 0.42, (120, 120, 140),
        )
        return

    # Stack thumbnails from the bottom up, newest at top
    thumb_w = sw - 28
    item_h = thumb_w + 70
    y_cursor = 68
    max_y = h - 80

    for i, d in enumerate(reversed(notebook.diagrams), 1):
        if y_cursor + item_h > max_y:
            extra = len(notebook.diagrams) - (i - 1)
            draw_text_with_shadow(
                frame, f"+ {extra} earlier...", (sx0 + 14, y_cursor + 16),
                0.42, (160, 160, 180),
            )
            break

        idx = len(notebook.diagrams) - i + 1

        # Main thumbnail (single image — no separate "raw" inset since the
        # live-snapped strokes already are the clean version)
        main_thumb = d.thumbnail if d.thumbnail is not None else d.clean_thumbnail
        if main_thumb is not None:
            target = cv2.resize(main_thumb, (thumb_w, thumb_w))
            frame[y_cursor:y_cursor + thumb_w, sx0 + 14:sx0 + 14 + thumb_w] = target
            cv2.rectangle(
                frame,
                (sx0 + 14, y_cursor),
                (sx0 + 14 + thumb_w, y_cursor + thumb_w),
                (60, 60, 70), 1, cv2.LINE_AA,
            )

        # Status dot + label
        status_color = STATUS_COLORS.get(d.status, (140, 140, 150))
        cv2.circle(
            frame, (sx0 + 22, y_cursor + thumb_w + 14), 5,
            status_color, -1, cv2.LINE_AA,
        )
        title = "Analyzing..."
        if d.analysis:
            title = d.analysis.title
        elif d.status == DiagramStatus.FAILED:
            title = "Failed"
        elif d.status == DiagramStatus.PENDING:
            title = "Pending..."

        # Truncate title to fit
        title_disp = title if len(title) <= 22 else title[:21] + "…"
        draw_text_with_shadow(
            frame, f"{idx}. {title_disp}",
            (sx0 + 34, y_cursor + thumb_w + 19), 0.46, (220, 220, 230),
        )

        if d.analysis and d.analysis.tags:
            tags_txt = " · ".join(d.analysis.tags[:3])
            if len(tags_txt) > 28:
                tags_txt = tags_txt[:27] + "…"
            draw_text_with_shadow(
                frame, tags_txt, (sx0 + 22, y_cursor + thumb_w + 40),
                0.4, (130, 170, 220),
            )

        y_cursor += item_h


def draw_controls_bar(frame: np.ndarray, cfg: AppConfig) -> None:
    h, w = frame.shape[:2]
    if cfg.gesture_mode == "pinch":
        gesture_hint = "Pinch=Draw"
    else:
        gesture_hint = "Point=Draw"
    text = (f"{gesture_hint}   Hold SPACE=force pen   "
            "N=New   Z=Undo   C=Clear   "
            "R=Re-analyze   E=Export   S=Snap   Q/ESC=Quit")
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    bw = ts[0] + 24
    bx = max(10, (w - cfg.sidebar_width - bw) // 2)
    draw_panel(frame, bx, h - 32, bw, 24, alpha=0.55)
    draw_text_with_shadow(frame, text, (bx + 12, h - 14), 0.42, (200, 200, 215))


def draw_pen_banner(
    frame: np.ndarray, cfg: AppConfig, pen_down: bool, hand_visible: bool,
) -> None:
    """Big, unmissable PEN UP / PEN DOWN / NO HAND banner on the left edge."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = 10, 160, 250, 56
    if not hand_visible:
        bg, fg, label = (60, 60, 70), (255, 200, 60), "NO HAND"
    elif pen_down:
        bg, fg, label = (10, 60, 30), (60, 255, 130), "PEN DOWN"
    else:
        bg, fg, label = (40, 30, 30), (255, 130, 100), "PEN UP"
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), bg, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), fg, 2, cv2.LINE_AA)
    draw_text_with_shadow(frame, label, (bx + 18, by + 38), 1.0, fg, thickness=2)


def draw_finger_debug(
    frame: np.ndarray,
    hand,
    gesture,
    pen_down: bool,
    space_active: bool,
) -> None:
    """Show live finger angles + compactness + decision near the fingertip."""
    if not hand.visible or hand.fingertip is None:
        return
    diag = getattr(gesture, "diagnostics", {}) or {}
    if not diag or diag.get("reason"):
        return

    tx, ty = int(hand.fingertip[0]), int(hand.fingertip[1])
    h, w = frame.shape[:2]
    pw, ph = 260, 130
    px, py = tx + 24, ty + 24
    px = min(max(0, px), w - pw - 10)
    py = min(max(0, py), h - ph - 50)

    draw_panel(frame, px, py, pw, ph, alpha=0.65)

    if "index" in diag and isinstance(diag["index"], dict):
        coord_space = diag.get("coords", "?")
        idx = diag["index"]
        mid = diag["middle"]
        rng = diag["ring"]
        pky = diag["pinky"]
        open_palm = diag.get("open_palm", False)

        # Header
        draw_text_with_shadow(
            frame, f"GESTURE  ({coord_space})",
            (px + 10, py + 18), 0.42, (180, 200, 230),
        )

        # Per-finger: name, angle, compactness
        def row(name, ev, y):
            ang = ev["angle"]
            comp = ev["compact"]
            ang_color = (60, 255, 130) if ang >= 145 else (255, 180, 100) if ang >= 125 else (255, 130, 100)
            cmp_color = (60, 255, 130) if comp >= 0.85 else (255, 180, 100) if comp >= 0.72 else (255, 130, 100)
            draw_text_with_shadow(frame, f"{name}", (px + 10, y), 0.42, (200, 200, 215))
            draw_text_with_shadow(frame, f"{ang:5.1f}°", (px + 80, y), 0.42, ang_color)
            draw_text_with_shadow(frame, f"c={comp:.2f}", (px + 160, y), 0.42, cmp_color)

        row("idx", idx, py + 40)
        row("mid", mid, py + 58)
        row("rng", rng, py + 76)
        row("pky", pky, py + 94)

        # Verdict
        if space_active:
            verdict, color = "SPACE → pen DOWN", (255, 240, 100)
        elif open_palm:
            verdict, color = "open palm → no pen", (255, 130, 100)
        elif pen_down:
            verdict, color = "pointing → pen DOWN", (60, 255, 130)
        else:
            verdict, color = "not pointing → no pen", (180, 180, 200)
        draw_text_with_shadow(frame, verdict, (px + 10, py + 118), 0.45, color)
    else:
        draw_text_with_shadow(
            frame, "(pinch mode active)",
            (px + 10, py + 26), 0.45, (200, 220, 255),
        )


def draw_key_indicator(
    frame: np.ndarray, cfg: AppConfig, last_key: int, last_key_at: float,
) -> None:
    """Bottom-left flashing dot + label proving keyboard focus."""
    h, w = frame.shape[:2]
    age = time.time() - last_key_at
    has_focus_hint = last_key != 255 and age < 1.5

    # Always show focus instructions; flash when a key was just received
    x, y = 10, h - 80
    draw_panel(frame, x, y, 250, 36, alpha=0.55)
    if has_focus_hint:
        if 32 <= last_key <= 126:
            txt = f"key: '{chr(last_key)}'  ({last_key}) ✓"
        else:
            txt = f"key: code {last_key} ✓"
        cv2.circle(frame, (x + 18, y + 18), 7, (60, 255, 130), -1, cv2.LINE_AA)
        draw_text_with_shadow(frame, txt, (x + 35, y + 23), 0.5, (200, 255, 200))
    else:
        cv2.circle(frame, (x + 18, y + 18), 7, (90, 90, 110), 1, cv2.LINE_AA)
        draw_text_with_shadow(
            frame, "click here for key input",
            (x + 35, y + 23), 0.45, (160, 160, 175),
        )


def draw_current_strokes(
    frame: np.ndarray, cfg: AppConfig, notebook: Notebook,
) -> None:
    """Render the current diagram's strokes (completed + in-progress)."""
    color = cfg.trail_color
    for stroke in notebook.current.strokes:
        pts = [(int(x), int(y)) for x, y in stroke.points]
        if len(pts) >= 2:
            draw_neon_line(frame, pts, color, base_thickness=2)
    if notebook.current_stroke is not None:
        pts = [(int(x), int(y)) for x, y in notebook.current_stroke.points]
        if len(pts) >= 2:
            draw_neon_line(frame, pts, (0, 255, 255), base_thickness=2)


# --- Save snapshot ---------------------------------------------------------

def save_snapshot(frame: np.ndarray, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"airnotes_snap_{ts}.png")
    cv2.imwrite(path, frame)
    return path


# --- Export helpers --------------------------------------------------------

def do_export(notebook: Notebook, output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"airnotes_{ts}.html")
    json_path = os.path.join(output_dir, f"airnotes_{ts}.json")
    export_html(notebook.diagrams, html_path, session_label=f"Session {ts}")
    export_json(notebook.diagrams, json_path)
    return html_path, json_path


# --- Main loop -------------------------------------------------------------

def main() -> None:
    cfg = parse_args()

    # Branch to the classroom challenge game if requested
    if getattr(cfg, "launch_classroom", False):
        from airsketch.classroom.app import ClassroomApp
        ClassroomApp(
            cfg,
            theme=getattr(cfg, "classroom_theme", "geometry"),
            voice=getattr(cfg, "classroom_voice", False),
        ).run()
        return

    # Video source
    source = cfg.video_path if cfg.video_path else cfg.camera_index
    video = VideoSource(
        source, cfg.frame_width, cfg.frame_height,
        rotate_180=cfg.rotate_180, mirror=cfg.mirror,
    )
    if not video.open():
        print(f"ERROR: cannot open video source: {video.source_label}", file=sys.stderr)
        sys.exit(1)
    print(f"[video]    Opened {video.source_label}")

    # Hand tracker (MediaPipe or OpenVINO, per --hand-backend)
    tracker = create_hand_tracker(cfg)

    # Pen-down gesture detector
    if cfg.gesture_mode == "pinch":
        gesture = PinchDetector(
            pinch_threshold=cfg.pinch_threshold,
            release_threshold=cfg.pinch_release_threshold,
            confirm_frames=cfg.pinch_confirm_frames,
        )
        gesture_label = "pinch (thumb + index)"
    else:
        gesture = IndexPointingDetector(confirm_frames=cfg.point_confirm_frames)
        gesture_label = "point (index only)"

    def gesture_update(hand) -> bool:
        if isinstance(gesture, PinchDetector):
            return gesture.update(hand.thumb_tip, hand.fingertip, hand.hand_size)
        # IndexPointingDetector: prefer 3D world landmarks (orientation-invariant)
        return gesture.update(hand.landmarks, world_landmarks=hand.world_landmarks)

    # Analyzer (may take 10-30s on first load)
    print("[analyzer] Initializing analyzer...")
    analyzer = create_analyzer(cfg)
    analyzer_name = getattr(analyzer, "name", "unknown")
    device_label = (
        cfg.vlm_device if analyzer_name.startswith("openvino") else "local"
    )

    # Notebook
    recognizer = ShapeRecognizer()
    notebook = Notebook(
        analyzer=analyzer,
        recognizer=recognizer,
        pause_seconds=cfg.diagram_pause_seconds,
        canvas_render_size=cfg.canvas_render_size,
        thumbnail_size=cfg.thumbnail_size,
        tail_trim=cfg.stroke_tail_trim,
        live_snap_enabled=cfg.live_snap_enabled,
        live_snap_min_confidence=cfg.live_snap_min_confidence,
        live_snap_arrow_min_confidence=cfg.live_snap_arrow_min_confidence,
    )

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_TOPMOST, 1)
    try:
        cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_TOPMOST, 0)
    except cv2.error:
        pass
    print(f"[ready]    AirNotes running. Gesture: {gesture_label}")
    if cfg.gesture_mode == "point":
        print("[ready]    Point with INDEX FINGER only (others curled) to draw.")
        print("[ready]    Open hand or fist to lift the pen between strokes.")
    else:
        print("[ready]    Touch thumb + index together to draw.")
    print(f"[ready]    Auto-finalize after {cfg.diagram_pause_seconds:.1f}s pause.")
    print("[ready]    >>> Click the AirNotes window first to give it keyboard focus <<<")

    prev_t = time.perf_counter()
    fps = 0.0
    last_key = 255
    last_key_at = 0.0
    space_pen_down_until = 0.0
    SPACE_OVERRIDE_TTL = 0.25  # seconds — held while SPACE keeps repeating

    try:
        while True:
            ok, frame = video.read()
            if not ok or frame is None:
                if cfg.video_path:
                    break
                print("WARNING: frame read failed, retrying...", file=sys.stderr)
                continue

            now = time.perf_counter()
            dt = now - prev_t
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_t = now

            hand = tracker.process(frame)

            # ---- Drawing gesture (point or pinch) ----
            gesture_pen_down = gesture_update(hand) if hand.visible else False
            space_active = time.time() < space_pen_down_until
            pen_down = (gesture_pen_down or space_active) and hand.visible

            if hand.visible and pen_down and hand.fingertip is not None:
                if not notebook.is_drawing:
                    notebook.begin_stroke(hand.fingertip)
                else:
                    notebook.append_to_stroke(hand.fingertip)
                # Pen cursor (BIG so user sees pen-down state)
                tx, ty = int(hand.fingertip[0]), int(hand.fingertip[1])
                cv2.circle(frame, (tx, ty), 14, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 13, (0, 255, 100), 3, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                if notebook.is_drawing:
                    notebook.end_stroke()
                # Pen-up cursor (when finger visible but pen up)
                if hand.visible and hand.fingertip is not None:
                    tx, ty = int(hand.fingertip[0]), int(hand.fingertip[1])
                    cv2.circle(frame, (tx, ty), 10, (60, 60, 80), 2, cv2.LINE_AA)

            # Auto-finalize after idle pause
            notebook.check_pause_timeout()

            # Render strokes (completed + active)
            draw_current_strokes(frame, cfg, notebook)

            # HUD
            draw_status_panel(
                frame, cfg, fps, pen_down, notebook,
                analyzer_name, device_label,
            )
            draw_sidebar(frame, cfg, notebook)
            draw_controls_bar(frame, cfg)
            draw_pen_banner(frame, cfg, pen_down, hand.visible)
            draw_finger_debug(frame, hand, gesture, pen_down, space_active)
            draw_key_indicator(frame, cfg, last_key, last_key_at)

            cv2.imshow(cfg.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                last_key = key
                last_key_at = time.time()
                if key != 32:  # don't spam the log on repeating SPACE
                    key_repr = chr(key) if 32 <= key <= 126 else f"code={key}"
                    print(f"[key]      pressed: {key_repr}")
            if key == 32:
                # SPACE pressed (and auto-repeats while held on most platforms).
                # Extend the override window from "now".
                space_pen_down_until = time.time() + SPACE_OVERRIDE_TTL
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("n"), ord("N")):
                notebook.end_stroke()
                finalized = notebook.finalize_current_diagram()
                if finalized is not None:
                    print(f"[diagram]  Finalized diagram #{notebook.diagram_count}.")
                else:
                    print("[diagram]  Current diagram is empty, nothing to finalize.")
            elif key in (ord("c"), ord("C")):
                notebook.clear_current()
                print("[diagram]  Current diagram cleared.")
            elif key in (ord("z"), ord("Z")):
                if notebook.undo_last_stroke():
                    print("[diagram]  Undid last stroke.")
            elif key in (ord("r"), ord("R")):
                if notebook.reanalyze_last():
                    print(f"[analyzer] Re-analyzing diagram #{notebook.diagram_count}.")
                else:
                    print("[analyzer] Nothing to re-analyze.")
            elif key in (ord("e"), ord("E")):
                # Finalize anything pending, then export
                notebook.end_stroke()
                notebook.finalize_current_diagram()
                html_path, json_path = do_export(notebook, cfg.output_dir)
                print(f"[export]   HTML: {html_path}")
                print(f"[export]   JSON: {json_path}")
            elif key in (ord("s"), ord("S")):
                path = save_snapshot(frame, cfg.output_dir)
                print(f"[snapshot] Saved to {path}")

    except KeyboardInterrupt:
        pass
    finally:
        # Finalize any unfinished diagram before exporting
        try:
            notebook.end_stroke()
            notebook.finalize_current_diagram()
        except Exception as e:
            print(f"[shutdown] error finalizing: {e}", file=sys.stderr)

        # If there are pending analyses, wait briefly so the export captures them
        if notebook.has_pending_analysis:
            print("[shutdown] Waiting for pending analyses (max 30s)...")
            wait_start = time.time()
            while notebook.has_pending_analysis and (time.time() - wait_start) < 30:
                time.sleep(0.5)

        # Auto-export on exit if there are any diagrams
        if notebook.diagram_count > 0:
            try:
                html_path, json_path = do_export(notebook, cfg.output_dir)
                print(f"[export]   HTML: {html_path}")
                print(f"[export]   JSON: {json_path}")
            except Exception as e:
                print(f"[export]   failed: {e}", file=sys.stderr)

        # Release resources
        try:
            notebook.shutdown(wait=False)
        except Exception as e:
            print(f"[shutdown] notebook: {e}", file=sys.stderr)
        try:
            analyzer.release()
        except Exception as e:
            print(f"[shutdown] analyzer: {e}", file=sys.stderr)
        try:
            video.release()
        except Exception as e:
            print(f"[shutdown] video: {e}", file=sys.stderr)
        try:
            tracker.release()
        except Exception as e:
            print(f"[shutdown] tracker: {e}", file=sys.stderr)
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        print("[shutdown] AirNotes stopped.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # MediaPipe + OpenVINO spawn background threads that prevent clean shutdown
        os._exit(0)
