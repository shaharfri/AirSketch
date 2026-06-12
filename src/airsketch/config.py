from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class Backend(Enum):
    MEDIAPIPE = "mediapipe"
    OPENCV_GEOMETRIC = "opencv_geometric"
    OPENVINO_CPU = "openvino_cpu"
    OPENVINO_GPU = "openvino_gpu"
    OPENVINO_NPU = "openvino_npu"


class DrawingMode(Enum):
    IDLE = "idle"
    DRAWING = "drawing"
    LOCKED = "locked"
    RECOGNIZED = "recognized"
    EFFECT = "effect"
    COOLDOWN = "cooldown"


class AppMode(Enum):
    """Top-level application mode."""
    AIRNOTES = "airnotes"   # multi-diagram notebook (default)
    LEGACY = "legacy"       # original single-shape recognize-then-effect loop


@dataclass
class RecognitionResult:
    label: str
    confidence: float
    explanation: str


@dataclass
class HandResult:
    """Returned by HandTracker.process()."""
    visible: bool
    fingertip: Tuple[float, float] | None = None       # index tip (landmark 8), pixel coords
    thumb_tip: Tuple[float, float] | None = None       # landmark 4, pixel coords
    wrist: Tuple[float, float] | None = None           # landmark 0, pixel coords
    hand_size: float = 1.0                             # reference pixel scale
    confidence: float = 0.0
    landmarks: List[Tuple[float, float]] | None = None # full 21-landmark list, pixel coords (2D)
    # 3D world landmarks from MediaPipe (x, y, z in meters, relative to hand center).
    # Used for orientation-invariant gesture detection.
    world_landmarks: List[Tuple[float, float, float]] | None = None


@dataclass
class AppConfig:
    # --- Application mode ---
    app_mode: AppMode = AppMode.AIRNOTES

    # --- Language ---
    # "en" (default) or "he" (Hebrew). Affects the OpenVINO-native AI paths:
    # Whisper STT (voice commands + dictation), the board-understanding LLM
    # prompt, and voice-command keyword matching. Board OCR has no OpenVINO
    # Hebrew model, so it stays English regardless.
    language: str = "en"                       # en | he

    # --- Camera ---
    camera_index: int = 0
    video_path: str | None = None
    window_name: str = "AirSketch"
    frame_width: int = 1280
    frame_height: int = 720
    rotate_180: bool = True
    mirror: bool = False

    # --- Backends ---
    hand_backend: Backend = Backend.MEDIAPIPE
    recognizer_backend: Backend = Backend.OPENCV_GEOMETRIC

    hand_detection_confidence: float = 0.7
    hand_tracking_confidence: float = 0.6

    # Hand-tracking implementation: "openvino" (BlazePalm + landmark on the
    # OpenVINO runtime, Intel CPU/GPU/NPU — DEFAULT, validated live) or
    # "mediapipe" (TFLite/CPU). If the OpenVINO models fail to load,
    # create_hand_tracker() automatically falls back to MediaPipe.
    hand_tracker_backend: str = "openvino"    # openvino | mediapipe
    hand_device: str = "AUTO"                 # AUTO | CPU | GPU | NPU (openvino only)
    hand_debug: bool = False                  # overlay OV hand-tracking diagnostics

    # --- Drawing / trail ---
    trail_max_points: int = 1000
    trail_smoothing_window: int = 5
    trail_color: Tuple[int, int, int] = (0, 255, 200)
    trail_min_distance: float = 3.0
    drawing_start_threshold: float = 8.0
    hand_lost_grace_frames: int = 5

    # --- AirNotes — gesture & diagram lifecycle ---
    # gesture_mode: "point" (index-only extended) or "pinch" (thumb + index touch)
    gesture_mode: str = "point"
    point_confirm_frames: int = 3
    pinch_threshold: float = 0.30
    pinch_release_threshold: float = 0.45
    pinch_confirm_frames: int = 3

    # When a stroke ends, drop the last N points — they're typically transition
    # artifacts from the finger curling into a fist.
    stroke_tail_trim: int = 4
    # Snap stroke to a clean primitive on pen-up if confidence ≥ this threshold.
    # Per-primitive overrides apply (e.g. arrow needs higher confidence).
    # Disabled by default — freehand drawing is kept as-drawn (more abstract).
    # Recognition still works: the judge / analyzer classify raw strokes at
    # judge-time. Re-enable with --snap for the clean snap-to-shape behavior.
    live_snap_min_confidence: float = 0.78
    live_snap_arrow_min_confidence: float = 0.85
    live_snap_enabled: bool = False
    diagram_pause_seconds: float = 3.0       # idle time before auto-finalize
    sidebar_width: int = 280                 # px reserved on the right for thumbnails
    thumbnail_size: int = 140
    canvas_render_size: int = 512            # square canvas size sent to VLM

    # --- Auto-closure detection (legacy mode) ---
    closure_distance_threshold: float = 30.0
    closure_ignore_recent_points: int = 40
    closure_min_points: int = 60
    cooldown_frames: int = 25

    # --- Effects (legacy) ---
    effect_duration_frames: int = 120
    effect_fps_target: int = 30

    # --- CNN sketch classifier (Skysketch's Quick-Draw CNN) ---
    # Loaded automatically if the trained model exists at cnn_model_path.
    # To train: run `python -m training.train_sketch_cnn` after `python -m training.download_quickdraw`.
    cnn_enabled: bool = True
    cnn_device: str = "AUTO"                 # AUTO | CPU | GPU | NPU
    cnn_model_path: str = "models/sketch_classifier.xml"

    # --- VLM (Qwen2-VL via OpenVINO GenAI) ---
    # Off by default so the app starts instantly with NO downloads.
    # Enable with --vlm (will download ~1.7 GB on first run) or
    # --vlm-offline (uses cached weights only, never touches the network).
    vlm_enabled: bool = False
    vlm_offline_only: bool = False           # if True, never download
    # Community-converted Qwen2-VL-2B-Instruct OpenVINO IR INT4 (~1.7 GB).
    # Alternatives:
    #   helenai/Qwen2-VL-2B-Instruct-ov-int4
    #   helenai/Qwen2.5-VL-7B-Instruct-ov-int4  (heavier, better quality)
    #   OpenVINO/Qwen2-VL-7B-Instruct-fp16-ov   (official org, FP16, 7B)
    vlm_model_id: str = "cydxg/Qwen2-VL-2B-Instruct-OpenVINO-INT4"
    vlm_device: str = "AUTO"                 # AUTO | CPU | GPU | NPU
    vlm_max_tokens: int = 150
    vlm_fallback_to_local: bool = True
    vlm_timeout_seconds: float = 60.0
    vlm_model_cache_dir: str = "models"

    # --- Board capture (Phase 3) — PP-OCR on OpenVINO ---
    # Whiteboard transcription in the classroom. Independent of the VLM above.
    board_enabled: bool = False
    ocr_device: str = "CPU"                  # CPU | GPU | NPU | AUTO

    # --- Board understanding (Phase B) — Qwen2.5-3B Instruct on OpenVINO GenAI ---
    # Summarizes / structures the OCR transcription. Implies board_enabled.
    board_llm_enabled: bool = False
    board_llm_offline_only: bool = False     # if True, never download
    board_llm_model_id: str = "EmbeddedLLM/Qwen2.5-3B-Instruct-int4-sym-ov"
    llm_device: str = "CPU"                  # CPU | GPU | NPU | AUTO

    # --- Teacher voice — speaker recognition (WeSpeaker on OpenVINO) ---
    # Enroll the teacher's voice ("say some words"), then only the teacher's
    # spoken dictation is transcribed into the lesson notes. Implies --voice.
    teacher_voice_enabled: bool = False
    speaker_device: str = "CPU"              # CPU | GPU | NPU | AUTO
    speaker_threshold: float = 0.5           # cosine acceptance threshold
    speaker_profile_path: str = "models/teacher_voice.json"
    # Seconds to wait for the microphone input stream to open during the startup
    # probe. Bumped from 3s — a VDI's redirected mic can be slow to open; raise it
    # further (e.g. --mic-timeout 10) if voice stays unavailable on a working mic.
    mic_probe_timeout: float = 6.0
    # Whisper model size for voice/dictation (OpenVINO IR; all run on OpenVINO):
    # "base" (default, fast, in models/whisper-base-ov) or "small" (more accurate,
    # esp. Hebrew, ~0.5 GB, in models/whisper-small-ov).
    whisper_model: str = "base"              # base | small

    # --- Output ---
    output_dir: str = "outputs"

    @property
    def backend_label(self) -> str:
        return f"backend: {self.hand_backend.value} + {self.recognizer_backend.value}"
