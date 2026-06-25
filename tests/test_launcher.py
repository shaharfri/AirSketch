"""Tests for the launcher's pure flag-building logic (no GUI / no display)."""
import importlib.util
import os

import pytest

# launcher.py lives at the project root, not in the package — load it directly.
_LAUNCHER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "launcher.py")
_spec = importlib.util.spec_from_file_location("airsketch_launcher", _LAUNCHER)
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)
build_args = launcher.build_args


class TestDefaults:
    def test_classroom_default(self):
        # Default-ish classroom run: openvino hand backend, AUTO devices, rotate on.
        args = build_args({"mode": "classroom"})
        assert "--classroom" in args
        assert args[args.index("--theme") + 1] == "geometry"
        assert args[args.index("--hand-backend") + 1] == "openvino"
        # AUTO devices are the defaults -> not emitted
        assert "--hand-device" not in args
        assert "--cnn-device" not in args
        # rotate default True -> no --no-rotate
        assert "--no-rotate" not in args
        assert "--mirror" not in args

    def test_notebook_has_no_classroom_or_theme(self):
        args = build_args({"mode": "notebook"})
        assert "--classroom" not in args
        assert "--theme" not in args


class TestCameraOrientation:
    def test_camera_index_emitted_when_nonzero(self):
        assert build_args({"camera": 2}).count("--camera") == 1
        assert build_args({"camera": 2})[build_args({"camera": 2}).index("--camera") + 1] == "2"

    def test_camera_zero_omitted(self):
        assert "--camera" not in build_args({"camera": 0})

    def test_rotate_off_emits_no_rotate(self):
        assert "--no-rotate" in build_args({"rotate180": False})

    def test_mirror(self):
        assert "--mirror" in build_args({"mirror": True})


class TestHand:
    def test_openvino_device_emitted_when_not_auto(self):
        a = build_args({"hand_backend": "openvino", "hand_device": "GPU"})
        assert a[a.index("--hand-device") + 1] == "GPU"

    def test_openvino_auto_device_omitted(self):
        assert "--hand-device" not in build_args(
            {"hand_backend": "openvino", "hand_device": "AUTO"})

    def test_mediapipe_ignores_device_and_debug(self):
        a = build_args({"hand_backend": "mediapipe", "hand_device": "GPU",
                        "hand_debug": True})
        assert a[a.index("--hand-backend") + 1] == "mediapipe"
        assert "--hand-device" not in a       # device is OV-only
        assert "--hand-debug" not in a        # debug is OV-only

    def test_hand_debug(self):
        assert "--hand-debug" in build_args(
            {"hand_backend": "openvino", "hand_debug": True})


class TestCNN:
    def test_no_cnn_wins_over_device(self):
        a = build_args({"no_cnn": True, "cnn_device": "GPU"})
        assert "--no-cnn" in a
        assert "--cnn-device" not in a

    def test_cnn_device_when_not_auto(self):
        a = build_args({"cnn_device": "CPU"})
        assert a[a.index("--cnn-device") + 1] == "CPU"


class TestFeatures:
    def test_voice_and_teacher(self):
        a = build_args({"voice": True, "teacher_voice": True})
        assert "--voice" in a and "--teacher-voice" in a

    def test_board_and_understand(self):
        a = build_args({"board": True, "understand": True})
        assert "--board" in a and "--understand" in a

    def test_ocr_device_only_when_board_and_nondefault(self):
        assert "--ocr-device" not in build_args({"ocr_device": "GPU"})  # board off
        a = build_args({"board": True, "ocr_device": "GPU"})
        assert a[a.index("--ocr-device") + 1] == "GPU"
        assert "--ocr-device" not in build_args({"board": True, "ocr_device": "CPU"})  # default

    def test_llm_device_only_when_understand_and_nondefault(self):
        a = build_args({"understand": True, "llm_device": "GPU"})
        assert a[a.index("--llm-device") + 1] == "GPU"
        assert "--llm-device" not in build_args({"board": True, "llm_device": "GPU"})  # not understand

    def test_vlm_notebook_only(self):
        assert "--vlm" not in build_args({"mode": "classroom", "vlm": True})
        a = build_args({"mode": "notebook", "vlm": True, "vlm_device": "GPU"})
        assert "--vlm" in a and a[a.index("--vlm-device") + 1] == "GPU"

    def test_snap(self):
        assert "--snap" in build_args({"snap": True})

    def test_mic_timeout_emitted_only_when_voice_and_nondefault(self):
        # default 6 -> not emitted; no voice -> not emitted
        assert "--mic-timeout" not in build_args({"voice": True, "mic_timeout": 6})
        assert "--mic-timeout" not in build_args({"mic_timeout": 12})  # voice off
        a = build_args({"voice": True, "mic_timeout": 12})
        assert a[a.index("--mic-timeout") + 1] == "12"
        a2 = build_args({"teacher_voice": True, "mic_timeout": 10})
        assert a2[a2.index("--mic-timeout") + 1] == "10"

    def test_whisper_model_emitted_only_when_voice_and_nonbase(self):
        assert "--whisper-model" not in build_args({"voice": True, "whisper_model": "base"})
        assert "--whisper-model" not in build_args({"whisper_model": "small"})  # voice off
        a = build_args({"voice": True, "whisper_model": "small"})
        assert a[a.index("--whisper-model") + 1] == "small"

    def test_voice_device_only_when_voice_and_nondefault(self):
        assert "--voice-device" not in build_args({"voice": True, "voice_device": "CPU"})
        assert "--voice-device" not in build_args({"voice_device": "GPU"})  # voice off
        a = build_args({"voice": True, "voice_device": "GPU"})
        assert a[a.index("--voice-device") + 1] == "GPU"

    def test_speaker_device_only_when_teacher_and_nondefault(self):
        assert "--speaker-device" not in build_args({"teacher_voice": True, "speaker_device": "CPU"})
        assert "--speaker-device" not in build_args({"speaker_device": "GPU"})  # teacher off
        a = build_args({"teacher_voice": True, "speaker_device": "NPU"})
        assert a[a.index("--speaker-device") + 1] == "NPU"

    def test_ocr_llm_vlm_devices_from_gui(self):
        a = build_args({"board": True, "ocr_device": "GPU"})
        assert a[a.index("--ocr-device") + 1] == "GPU"
        b = build_args({"understand": True, "llm_device": "GPU"})
        assert b[b.index("--llm-device") + 1] == "GPU"
        c = build_args({"mode": "notebook", "vlm": True, "vlm_device": "GPU"})
        assert c[c.index("--vlm-device") + 1] == "GPU"


class TestWindowedStdStreams:
    """A PyInstaller --windowed exe has sys.stdout/stderr == None; the app must
    not crash on shutdown flush. Guards both the launcher and the app side."""

    def test_ensure_std_streams_installs_flushable(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdout", None)
        monkeypatch.setattr("sys.stderr", None)
        launcher._ensure_std_streams(str(tmp_path))
        import sys
        assert sys.stdout is not None and sys.stderr is not None
        sys.stdout.flush()   # must not raise
        sys.stderr.flush()

    def test_ensure_std_streams_noop_when_present(self, monkeypatch):
        import io
        sentinel = io.StringIO()
        monkeypatch.setattr("sys.stdout", sentinel)
        monkeypatch.setattr("sys.stderr", sentinel)
        launcher._ensure_std_streams(".")
        import sys
        assert sys.stdout is sentinel and sys.stderr is sentinel  # untouched

    def test_app_std_flush_none_safe(self, monkeypatch):
        from airsketch.classroom.app import _std_flush
        monkeypatch.setattr("sys.stdout", None)
        monkeypatch.setattr("sys.stderr", None)
        _std_flush()   # must not raise


def test_args_round_trip_through_main_parser():
    """The flags the launcher emits must actually parse in airsketch.main."""
    import sys
    from airsketch.main import parse_args
    opts = {"mode": "classroom", "theme": "objects", "camera": 1, "mirror": True,
            "rotate180": False, "hand_backend": "openvino", "hand_device": "GPU",
            "hand_debug": True, "voice": True, "teacher_voice": True, "board": True,
            "understand": True, "ocr_device": "GPU", "llm_device": "GPU", "snap": True}
    flags = build_args(opts)
    old = sys.argv
    try:
        sys.argv = ["airsketch"] + flags
        cfg = parse_args()
    finally:
        sys.argv = old
    assert cfg.launch_classroom is True
    assert cfg.classroom_theme == "objects"
    assert cfg.hand_tracker_backend == "openvino"
    assert cfg.hand_device == "GPU"
    assert cfg.rotate_180 is False
    assert cfg.mirror is True
    assert cfg.board_enabled is True
    assert cfg.board_llm_enabled is True
    assert cfg.teacher_voice_enabled is True
