"""Tests for VoiceController graceful degradation (mocked audio backend).

Covers the three VDI failure modes:
  - no mic: opening an input stream raises
  - blocking mic: opening an input stream hangs (must time out, not freeze)
  - working mic but stream-open fails at record time
"""
import sys
import time
import types

import pytest

from airsketch.classroom.voice_controller import VoiceController


def _install_fake_sounddevice(monkeypatch, mode: str):
    """mode: 'ok' | 'raise' | 'block'. Fakes the sd.rec/sd.wait/sd.stop path."""
    import numpy as np
    fake = types.ModuleType("sounddevice")

    def rec(frames, **k):
        if mode == "raise":
            raise RuntimeError("PortAudio: no input device")
        if mode == "block":
            time.sleep(30)  # simulate a hung mic open
        return np.zeros((frames, 1), dtype="float32")

    fake.rec = rec
    fake.wait = lambda *a, **k: None
    fake.stop = lambda *a, **k: None
    # Some code paths still import these; keep harmless stubs.
    fake.query_devices = lambda kind=None: {"max_input_channels": 1}

    class _Stream:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass
    fake.InputStream = _Stream

    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


class TestProbe:
    def test_no_mic_raises_unavailable(self, monkeypatch):
        _install_fake_sounddevice(monkeypatch, "raise")
        vc = VoiceController()
        assert not vc.available

    def test_blocking_mic_times_out_unavailable(self, monkeypatch):
        _install_fake_sounddevice(monkeypatch, "block")
        t0 = time.time()
        # Use a short timeout by patching the default via a subclass call
        vc = VoiceController()
        elapsed = time.time() - t0
        assert not vc.available
        # Must NOT hang indefinitely — probe times out (default 3s, allow margin)
        assert elapsed < 6.0

    def test_toggle_noop_when_unavailable(self, monkeypatch):
        _install_fake_sounddevice(monkeypatch, "raise")
        vc = VoiceController()
        vc.toggle()  # should not raise
        assert not vc.is_recording


class TestWorkingMicButRecorderFails:
    def test_runtime_start_failure_disables(self, monkeypatch):
        _install_fake_sounddevice(monkeypatch, "ok")
        import airsketch.voice as voice_mod

        class FakeRecorder:
            def __init__(self, *a, **k):
                self.is_transcribing = False
                self.text = ""
            def start(self):
                raise RuntimeError("PortAudio: cannot open input stream")
            def stop(self): pass
            def close(self): pass

        monkeypatch.setattr(voice_mod, "VoiceRecorder", FakeRecorder)
        vc = VoiceController()
        assert vc.available           # probe (fake InputStream) opened fine
        vc.toggle()                   # recorder.start() raises -> caught
        assert not vc.available       # disabled
        assert not vc.is_recording
