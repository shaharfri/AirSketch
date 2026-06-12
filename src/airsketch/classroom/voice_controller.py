"""VoiceController — robust wrapper around VoiceRecorder + command parsing.

Degrades gracefully: if sounddevice isn't installed, no mic is present, or the
Whisper model is missing, the controller reports `available = False` and the
classroom app keeps working with the keyboard.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Optional

from airsketch.classroom.voice_commands import Intent, parse_command

SAMPLE_RATE = 16000

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
_DEFAULT_WHISPER = _MODELS_DIR / "whisper-base-ov"


class VoiceController:
    """Toggle-to-record voice control with background transcription.

    Usage:
        vc = VoiceController(device="CPU")
        if vc.available:
            vc.toggle()              # start recording
            ...
            vc.toggle()              # stop -> transcribes in background
            intent = vc.poll()       # returns an Intent once transcription lands, else None
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "CPU",
                 language: str = "en", probe_timeout: float = 3.0,
                 whisper_model: str = "base"):
        self._recorder = None
        self._language = language
        self._whisper_model = whisper_model
        self._probe_timeout = max(0.5, float(probe_timeout))
        self._error = ""
        self._recording = False
        self._last_text = ""
        self._consumed = True
        self._mic_failed = False

        # 1) Probe for a usable microphone FIRST. On a VDI the mic may be listed
        #    but opening a stream BLOCKS forever (it doesn't raise) — so we must
        #    actually open+close a stream in a worker thread with a timeout.
        if not self._probe_microphone(self._probe_timeout):
            self._error = (
                "No working microphone (input stream did not open in time). "
                "On a VDI, enable mic redirection in the client."
            )
            return

        # 2) Load the Whisper recorder (model dir chosen by size; all OpenVINO IR)
        try:
            from airsketch.voice import VoiceRecorder, whisper_model_dir
            path = model_path or whisper_model_dir(self._whisper_model)
            self._recorder = VoiceRecorder(model_path=path, device=device,
                                           language=self._language)
        except FileNotFoundError:
            self._error = (
                f"Whisper model not found at {path}. "
                f"Set it up with `python -m training.setup_whisper`."
            )
        except RuntimeError as e:
            self._error = str(e)  # e.g. sounddevice / openvino-genai missing
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _probe_microphone(timeout: float = 3.0) -> bool:
        """Return True only if an input stream actually opens cleanly in time.

        Critically, this catches the VDI case where opening the mic *blocks*
        forever instead of raising: we run the open+close in a daemon thread
        and give up after `timeout` seconds, leaving the UI thread free.
        """
        try:
            import sounddevice as sd
        except Exception:
            return False

        result = {"ok": False, "done": False}

        def worker():
            try:
                # Use the same one-shot sd.rec path the recorder uses — a brief
                # 0.1s capture proves the mic opens without hanging.
                sd.rec(int(0.1 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
                sd.wait()
                result["ok"] = True
            except Exception:
                result["ok"] = False
            finally:
                try:
                    sd.stop()
                except Exception:
                    pass
                result["done"] = True

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout)
        # If the worker is still alive, the open blocked — treat as no mic.
        # The daemon thread is abandoned (harmless; dies with the process).
        return result["done"] and result["ok"]

    @property
    def available(self) -> bool:
        return self._recorder is not None and not self._mic_failed

    @property
    def error(self) -> str:
        return self._error

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_transcribing(self) -> bool:
        return bool(self._recorder and self._recorder.is_transcribing)

    @property
    def last_text(self) -> str:
        return self._last_text

    @property
    def last_audio(self):
        """Audio of the most recent utterance (for speaker verification)."""
        return self._recorder.last_audio if self._recorder else None

    def take_result(self):
        """Return (text, audio) once a fresh utterance is ready, else None.

        Like poll() but yields the raw transcription + audio (no intent parsing),
        so the caller can both verify the speaker and parse the command. Fires
        once per utterance.
        """
        if not self.available or self._recording:
            return None
        if not self._recorder.result_ready or self._recorder.is_transcribing:
            return None
        text = self._recorder.consume_result()
        audio = self._recorder.last_audio
        self._last_text = text
        return (text, audio)

    def toggle(self, purpose: str = "command") -> None:
        """Start recording if idle; stop + transcribe if recording.

        `purpose` ("command" | "dictation") selects the Whisper config used for
        this utterance. Any microphone/PortAudio failure disables voice gracefully
        rather than crashing the app (important on VDIs where the mic may not redirect).
        """
        if not self.available:
            return
        try:
            if self._recording:
                self._recorder.stop()
                self._recording = False
            else:
                self._last_text = ""
                self._consumed = True
                self._recorder.start(purpose=purpose)
                self._recording = True
        except Exception as e:
            self._mic_failed = True
            self._recording = False
            self._error = f"Microphone error: {type(e).__name__}: {e}"
            print(f"[voice] {self._error} — voice disabled for this session.")

    def poll(self) -> Optional[Intent]:
        """Return a parsed Intent once a fresh transcription is ready, else None.

        Call once per frame. Fires exactly once per utterance via the recorder's
        result-ready flag — robust to repeated/identical commands.
        """
        if not self.available or self._recording:
            return None
        if not self._recorder.result_ready or self._recorder.is_transcribing:
            return None
        text = self._recorder.consume_result()
        self._last_text = text
        if not text:
            return None
        return parse_command(text)

    def close(self) -> None:
        if self._recorder:
            try:
                self._recorder.close()
            except Exception:
                pass
