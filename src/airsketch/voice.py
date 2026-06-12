"""Voice input via OpenVINO Whisper — speech-to-text for LLM interaction.

Usage:
    recorder = VoiceRecorder()
    recorder.start()      # begin recording
    recorder.stop()       # stop and transcribe
    text = recorder.text  # get transcription result
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    import openvino_genai as ov_genai
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_WHISPER_MODEL = _MODELS_DIR / "whisper-base-ov"

SAMPLE_RATE = 16000


def whisper_model_dir(size: str = "base") -> str:
    """Resolve the OpenVINO Whisper model dir for a size ('base' | 'small').

    All sizes are OpenVINO IR (run via ov_genai.WhisperPipeline). Falls back to
    the base dir if an unknown size is given.
    """
    size = (size or "base").lower()
    name = "whisper-small-ov" if size == "small" else "whisper-base-ov"
    return str(_MODELS_DIR / name)


class VoiceRecorder:
    """Record audio from microphone and transcribe via Whisper."""

    # Max length of a single utterance (buffer is pre-allocated; stopping early
    # just leaves trailing silence, which Whisper ignores).
    MAX_SECONDS = 10

    def __init__(self, model_path: str | None = None, device: str = "CPU",
                 language: str = "en"):
        if not HAS_AUDIO:
            raise RuntimeError("sounddevice is not installed (pip install sounddevice)")
        if not HAS_WHISPER:
            raise RuntimeError("openvino-genai is not installed")
        path = model_path or str(_WHISPER_MODEL)
        if not Path(path).exists():
            raise FileNotFoundError(f"Whisper model not found: {path}")

        # whisper-base-ov is the multilingual base (has <|he|>), so Hebrew works
        # on the same OpenVINO pipeline — we just pick the language token.
        self._language = (language or "en").lower()
        self._pipe = ov_genai.WhisperPipeline(path, device)
        # Two configs: short + command-vocabulary-biased for V commands, and a
        # neutral + longer one for free-form dictation (the command bias would
        # otherwise nudge dictated sentences toward shape words and truncate them).
        self._gen_command = self._build_config("command")
        self._gen_dictation = self._build_config("dictation")
        self._purpose = "command"
        self._buffer: np.ndarray | None = None
        self._recording = False
        self._transcribing = False
        self._text = ""
        self._last_audio: np.ndarray | None = None   # trimmed audio of last utterance
        self._result_ready = False   # set True when a fresh transcription lands

    # Bias the decoder toward our command vocabulary so short utterances like
    # "draw a circle" aren't mis-heard as other languages / nonsense.
    _CONTEXT_PROMPT = (
        "Draw a circle, triangle, square, rectangle, star, arrow, line, "
        "house, cat, tree, sun, flower, fish, car, airplane. Next. Submit. Clear."
    )
    # Hebrew equivalent — the shape + command vocabulary, to bias Hebrew decoding.
    _CONTEXT_PROMPT_HE = (
        "צייר עיגול, משולש, ריבוע, מלבן, כוכב, חץ, קו, בית, חתול, עץ, שמש, "
        "פרח, דג, מכונית, מטוס. הבא. שלח. נקה."
    )

    @staticmethod
    def language_token(lang: str) -> str:
        """Map a language code to the Whisper language token."""
        return "<|he|>" if str(lang).lower().startswith("he") else "<|en|>"

    def _build_config(self, purpose: str = "command"):
        """Build a Whisper generation config.

        purpose="command"  → short + biased toward the shape/command vocabulary
                              (rescues short, mis-heard commands like "circle").
        purpose="dictation" → neutral (no vocab bias) + more tokens, so free
                              speech transcribes naturally and isn't truncated.

        Defensive — different openvino-genai versions expose slightly different
        fields, so each is set in its own try/except and unknown ones are skipped.
        """
        try:
            cfg = self._pipe.get_generation_config()
        except Exception:
            return None
        attrs = [
            ("language", self.language_token(self._language)),
            ("task", "transcribe"),
        ]
        if purpose == "dictation":
            attrs.append(("max_new_tokens", 220))
            # no initial_prompt / hotwords → no command-vocabulary bias
        else:
            is_he = self._language.startswith("he")
            prompt = self._CONTEXT_PROMPT_HE if is_he else self._CONTEXT_PROMPT
            attrs += [
                ("max_new_tokens", 80),
                ("initial_prompt", prompt),
                ("hotwords", prompt),
            ]
        for attr, val in attrs:
            try:
                setattr(cfg, attr, val)
            except Exception:
                pass
        return cfg

    def start(self, purpose: str = "command") -> None:
        """Begin recording via sd.rec() (non-blocking, fills a pre-allocated
        buffer in the background). This one-shot API is far more robust than a
        continuous InputStream + callback, especially alongside a live camera.

        `purpose` selects the transcription config: "command" (biased) or
        "dictation" (neutral, longer). Raises on mic/PortAudio failure.
        """
        if self._recording:
            return
        self._purpose = purpose if purpose in ("command", "dictation") else "command"
        self._text = ""
        self._result_ready = False
        n = int(self.MAX_SECONDS * SAMPLE_RATE)
        # Pre-allocate a ZERO-filled buffer and have sd.rec write into it. sd.rec's
        # own buffer is np.empty (uninitialized) — frames PortAudio never fills keep
        # garbage (NaN/Inf/huge floats), which then poisons Whisper into emitting
        # "!!!!". A zeroed `out=` means unfilled frames read as clean silence.
        buf = np.zeros((n, 1), dtype="float32")
        try:
            # sd.rec returns immediately; recording proceeds on the PortAudio thread.
            sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32", out=buf)
            self._buffer = buf
            self._recording = True
        except Exception:
            self._buffer = None
            self._recording = False
            raise

    def stop(self) -> None:
        """Stop recording (sd.stop) and transcribe the captured buffer."""
        if not self._recording:
            return
        self._recording = False
        try:
            sd.stop()
        except Exception:
            pass
        self._transcribing = True
        thread = threading.Thread(target=self._transcribe, daemon=True)
        thread.start()

    def _transcribe(self) -> None:
        try:
            if self._buffer is None:
                self._text = ""
                return
            audio = self._buffer.flatten()
            # Defensive: scrub any non-finite samples (NaN/Inf) and clamp to the
            # valid PCM range. Protects Whisper even if PortAudio writes garbage
            # into the unfilled tail of the pre-allocated buffer.
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(audio, -1.0, 1.0, out=audio)
            # Trim trailing all-zero silence (the unused tail of the buffer)
            nz = np.nonzero(np.abs(audio) > 1e-4)[0]
            if len(nz):
                audio = audio[: nz[-1] + 1]
            self._last_audio = audio.copy()   # keep for speaker verification
            if audio.size < SAMPLE_RATE // 4:   # < 0.25s of sound → nothing useful
                self._text = ""
                return
            gen_config = (self._gen_dictation if self._purpose == "dictation"
                          else self._gen_command)
            if gen_config is not None:
                try:
                    result = self._pipe.generate(audio, gen_config)
                except Exception:
                    result = self._pipe.generate(audio)   # fallback: default config
            else:
                result = self._pipe.generate(audio)
            self._text = result.texts[0].strip() if result.texts else ""
        except Exception:
            self._text = ""
        finally:
            self._result_ready = True   # a result (even empty) is now available
            self._transcribing = False

    def consume_result(self) -> str:
        """Return the latest transcription once, clearing the ready flag."""
        self._result_ready = False
        return self._text

    @property
    def result_ready(self) -> bool:
        return self._result_ready

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_transcribing(self) -> bool:
        return self._transcribing

    @property
    def is_busy(self) -> bool:
        return self._recording or self._transcribing

    @property
    def text(self) -> str:
        return self._text

    @property
    def last_audio(self) -> np.ndarray | None:
        """Trimmed float32 audio of the most recent utterance (for speaker ID)."""
        return self._last_audio

    def close(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass
