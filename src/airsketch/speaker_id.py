"""Speaker recognition for the teacher-voice feature, on the OpenVINO runtime.

Pipeline:
    waveform (16 kHz mono) -> 80-dim Kaldi-style log-mel fbank (+ per-utterance CMN)
        -> WeSpeaker ResNet34 ONNX (OpenVINO) -> 256-dim speaker embedding

Enrollment averages a few embeddings of the teacher's voice into a profile; later
utterances are accepted when their cosine similarity to the profile clears a
threshold. The profile persists to JSON so enrollment is a one-time step.

The fbank matches WeSpeaker's feature config (25 ms / 10 ms / 80 mel / hamming).
It need not be bit-exact with Kaldi: enrollment and verification use the SAME
extractor, and CMN removes global scaling, so relative cosine scores are stable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

_MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "speaker" / "onnx__model.onnx"

SAMPLE_RATE = 16000
_N_MELS = 80
_FRAME_LEN = 400      # 25 ms @ 16 kHz
_FRAME_SHIFT = 160    # 10 ms
_N_FFT = 512          # round_to_power_of_two(400)
_PREEMPH = 0.97
_MIN_FRAMES = 9       # model's min_num_frames


def _mel_filterbank(n_fft: int, sr: int, n_mels: int,
                    low: float = 20.0, high: float = 8000.0) -> np.ndarray:
    def hz2mel(f): return 1127.0 * np.log(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (np.exp(m / 1127.0) - 1.0)
    n_bins = n_fft // 2 + 1
    fftfreqs = np.linspace(0.0, sr / 2.0, n_bins)
    mpts = np.linspace(hz2mel(low), hz2mel(high), n_mels + 2)
    fpts = mel2hz(mpts)
    fb = np.zeros((n_mels, n_bins), np.float64)
    for m in range(1, n_mels + 1):
        l, c, r = fpts[m - 1], fpts[m], fpts[m + 1]
        left = (fftfreqs - l) / max(c - l, 1e-9)
        right = (r - fftfreqs) / max(r - c, 1e-9)
        fb[m - 1] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


_FB = _mel_filterbank(_N_FFT, SAMPLE_RATE, _N_MELS)
_WIN = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(_FRAME_LEN) / (_FRAME_LEN - 1))


def compute_fbank(wav: np.ndarray) -> np.ndarray:
    """80-dim log-mel fbank with per-utterance CMN. Returns [frames, 80] float32."""
    wav = np.asarray(wav, dtype=np.float64).flatten()
    if wav.size < _FRAME_LEN:
        return np.zeros((0, _N_MELS), np.float32)
    n_frames = 1 + (wav.size - _FRAME_LEN) // _FRAME_SHIFT
    idx = np.arange(_FRAME_LEN)[None, :] + _FRAME_SHIFT * np.arange(n_frames)[:, None]
    frames = wav[idx]                                   # [n_frames, 400]
    frames = frames - frames.mean(axis=1, keepdims=True)  # remove DC offset
    pre = frames.copy()                                  # pre-emphasis
    pre[:, 1:] -= _PREEMPH * frames[:, :-1]
    pre[:, 0] -= _PREEMPH * frames[:, 0]
    frames = pre * _WIN                                  # hamming window
    spec = np.fft.rfft(frames, n=_N_FFT, axis=1)
    power = spec.real ** 2 + spec.imag ** 2              # [n_frames, 257]
    mel = np.clip(power @ _FB.T, 1.19e-7, None)          # [n_frames, 80]
    feat = np.log(mel).astype(np.float64)
    feat -= feat.mean(axis=0, keepdims=True)             # CMN (per-bin over time)
    return feat.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerEmbedder:
    """WeSpeaker ResNet34 (ONNX) on OpenVINO → 256-dim L2-normalized embedding."""

    def __init__(self, device: str = "CPU", model_path: Optional[str] = None):
        import openvino as ov
        path = model_path or str(_MODEL)
        if not Path(path).exists():
            raise FileNotFoundError(f"Speaker model not found: {path}")
        core = ov.Core()
        self._model = core.compile_model(core.read_model(path), device)
        self._out = self._model.output(0)

    def embed(self, wav: np.ndarray) -> Optional[np.ndarray]:
        feat = compute_fbank(wav)
        if feat.shape[0] < _MIN_FRAMES:
            return None
        emb = self._model([feat[None]])[self._out][0]    # [256]
        n = np.linalg.norm(emb)
        return (emb / n).astype(np.float32) if n > 0 else emb.astype(np.float32)


@dataclass
class SpeakerProfile:
    """Enrolled teacher voice: a mean embedding + an acceptance threshold."""
    embedding: List[float]
    threshold: float = 0.5
    count: int = 0

    def score(self, emb: np.ndarray) -> float:
        return cosine(np.asarray(self.embedding, np.float32), emb)

    def matches(self, emb: np.ndarray) -> bool:
        return self.score(emb) >= self.threshold

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({
            "embedding": list(self.embedding),
            "threshold": self.threshold,
            "count": self.count,
        }), encoding="utf-8")

    @staticmethod
    def load(path: str) -> "Optional[SpeakerProfile]":
        p = Path(path)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return SpeakerProfile(embedding=d["embedding"],
                                  threshold=float(d.get("threshold", 0.5)),
                                  count=int(d.get("count", 0)))
        except Exception:
            return None


def build_profile(embeddings: List[np.ndarray], threshold: float = 0.5) -> SpeakerProfile:
    """Average enrollment embeddings (L2-normalized) into a profile."""
    arr = np.asarray([e for e in embeddings if e is not None], np.float64)
    if arr.size == 0:
        raise ValueError("no embeddings to enroll")
    mean = arr.mean(axis=0)
    n = np.linalg.norm(mean)
    if n > 0:
        mean = mean / n
    return SpeakerProfile(embedding=mean.astype(np.float32).tolist(),
                          threshold=threshold, count=len(arr))
