"""Tests for speaker-recognition math (no ONNX model / mic needed)."""
import numpy as np
import pytest

from airsketch.speaker_id import (
    compute_fbank, cosine, build_profile, SpeakerProfile,
)


def test_fbank_shape_and_dims():
    wav = np.zeros(16000, np.float32)        # 1 s of silence
    feat = compute_fbank(wav)
    assert feat.ndim == 2 and feat.shape[1] == 80
    assert feat.shape[0] == 1 + (16000 - 400) // 160

def test_fbank_too_short_returns_empty():
    assert compute_fbank(np.zeros(100, np.float32)).shape == (0, 80)

def test_cosine_bounds():
    a = np.array([1.0, 0.0, 0.0]); b = np.array([1.0, 0.0, 0.0])
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert cosine(a, np.array([-1.0, 0.0, 0.0])) == pytest.approx(-1.0)
    assert cosine(a, np.zeros(3)) == 0.0      # zero-vector guard

def test_build_profile_averages_and_normalizes():
    e1 = np.array([1.0, 0.0]); e2 = np.array([0.0, 1.0])
    prof = build_profile([e1, e2], threshold=0.6)
    v = np.array(prof.embedding)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    assert prof.count == 2 and prof.threshold == 0.6

def test_build_profile_empty_raises():
    with pytest.raises(ValueError):
        build_profile([])

def test_profile_matches_threshold():
    prof = build_profile([np.array([1.0, 0.0])], threshold=0.8)
    assert prof.matches(np.array([1.0, 0.0]))          # cosine 1.0 >= 0.8
    assert not prof.matches(np.array([0.3, 1.0]))      # well below 0.8

def test_profile_save_load_roundtrip(tmp_path):
    prof = build_profile([np.array([1.0, 2.0, 3.0])], threshold=0.55)
    p = tmp_path / "prof.json"
    prof.save(str(p))
    loaded = SpeakerProfile.load(str(p))
    assert loaded is not None
    assert loaded.threshold == 0.55
    assert loaded.embedding == prof.embedding

def test_profile_load_missing_returns_none(tmp_path):
    assert SpeakerProfile.load(str(tmp_path / "nope.json")) is None
