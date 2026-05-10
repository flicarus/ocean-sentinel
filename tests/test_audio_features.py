"""Tests for the multi-band ambient classifier.

These tests build synthetic mel-spectrograms directly so we can exercise
the classifier's branches without depending on audio synthesis quality
(at 16 kHz mono, 1-2 ms snapping-shrimp clicks smear across 64 ms STFT
frames, which makes pure-audio reproduction of the reef p95-median gap
unreliable in unit tests).
"""
from __future__ import annotations

import librosa
import numpy as np
import pytest

from ocean_sentinel.gemma import audio_features


N_MELS = 64
SR = 16_000


@pytest.fixture
def mel_freqs() -> np.ndarray:
    return librosa.mel_frequencies(n_mels=N_MELS, fmin=20, fmax=8000)


def _flat(value_db: float, n_frames: int = 200) -> np.ndarray:
    return np.full((N_MELS, n_frames), value_db, dtype=np.float32)


# ── Snapping-shrimp / biological transient detection ────────────────────


def test_classifier_detects_biological_transients(mel_freqs):
    """Big p95 vs median gap in HF bands → biological label."""
    mel_db = _flat(-20.0)
    n_frames = mel_db.shape[1]

    # Steady mid-power baseline in low band (fish chorus stand-in).
    low_band = mel_freqs < 500
    mel_db[low_band, :] = -10.0

    # Sparse loud transients in HF bands (snapping shrimp stand-in).
    hf_band = mel_freqs > 1500
    rng = np.random.RandomState(0)
    transient_frames = rng.choice(n_frames, size=20, replace=False)
    mel_db[np.ix_(hf_band, transient_frames)] = +20.0

    label, feats = audio_features._classify_ambient(mel_db, mel_freqs)
    assert "biological" in label
    assert "transient-rich" in label
    assert feats["transient_gap_hf_db"] > 15.0


# ── Steady vessel-band (no transients) ─────────────────────────────────


def test_classifier_calls_steady_low_band_vessel(mel_freqs):
    """Steady 100-400 Hz dominant, no HF transients → vessel band."""
    mel_db = _flat(-30.0)

    vessel_band = (mel_freqs >= 100) & (mel_freqs < 500)
    mel_db[vessel_band, :] = +5.0  # bright sustained low-band

    label, feats = audio_features._classify_ambient(mel_db, mel_freqs)
    assert "vessel band" in label
    assert feats["transient_gap_hf_db"] < 5.0


# ── Quiet deep-water ambient ───────────────────────────────────────────


def test_classifier_calls_infrasound_when_dominant_below_80hz(mel_freqs):
    """Energy concentrated in lowest mel bands → infrasound."""
    mel_db = _flat(-30.0)
    very_low = mel_freqs < 60
    mel_db[very_low, :] = +5.0

    label, _ = audio_features._classify_ambient(mel_db, mel_freqs)
    assert "infrasound" in label


# ── Mid-band (mixed) ───────────────────────────────────────────────────


def test_classifier_calls_mid_band_when_dominant_in_500_2000hz(mel_freqs):
    mel_db = _flat(-30.0)
    mid_band = (mel_freqs >= 500) & (mel_freqs < 2000)
    mel_db[mid_band, :] = +5.0

    label, _ = audio_features._classify_ambient(mel_db, mel_freqs)
    assert "mid-frequency" in label


# ── Steady high-frequency tonal (whale chorus etc) ─────────────────────


def test_classifier_calls_steady_high_frequency_biological_when_no_transients(
    mel_freqs,
):
    """If the dominant median is >2 kHz but there are no big transients,
    fall through to the steady high-frequency biological label rather
    than transient-rich."""
    mel_db = _flat(-30.0)
    hf_steady = mel_freqs > 3000
    mel_db[hf_steady, :] = +5.0  # steady, no transient gap

    label, feats = audio_features._classify_ambient(mel_db, mel_freqs)
    assert "high-frequency" in label
    assert "biological" in label
    # transient gap should NOT be inflated for steady tonal HF
    assert feats["transient_gap_hf_db"] < 5.0


# ── End-to-end: signature pipeline returns the new fields ──────────────


def test_compute_spectral_signature_returns_new_diagnostic_fields(tmp_path):
    """Smoke test that the public function exposes transient_gap_hf_db
    and snapping_shrimp."""
    import soundfile as sf

    rng = np.random.RandomState(42)
    n = SR * 10
    y = (0.2 * rng.randn(n)).astype(np.float32)  # white noise
    p = tmp_path / "noise.wav"
    sf.write(str(p), y, SR)

    out = audio_features.compute_spectral_signature(str(p), n_mels=N_MELS)
    assert out["ok"]
    assert "transient_gap_hf_db" in out
    assert "snapping_shrimp" in out
    assert "ambient_class" in out
    # White noise should not trip the snapping_shrimp flag
    assert out["snapping_shrimp"] is False


# ── Adapter assessment ─────────────────────────────────────────────────


def test_assessment_validated_for_high_val_acc():
    from ocean_sentinel.gemma.adapter import assessment_from_val_acc
    assessment, rec = assessment_from_val_acc(0.95)
    assert assessment == "model_validated"
    assert "correctly" in rec


def test_assessment_marginal_in_middle_band():
    from ocean_sentinel.gemma.adapter import assessment_from_val_acc
    for val in (0.5, 0.65, 0.84):
        assessment, rec = assessment_from_val_acc(val)
        assert assessment == "marginal"
        assert "uncertain" in rec.lower() or "compensate" in rec.lower()


def test_assessment_ood_confusion_when_cnn_calls_ambient_ship():
    from ocean_sentinel.gemma.adapter import assessment_from_val_acc
    for val in (0.0, 0.2, 0.49):
        assessment, rec = assessment_from_val_acc(val)
        assert assessment == "ood_confusion"
        assert "out-of-distribution" in rec or "PER-EVENT" in rec
