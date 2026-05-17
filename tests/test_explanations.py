"""Tests for the real `explain_decision` implementation.

Validates that:
- A persisted decision can be looked up.
- Spectrogram PNG is rendered when modality requests it.
- Real spectral features are computed (no fabricated blade-rate / harmonics).
- Multimodal Gemma is called when reachable, otherwise the templated
  fallback is used — but the fallback ONLY references real measured values.
- Bad inputs (missing decision, missing clip, bad modality) are handled.

We never call the live Ollama server in these tests — the multimodal call
is monkeypatched to return a stub string (or None to simulate failure).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from ocean_sentinel.gemma import explanations


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_decisions_dir(monkeypatch, tmp_path) -> Path:
    """Redirect explanations.DECISIONS_DIR at the module level so each test
    gets a clean directory under tmp_path."""
    d = tmp_path / "decisions"
    d.mkdir()
    monkeypatch.setattr(explanations, "DECISIONS_DIR", d)
    return d


@pytest.fixture
def synthetic_vessel_clip(tmp_path) -> Path:
    """A 10-second WAV with strong energy below 200 Hz — should look like
    a vessel signature in our log-mel view."""
    sr = 16_000
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    # Two low-frequency tones (60 Hz fundamental + 120 Hz harmonic) over noise.
    y = (
        0.5 * np.sin(2 * np.pi * 60 * t)
        + 0.3 * np.sin(2 * np.pi * 120 * t)
        + 0.05 * np.random.RandomState(0).randn(len(t))
    ).astype(np.float32)
    p = tmp_path / "vessel.wav"
    sf.write(str(p), y, sr)
    return p


@pytest.fixture
def synthetic_ambient_clip(tmp_path) -> Path:
    """A 10-second WAV of broadband noise with no narrowband structure —
    should look like ambient."""
    sr = 16_000
    rng = np.random.RandomState(1)
    y = (0.2 * rng.randn(sr * 10)).astype(np.float32)
    p = tmp_path / "ambient.wav"
    sf.write(str(p), y, sr)
    return p


def _write_decision(d_dir: Path, decision_id: str, clip: Path,
                    tier: str = "DARK_VESSEL", **kwargs) -> None:
    record = {
        "ok": True,
        "decision_id": decision_id,
        "site_id": "test-site",
        "clip": str(clip),
        "cnn_label": "ship",
        "cnn_confidence": 0.92,
        "cnn_uncertainty": 0.08,
        "conformal_threshold": 0.61,
        "conformal_p": 0.92,
        "conformal_pass": True,
        "ais_vessels_in_radius": 0,
        "decision_tier": tier,
        "severity": "HIGH",
        "checkpoint": "data/models/cnn_v7_4.pt",
        "summary": f"{tier} (HIGH) · CNN p=0.92",
        **kwargs,
    }
    (d_dir / f"{decision_id}.json").write_text(json.dumps(record, indent=2))


# ── Decision lookup / persistence ───────────────────────────────────────


def test_write_and_load_decision_record(tmp_decisions_dir):
    rec = {"decision_id": "DET-00001", "clip": "x.wav", "decision_tier": "AMBIENT"}
    explanations.write_decision_record("DET-00001", rec)
    loaded = explanations._load_decision_record("DET-00001")
    assert loaded == rec


def test_missing_decision_returns_error(tmp_decisions_dir):
    out = explanations.explain_decision_real("DET-NOPE")
    assert out["ok"] is False
    assert "not found" in out["error"]


def test_missing_clip_returns_error(tmp_decisions_dir, tmp_path):
    _write_decision(tmp_decisions_dir, "DET-00002", tmp_path / "does_not_exist.wav")
    out = explanations.explain_decision_real("DET-00002")
    assert out["ok"] is False
    assert "missing" in out["error"]


def test_bad_modality_rejected(tmp_decisions_dir, synthetic_vessel_clip):
    _write_decision(tmp_decisions_dir, "DET-00003", synthetic_vessel_clip)
    out = explanations.explain_decision_real("DET-00003", modality="emoji")
    assert out["ok"] is False
    assert "modality" in out["error"]


# ── Spectral features (the load-bearing measurements) ──────────────────


def test_features_distinguish_vessel_from_ambient(
    synthetic_vessel_clip, synthetic_ambient_clip,
):
    v = explanations._compute_spectral_features(synthetic_vessel_clip)
    a = explanations._compute_spectral_features(synthetic_ambient_clip)

    # Vessel clip has tonal energy in the 60-120 Hz band → very high
    # low-band fraction. Ambient is broadband noise → low fraction.
    assert v["low_band_energy_fraction_below_200hz"] > 0.5
    assert a["low_band_energy_fraction_below_200hz"] < 0.1

    # Vessel has lower spectral centroid than ambient (energy is concentrated low).
    assert v["spectral_centroid_hz"] < a["spectral_centroid_hz"]

    # Vessel is more tonal (lower flatness) than ambient (≈ white noise).
    assert v["spectral_flatness"] < a["spectral_flatness"]


def test_features_round_to_sane_precision(synthetic_vessel_clip):
    f = explanations._compute_spectral_features(synthetic_vessel_clip)
    for k in (
        "peak_frequency_hz",
        "spectral_centroid_hz",
        "spectral_flatness",
        "low_band_energy_fraction_below_200hz",
        "rms_db",
        "duration_s",
    ):
        assert k in f
        assert isinstance(f[k], float)


# ── Spectrogram rendering ───────────────────────────────────────────────


def test_spectrogram_modality_writes_png(tmp_decisions_dir, synthetic_vessel_clip):
    _write_decision(tmp_decisions_dir, "DET-IMG", synthetic_vessel_clip)
    out = explanations.explain_decision_real(
        "DET-IMG", modality="spectrogram", use_multimodal=False,
    )
    assert out["ok"] is True
    assert out["spectrogram_path"]
    assert Path(out["spectrogram_path"]).exists()
    # spectrogram-only mode does not produce a narration
    assert out["narration_source"] == "none"
    assert out["explanation"] is None


def test_text_only_modality_skips_png(tmp_decisions_dir, synthetic_vessel_clip):
    _write_decision(tmp_decisions_dir, "DET-TXT", synthetic_vessel_clip)
    out = explanations.explain_decision_real(
        "DET-TXT", modality="text", use_multimodal=False,
    )
    assert out["ok"] is True
    assert out["spectrogram_path"] is None
    assert out["narration_source"] == "templated"
    assert out["explanation"]
    # Templated narration must reference real measured values, never
    # invented blade-rate / harmonics.
    assert "blade" not in out["explanation"].lower()
    assert "harmonic" not in out["explanation"].lower()


# ── Multimodal call: mocked ─────────────────────────────────────────────


def test_multimodal_call_is_used_when_available(
    tmp_decisions_dir, synthetic_vessel_clip, monkeypatch,
):
    captured: dict[str, Any] = {}

    def fake_call(*, image_path, context, host, model, timeout_s=60.0):
        captured["image_path"] = image_path
        captured["host"] = host
        captured["model"] = model
        captured["context_keys"] = sorted(context.keys())
        return "Steady bright band near the bottom of the spectrogram."

    monkeypatch.setattr(explanations, "_call_gemma_multimodal", fake_call)

    _write_decision(tmp_decisions_dir, "DET-MM", synthetic_vessel_clip)
    out = explanations.explain_decision_real(
        "DET-MM", modality="spectrogram+text", use_multimodal=True,
    )

    assert out["ok"] is True
    assert out["narration_source"] == "gemma_multimodal"
    assert out["explanation"].startswith("Steady bright band")
    # The image path passed to Gemma is the rendered PNG.
    assert captured["image_path"] == Path(out["spectrogram_path"])
    # Real spectral features must be merged into the context Gemma sees.
    assert "low_band_energy_fraction_below_200hz" in captured["context_keys"]
    assert "decision_tier" in captured["context_keys"]


def test_multimodal_failure_falls_back_to_templated(
    tmp_decisions_dir, synthetic_vessel_clip, monkeypatch,
):
    monkeypatch.setattr(
        explanations, "_call_gemma_multimodal",
        lambda **kwargs: None,  # simulates ollama unreachable / timeout
    )

    _write_decision(tmp_decisions_dir, "DET-FB", synthetic_vessel_clip)
    out = explanations.explain_decision_real(
        "DET-FB", modality="spectrogram+text", use_multimodal=True,
    )

    assert out["ok"] is True
    assert out["narration_source"] == "templated"
    assert out["explanation"]
    # PNG is still rendered even though Gemma was unreachable.
    assert Path(out["spectrogram_path"]).exists()


def test_use_multimodal_false_skips_gemma_entirely(
    tmp_decisions_dir, synthetic_vessel_clip, monkeypatch,
):
    """When use_multimodal=False (env flag or test toggle), we must not
    even attempt the Gemma call."""
    called = {"hit": False}

    def boom(**kwargs):
        called["hit"] = True
        return "should-not-be-used"

    monkeypatch.setattr(explanations, "_call_gemma_multimodal", boom)

    _write_decision(tmp_decisions_dir, "DET-NOMM", synthetic_vessel_clip)
    out = explanations.explain_decision_real(
        "DET-NOMM", modality="spectrogram+text", use_multimodal=False,
    )
    assert called["hit"] is False
    assert out["narration_source"] == "templated"


# ── Trace shape ─────────────────────────────────────────────────────────


def test_trace_includes_real_features_and_decision_context(
    tmp_decisions_dir, synthetic_vessel_clip, monkeypatch,
):
    monkeypatch.setattr(
        explanations, "_call_gemma_multimodal", lambda **kw: "ok"
    )
    _write_decision(tmp_decisions_dir, "DET-TR", synthetic_vessel_clip,
                    tier="DARK_VESSEL")
    out = explanations.explain_decision_real("DET-TR")

    trace = out["trace"]
    # Real spectral features
    assert "peak_frequency_hz" in trace
    assert "spectral_centroid_hz" in trace
    assert "low_band_energy_fraction_below_200hz" in trace
    # Decision context preserved
    assert trace["decision_tier"] == "DARK_VESSEL"
    assert trace["site_id"] == "test-site"
    # No fabricated keys from the old mock
    assert "blade_rate_hz" not in trace
    assert "harmonics" not in trace


# ── Env-flag short circuit ──────────────────────────────────────────────


def test_env_flag_disables_multimodal(monkeypatch):
    monkeypatch.setenv("OS_DISABLE_MULTIMODAL", "1")
    assert explanations.is_multimodal_disabled() is True
    monkeypatch.setenv("OS_DISABLE_MULTIMODAL", "no")
    assert explanations.is_multimodal_disabled() is False
