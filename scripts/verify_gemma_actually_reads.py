"""Adversarial test: does Gemma actually read the spectrogram, or does it
just regurgitate the numeric context?

Strategy: feed Gemma a spectrogram of pure white noise (clearly no vessel)
together with a numeric context that LIES and claims it's a DARK_VESSEL
detection with high CNN confidence. The prompt explicitly tells Gemma
that if what it sees disagrees with the label, it should say so.

Outcomes:
- If Gemma is reading the image: should call out the disagreement
  ("looks ambiguous", "no persistent band", etc.).
- If Gemma is just paraphrasing the numbers: will agree with the label
  ("consistent with DARK_VESSEL", "low-band signature is clear").

Then we run the symmetric test: a strong tonal clip with numeric context
falsely labelled AMBIENT. Gemma should also call out the disagreement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ocean_sentinel.gemma import explanations


def _make_white_noise(path: Path, sr: int = 16_000, dur_s: float = 30.0) -> Path:
    rng = np.random.RandomState(42)
    y = (0.2 * rng.randn(int(sr * dur_s))).astype(np.float32)
    sf.write(str(path), y, sr)
    return path


def _make_tonal_vessel(path: Path, sr: int = 16_000, dur_s: float = 30.0) -> Path:
    """Strong narrowband tones at 60 + 120 Hz — looks unambiguously vessel-like."""
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    y = (
        0.6 * np.sin(2 * np.pi * 60 * t)
        + 0.4 * np.sin(2 * np.pi * 120 * t)
        + 0.05 * np.random.RandomState(0).randn(len(t))
    ).astype(np.float32)
    sf.write(str(path), y, sr)
    return path


def _craft(decision_id: str, clip: Path, fake_tier: str, fake_prob: float) -> None:
    explanations.DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ok": True,
        "decision_id": decision_id,
        "site_id": "adversarial-test",
        "clip": str(clip),
        "cnn_label": "ship" if fake_tier != "AMBIENT" else "not_ship",
        "cnn_confidence": fake_prob,
        "cnn_uncertainty": 0.10,
        "conformal_threshold": 0.61,
        "conformal_p": fake_prob,
        "conformal_pass": fake_prob > 0.61,
        "ais_vessels_in_radius": 0,
        "decision_tier": fake_tier,
        "severity": "HIGH" if fake_tier == "DARK_VESSEL" else "NONE",
        "checkpoint": "data/models/cnn_v7_4.pt",
        "summary": f"{fake_tier} (test)",
    }
    explanations.write_decision_record(decision_id, record)


def run_case(label: str, clip: Path, fake_tier: str, fake_prob: float) -> None:
    decision_id = f"ADV-{label}"
    _craft(decision_id, clip, fake_tier, fake_prob)
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"  ground truth (visual): {clip.stem}")
    print(f"  faked label:           {fake_tier}  (CNN p = {fake_prob})")

    out = explanations.explain_decision_real(decision_id, modality="spectrogram+text")
    print(f"\n  narration_source: {out.get('narration_source')}")
    print(f"  spectrogram_path: {out.get('spectrogram_path')}")
    print(f"\n  explanation:")
    print(f"    {out.get('explanation')}")

    # Heuristic: does Gemma flag a disagreement?
    text = (out.get("explanation") or "").lower()
    flagged = any(w in text for w in [
        "ambiguous", "disagree", "inconsistent", "does not match",
        "contradict", "no persistent", "no clear", "without a",
        "no vessel", "no structured", "lacks", "broadband",
        "unstructured", "diffuse", "white noise", "noise",
    ])
    confirmed = any(w in text for w in [
        "consistent with", "supports", "matches the", "as expected",
    ])
    print(f"\n  flagged disagreement: {flagged}")
    print(f"  echoed agreement:     {confirmed}")


if __name__ == "__main__":
    tmp = Path("data/tmp_adversarial")
    tmp.mkdir(parents=True, exist_ok=True)

    noise = _make_white_noise(tmp / "white_noise.wav")
    vessel = _make_tonal_vessel(tmp / "tonal_vessel.wav")

    # Case 1: white noise mislabelled as DARK_VESSEL.
    run_case("noise-as-vessel", noise, "DARK_VESSEL", 0.92)

    # Case 2: tonal vessel mislabelled as AMBIENT.
    run_case("vessel-as-ambient", vessel, "AMBIENT", 0.10)
