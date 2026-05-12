"""Real CNN v7.4 inference for simulate_detection.

Replaces the random-mock in tools.py with the actual production pipeline:

  audio file
    → librosa load (16 kHz, mono)
    → mel-spec (128 bands, log-dB, ref=1.0)
    → CNNV7Classifier.predict()  →  {ship_prob, uncertainty, vessel_type}
    → apply conformal threshold from data/calibration/conformal_v7_4.json
    → assemble decision tier (DARK_VESSEL / CONFIRMED_VESSEL / AMBIENT / UNCERTAIN)

This is the load-bearing claim of "Gemma orchestrates real ML." Without
it, the whole onboarding demo is theatre.

What we still mock for now (to be wired later):
- AIS lookup at the clip's timestamp (requires GFW token + timestamp).
- Per-site adapter weights (we use base v7.4 — adapter fine-tune is its
  own follow-up). When ctx.adapter_checkpoint exists we'd swap weights here.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np

# Match training preprocessing exactly. Drift here silently degrades calibration.
_TARGET_SR_HZ = 16_000
_N_MELS = 128
_FMAX_HZ = 1_000          # CNN v7 was trained with fmax=1kHz (low-band focus)
_DEFAULT_DURATION_S = 60.0  # CNN v7 expects ~60s windows
_DEFAULT_CHECKPOINT = "data/models/cnn_v7_6.pt"
_DEFAULT_CONFORMAL = "data/calibration/conformal_v7_6.json"
_DEFAULT_SITE_THRESHOLDS = "data/calibration/per_site_thresholds_v7_6.json"


@lru_cache(maxsize=2)
def _load_classifier(checkpoint: str):
    """Load the v7 classifier once per process. Cached because instantiation
    is heavy (model + weights + MPS device move).

    Auto-loads per-site thresholds from the default path when present —
    these correct balanced-sampler under-weighting for class-skewed sites
    (mbari, sanctsound, etc.). Falls back silently to default 0.5 when the
    calibration file is missing.
    """
    from ocean_sentinel.services.cnn_v7_classifier import CNNV7Classifier
    clf = CNNV7Classifier(checkpoint)
    thresholds_path = Path(_DEFAULT_SITE_THRESHOLDS)
    if thresholds_path.exists():
        clf.set_site_thresholds(thresholds_path)
    return clf


def _set_site_adapter_if_present(classifier, site_id: str) -> str | None:
    """If data/sites/{site_id}/adapter.pt exists, load it onto the
    classifier; otherwise clear any previously-set adapter. Returns the
    path string when loaded, None otherwise.
    """
    candidate = Path("data/sites") / site_id / "adapter.pt"
    if candidate.exists():
        classifier.set_site_adapter(candidate)
        return str(candidate)
    classifier.set_site_adapter(None)
    return None


@lru_cache(maxsize=2)
def _load_conformal(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _make_spec(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=_N_MELS, fmax=_FMAX_HZ,
    )
    return librosa.power_to_db(mel, ref=1.0)


def _decide_tier(
    ship_prob: float,
    conformal_pass: bool,
    uncertainty: float,
    ais_vessels_in_radius: int,
) -> tuple[str, str]:
    """Map raw model outputs → (decision_tier, severity).

    Mirrors the production policy in src/ocean_sentinel/decision/engine.py
    at a coarse level; finer gates can be layered on later.

    UNCERTAINTY_MAX = 0.25 was chosen empirically: scripts/benchmark_v7_4.py
    swept candidates 0.18-0.30 across n=35 held-out unseen samples and
    found 0.25 maximises confident-decision rate (100%) with no loss of
    accuracy (still 100% when the model decides). The earlier 0.20
    bisected the day-9 unc_mean distribution (0.16-0.22) and forced
    abstention on ~54% of borderline-confident clips. See
    data/eval/threshold_sweep.json for the full sweep table.
    """
    UNCERTAINTY_MAX = 0.25
    if uncertainty > UNCERTAINTY_MAX:
        return "UNCERTAIN", "MEDIUM"
    if not conformal_pass:
        return "AMBIENT", "NONE"
    if ais_vessels_in_radius == 0 and ship_prob >= 0.85:
        return "DARK_VESSEL", "HIGH"
    if ais_vessels_in_radius >= 1 and ship_prob >= 0.65:
        return "CONFIRMED_VESSEL", "LOW"
    return "ACOUSTIC_ONLY_LOW", "MEDIUM"


def simulate_detection(
    site_id: str,
    clip: str,
    *,
    checkpoint: str = _DEFAULT_CHECKPOINT,
    conformal_path: str = _DEFAULT_CONFORMAL,
    ais_vessels_in_radius: int = 0,    # mocked until GFW is wired
) -> dict[str, Any]:
    """Run the real v7.4 pipeline on `clip` for the given `site_id`.

    Returns the same shape as the previous mock, so callers don't change.
    """
    clip_path = Path(clip)
    if not clip_path.exists():
        return {"ok": False, "error": f"clip not found: {clip}"}

    try:
        y, sr = librosa.load(
            str(clip_path), sr=_TARGET_SR_HZ, mono=True,
            duration=_DEFAULT_DURATION_S,
        )
    except Exception as e:
        return {"ok": False, "error": f"failed to load clip: {type(e).__name__}: {e}"}

    if y.size < sr * 5:
        return {"ok": False, "error": f"clip too short: {y.size / sr:.1f}s"}

    spec = _make_spec(y, sr)

    try:
        classifier = _load_classifier(checkpoint)
    except FileNotFoundError:
        return {"ok": False, "error": f"checkpoint not found: {checkpoint}"}

    # If this site has a trained adapter, splice it in for this prediction.
    site_adapter_path = _set_site_adapter_if_present(classifier, site_id)

    try:
        conformal = _load_conformal(conformal_path)
    except FileNotFoundError:
        return {"ok": False, "error": f"conformal calibration not found: {conformal_path}"}

    pred = classifier.predict(spec, source_id=site_id)
    label = pred.get("label", "not_ship")
    confidence = float(pred.get("confidence", 0.5))
    ship_prob = confidence if label == "ship" else (1.0 - confidence)
    uncertainty = float(pred.get("uncertainty", 0.0))

    threshold = float(conformal["threshold"])
    conformal_pass = ship_prob >= threshold

    tier, severity = _decide_tier(
        ship_prob=ship_prob,
        conformal_pass=conformal_pass,
        uncertainty=uncertainty,
        ais_vessels_in_radius=ais_vessels_in_radius,
    )

    decision_id = f"DET-{abs(hash(clip + site_id)) % 100000:05d}"

    record = {
        "ok": True,
        "decision_id": decision_id,
        "site_id": site_id,
        "clip": str(clip_path),
        "cnn_label": label,
        "cnn_confidence": round(ship_prob, 3),
        "cnn_uncertainty": round(uncertainty, 3),
        "conformal_threshold": round(threshold, 3),
        "conformal_p": round(ship_prob, 3),
        "conformal_pass": bool(conformal_pass),
        "ais_vessels_in_radius": ais_vessels_in_radius,
        "decision_tier": tier,
        "severity": severity,
        "checkpoint": checkpoint,
        "site_adapter": site_adapter_path,
        "summary": (
            f"{tier} ({severity}) · CNN p={ship_prob:.2f} "
            f"vs conformal {threshold:.2f} · AIS {ais_vessels_in_radius}"
        ),
    }

    try:
        from .explanations import write_decision_record
        write_decision_record(decision_id, record)
    except Exception:
        pass

    return record
