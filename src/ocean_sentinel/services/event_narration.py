"""Narrate one detection for the dashboard's detail pane.

Produces `gemma_reasoning` — an analyst-paragraph (~3-5 sentences) built
from *real* spectral features extracted from the clip. Two paths:

  1. Local Ollama (gemma3n:e4b) when reachable — Gemma reads the numbers
     + site context and writes the paragraph itself. ~1-2 s per event.
  2. Templated fallback grounded in the same numbers, used when Ollama
     isn't running. Deterministic, instant, never invents data.

The CLI ships `narration_source` on each event so the dashboard / log
can tell which path produced the text. Jurors who follow the `os
onboard` quickstart already have Ollama running (onboard requires it),
so they get path 1 automatically — no extra setup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import structlog

log = structlog.get_logger()

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "gemma4:e4b"   # matches Settings.gemma_model default
_OLLAMA_TIMEOUT_S = 12.0               # cap per detection (Gemma 4 ~1-2s typical)


def compute_features(clip: Path) -> dict[str, Any]:
    """Real spectral features — same as gemma/explanations._compute_spectral_features."""
    from ocean_sentinel.gemma.explanations import _compute_spectral_features
    return _compute_spectral_features(clip)


# ── analyst-facing long-form reasoning ──────────────────────────────────

_NARRATION_PROMPT = """\
You are an analyst writing a 2-4 sentence summary of one acoustic
detection event for an ocean reserve operator's dashboard.

Use ONLY the numbers I give you. Do NOT invent blade-rate, RPM,
harmonic values, vessel class, or any other measurement that is not
listed below. Write plain analyst English. No headings, no bullets,
no markdown. Reference the decision tier and confidence directly.

DETECTION CONTEXT (JSON):
{context}

Write the paragraph now."""


def _build_context_json(
    features: dict[str, Any],
    decision: dict[str, Any],
    site_config: dict[str, Any],
) -> str:
    import json
    mpa_dist = site_config.get("mpa_distance_km")
    return json.dumps({
        "site_id":               decision.get("site_id"),
        "nearest_mpa":           site_config.get("nearest_mpa"),
        "mpa_distance_km":       float(mpa_dist) if isinstance(mpa_dist, (int, float)) else None,
        "decision_tier":         decision.get("decision_tier"),
        "severity":              decision.get("severity"),
        "ship_probability":      round(float(decision.get("cnn_confidence") or 0.0), 3),
        "site_threshold":        round(float(decision.get("conformal_threshold") or 0.0), 3),
        "uncertainty":           round(float(decision.get("cnn_uncertainty") or 0.0), 3),
        "ais_vessels_in_radius": int(decision.get("ais_vessels_in_radius") or 0),
        "recently_gone_dark":    int(decision.get("recently_gone_dark_count") or 0),
        "spectral_centroid_hz":  features.get("spectral_centroid_hz"),
        "peak_freq_hz":          features.get("peak_frequency_hz"),
        "low_band_fraction":     features.get("low_band_energy_fraction_below_200hz"),
        "spectral_flatness":     features.get("spectral_flatness"),
        "rms_db":                features.get("rms_db"),
    }, indent=2)


def _call_ollama(
    features: dict[str, Any],
    decision: dict[str, Any],
    site_config: dict[str, Any],
    *,
    host: str,
    model: str,
) -> str | None:
    """Ask the local Ollama-hosted Gemma to write the analyst paragraph.

    Returns the cleaned-up text on success, or None on any failure
    (Ollama not running, model missing, timeout, empty response). The
    caller falls back to the template when this returns None.
    """
    prompt = _NARRATION_PROMPT.format(
        context=_build_context_json(features, decision, site_config),
    )
    try:
        resp = httpx.post(
            f"{host.rstrip('/')}/api/chat",
            json={
                "model": model,
                "stream": False,
                # Disable Gemma 4's "thinking" preamble — we want the answer,
                # not the reasoning trace. Without this the model spends its
                # num_predict budget on `thinking` and returns content="".
                "think": False,
                "messages": [{"role": "user", "content": prompt}],
                # 512 leaves comfortable room for a 4-sentence paragraph
                # even if the model tokenises verbose words.
                "options": {"temperature": 0.2, "num_predict": 512},
            },
            timeout=_OLLAMA_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message") or {}
        text = (msg.get("content") or "").strip()
        # Gemma 4 sometimes emits the answer into `thinking` when `think`
        # isn't recognised by the runtime — accept either, but only if the
        # content channel is truly empty.
        if not text:
            text = (msg.get("thinking") or "").strip()
        if not text:
            return None
        # Strip markdown fences if the model adds them despite the prompt.
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: text.rfind("```")].rstrip()
        return text.strip() or None
    except Exception as e:
        log.info("ollama_narration_unavailable", error=f"{type(e).__name__}: {e}")
        return None


def gemma_reasoning(
    features: dict[str, Any],
    decision: dict[str, Any],
    site_config: dict[str, Any] | None = None,
    *,
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
    ollama_model: str = _DEFAULT_OLLAMA_MODEL,
    use_ollama: bool = True,
) -> tuple[str, str]:
    """Return `(reasoning_text, narration_source)`.

    Tries local Ollama (Gemma) first. Falls back to a deterministic
    template grounded in the same numbers. Either way the numbers are
    real — Gemma never sees a measurement we didn't measure.
    """
    if use_ollama:
        gemma_text = _call_ollama(
            features, decision, site_config or {},
            host=ollama_host, model=ollama_model,
        )
        if gemma_text:
            return gemma_text, "gemma_ollama"

    return _templated_reasoning(features, decision, site_config or {}), "templated"


def _templated_reasoning(
    features: dict[str, Any],
    decision: dict[str, Any],
    site_config: dict[str, Any],
) -> str:
    """Deterministic narrative used when Ollama is unavailable."""
    site_config = site_config or {}
    tier = str(decision.get("decision_tier", "UNCERTAIN"))
    severity = str(decision.get("severity", "—"))
    cnn_p = float(decision.get("cnn_confidence") or 0.0)
    threshold = float(decision.get("conformal_threshold") or 0.0)
    ais_n = int(decision.get("ais_vessels_in_radius") or 0)
    site_id = str(decision.get("site_id") or "unknown")

    mpa = site_config.get("nearest_mpa") or "uncharted area"
    mpa_dist = site_config.get("mpa_distance_km")

    location_clause = (
        f"{mpa} ({float(mpa_dist):.1f} km from the boundary)"
        if isinstance(mpa_dist, (int, float)) else f"the {mpa} area"
    )

    centroid = float(features.get("spectral_centroid_hz") or 0.0)
    pct_low = 100.0 * float(features.get("low_band_energy_fraction_below_200hz") or 0.0)

    if tier in {"DARK_VESSEL", "GONE_DARK_VESSEL", "CONFIRMED_VESSEL", "ACOUSTIC_ONLY_LOW"}:
        gone_dark = int(decision.get("recently_gone_dark_count") or 0)
        if tier == "GONE_DARK_VESSEL":
            ais_clause = (
                f"AIS shows {gone_dark} vessel(s) that were broadcasting in the last "
                "30 minutes and went dark — the classic AIS-off-before-MPA playbook."
            )
        else:
            ais_clause = (
                "AIS shows zero vessels in radius — this is an acoustic-only "
                "contact, no transponder broadcasting position."
                if ais_n == 0 else
                f"AIS shows {ais_n} transponder broadcasts in range; "
                "the acoustic signature is being cross-checked against those tracks."
            )
        return (
            f"Hydrophone {site_id} surfaced a {severity}-severity {tier} "
            f"event near {location_clause}. The CNN reports ship probability "
            f"{cnn_p:.2f} against a per-site conformal threshold of "
            f"{threshold:.2f}, so the detection clears the false-alarm "
            f"budget calibrated for this deployment. Acoustically, "
            f"{pct_low:.0f}% of clip energy falls below 200 Hz with a "
            f"spectral centroid of {centroid:.0f} Hz — the band where vessel "
            f"propulsion and engine noise concentrate. {ais_clause} "
            f"Recommended action: review the spectrogram and Grad-CAM "
            f"attention overlay, then decide on dispatch."
        )

    if tier == "AMBIENT":
        return (
            f"Hydrophone {site_id} processed a clip near {location_clause} "
            f"and the CNN scored it at ship probability {cnn_p:.2f}, below "
            f"the per-site threshold of {threshold:.2f}. The acoustic profile "
            f"is diffuse ({pct_low:.0f}% energy below 200 Hz, centroid "
            f"{centroid:.0f} Hz) — no dominant vessel signature. "
            f"No alert dispatched."
        )

    return (
        f"Hydrophone {site_id} returned an uncertain detection near "
        f"{location_clause}: ship probability {cnn_p:.2f} against threshold "
        f"{threshold:.2f}. Acoustic features are mixed ({pct_low:.0f}% energy "
        f"below 200 Hz, centroid {centroid:.0f} Hz). Flagged for analyst review "
        f"rather than auto-dispatched."
    )
