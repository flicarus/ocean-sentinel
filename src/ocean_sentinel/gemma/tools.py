"""Tool definitions for Gemma's function-calling onboarding.

Each tool exposes:
- name + description (for Gemma to choose it)
- JSON schema for parameters (Ollama / OpenAI tool-call format)
- a Python implementation that returns a JSON-serializable dict

Tools are split into 5 categories matching the 8-step onboarding flow:

  Discovery & validation
    - validate_site_coords
    - fetch_hydrophone_metadata
    - fetch_ais_baseline

  Acoustic fingerprinting
    - record_ambient
    - compute_spectral_signature

  Transfer learning & calibration
    - compare_to_known_sites
    - select_adapter_strategy
    - finetune_adapter
    - calibrate_conformal

  Configuration & registration
    - set_alert_policy
    - register_site

  Test & explain
    - simulate_detection
    - explain_decision
    - flag_for_review

Implementations are intentionally lightweight so the demo runs without
external services. The Tool contract is stable — swap in real impls
(GFW, librosa, torch fine-tune) one at a time without changing call sites.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

# ── Tool record ─────────────────────────────────────────────────────────


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        """Return Ollama / OpenAI tool-call JSON schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


SITES_DIR = Path("data/sites")


# ── 1. validate_site_coords ─────────────────────────────────────────────
def _validate_site_coords(lat: float, lon: float) -> dict[str, Any]:
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return {"ok": False, "error": "coordinates out of range"}
    # naive ocean check: if continental US bounding box and lon > -100, likely land
    on_land_guess = (-100.0 < lon < -70.0) and (28.0 < lat < 49.0)
    depth_m = max(50, int(500 + math.cos(lat) * 200 + random.randint(-50, 50)))
    nearest_mpa = "MBNMS" if -125 < lon < -120 and 35 < lat < 38 else "n/a"
    return {
        "ok": True,
        "ocean_confirmed": not on_land_guess,
        "depth_m": depth_m,
        "nearest_mpa": nearest_mpa,
        "summary": f"ocean confirmed · {depth_m}m depth · {nearest_mpa} boundary nearby",
    }


# ── 2. fetch_hydrophone_metadata ────────────────────────────────────────
def _fetch_hydrophone_metadata(url: str) -> dict[str, Any]:
    known = {
        "orcasound": {"network": "Orcasound", "sample_rate_hz": 48000, "format": "HLS .ts"},
        "mbari":     {"network": "MBARI MARS", "sample_rate_hz": 256000, "format": "WAV"},
        "ocean":     {"network": "Ocean Networks Canada", "sample_rate_hz": 64000, "format": "WAV"},
    }
    for key, meta in known.items():
        if key in url.lower():
            return {
                "ok": True, **meta, "source_url": url,
                "summary": f"{meta['network']} · {meta['sample_rate_hz'] // 1000} kHz · {meta['format']}",
            }
    return {
        "ok": True,
        "network": "unknown",
        "sample_rate_hz": 48000,
        "format": "unknown",
        "source_url": url,
        "summary": "unknown network · assuming 48 kHz",
    }


# ── 3. fetch_ais_baseline ───────────────────────────────────────────────
def _fetch_ais_baseline(lat: float, lon: float, radius_km: int = 10, days: int = 30) -> dict[str, Any]:
    # Plausible mock: high traffic if near US west coast shipping lanes
    traffic = "high" if -125 < lon < -118 and 30 < lat < 50 else "low"
    avg = {"high": 47, "medium": 18, "low": 4}[traffic if traffic in ["high", "low"] else "low"]
    lane_distance = round(abs(lon + 122.0) * 30 + 2.1, 1) if traffic == "high" else None
    return {
        "ok": True,
        "avg_vessels_per_day": avg,
        "vessel_mix": {"tanker": 0.38, "fishing": 0.29, "cargo": 0.22, "other": 0.11},
        "shipping_lane_km": lane_distance,
        "transit_interval_min": 38 if traffic == "high" else None,
        "peak_hours_local": "14:00-18:00",
        "traffic_class": traffic,
        "summary": f"{avg} vessels/day · {traffic} traffic"
                   + (f" · lane {lane_distance}km" if lane_distance else ""),
    }


# ── 4. record_ambient  [REAL — librosa probe] ──────────────────────────
def _record_ambient(source: str, duration_min: int = 5) -> dict[str, Any]:
    """Verify the audio source is loadable and report real metadata.

    Doesn't keep the buffer — compute_spectral_signature reads it again.
    Costs little extra time (just a header probe) and gives the user fast
    feedback that their file/URL is valid before we move to fingerprinting.
    """
    from pathlib import Path
    p = Path(source)
    if source.startswith(("http://", "https://")):
        return {
            "ok": True,
            "source": source,
            "is_stream": True,
            "summary": f"stream URL accepted (network probe deferred to spec stage)",
        }
    if not p.exists():
        return {"ok": False, "error": f"audio file not found: {source}"}
    try:
        import librosa
        # `librosa.get_duration` reads the header without decoding — fast.
        duration_s = librosa.get_duration(path=str(p))
        sr = librosa.get_samplerate(str(p))
    except Exception as e:
        return {"ok": False, "error": f"failed to probe audio: {type(e).__name__}: {e}"}
    return {
        "ok": True,
        "source": source,
        "duration_seconds": round(float(duration_s), 1),
        "sample_rate_hz": int(sr),
        "is_stream": False,
        "summary": f"loaded {duration_s:.1f}s · {sr // 1000} kHz · file ok",
    }


# ── 5. compute_spectral_signature  [REAL — librosa] ─────────────────────
def _compute_spectral_signature(audio: str, n_bands: int = 64) -> dict[str, Any]:
    """Real impl — see ocean_sentinel.gemma.audio_features."""
    from .audio_features import compute_spectral_signature
    return compute_spectral_signature(audio_source=audio, n_mels=n_bands)


# ── 6. compare_to_known_sites  [REAL — cosine vs precomputed registry] ──
def _compare_to_known_sites(signature: list[float]) -> dict[str, Any]:
    """Real impl — see ocean_sentinel.gemma.known_sites.

    Requires data/known_sites.json (built by scripts/precompute_signatures.py).
    Falls back to a clear error if not yet bootstrapped."""
    from .known_sites import compare_to_known_sites
    return compare_to_known_sites(signature)


# Legacy stub kept for compatibility — never reached at runtime.
def _compare_to_known_sites_legacy(signature: list[float]) -> dict[str, Any]:
    rng = random.Random(int(sum(signature) * 1000) & 0xFFFFFFFF if signature else 0)
    ranked = []
    for site in [{"id": "mbari-mars", "label": "MBARI MARS", "lat": 36.71, "lon": -122.19}]:
        sim = round(rng.uniform(0.30, 0.90), 2)
        ranked.append({**site, "cosine_sim": sim})
    ranked.sort(key=lambda r: r["cosine_sim"], reverse=True)
    top = ranked[0]
    recommendation = (
        "use_existing"  if top["cosine_sim"] > 0.85
        else "finetune" if top["cosine_sim"] > 0.65
        else "full_calibration"
    )
    return {
        "ok": True,
        "ranked": ranked[:3],
        "recommendation": recommendation,
        "summary": f"closest: {top['label']} (cos {top['cosine_sim']:.2f}) → {recommendation}",
    }


# ── 7. select_adapter_strategy ──────────────────────────────────────────
def _select_adapter_strategy(similarity: float) -> dict[str, Any]:
    if similarity > 0.85:
        strategy, epochs, lr = "none", 0, 0.0
    elif similarity > 0.65:
        strategy, epochs, lr = "finetune_last2", 10, 3e-4
    else:
        strategy, epochs, lr = "full_retrain", 30, 1e-4
    return {
        "ok": True,
        "strategy": strategy,
        "epochs": epochs,
        "learning_rate": lr,
        "summary": f"strategy={strategy}, epochs={epochs}",
    }


# ── 8. finetune_adapter  [REAL — label-free validation on ambient] ─────
def _finetune_adapter(
    site_id: str,
    ambient_source: str | None = None,
    epochs: int = 10,
    lr: float = 3e-4,
) -> dict[str, Any]:
    """Real per-site adapter step. Without labelled ship+ambient pairs we
    can't do supervised fine-tuning on the user's data, so this step
    instead VALIDATES the base v7.4 against the user's ambient and reports
    real recall. Per-site adaptation is split between this validation
    (here) and per-site threshold calibration (Step 5)."""
    del epochs, lr  # accepted for back-compat with the schema but unused
    if not ambient_source:
        return {
            "ok": False,
            "error": "ambient_source required — provide the user's ambient .wav from Step 2",
        }
    from .adapter import validate_adapter_on_ambient
    return validate_adapter_on_ambient(site_id=site_id, ambient_source=ambient_source)


# ── 9. calibrate_conformal  [REAL — split-conformal on ambient] ────────
def _calibrate_conformal(
    site_id: str,
    ambient_source: str,
    alpha: float = 0.05,
    n_samples: int = 200,   # accepted for back-compat; real impl auto-sizes
) -> dict[str, Any]:
    """Real split-conformal calibration on user's ambient audio.

    See ocean_sentinel.gemma.conformal for the algorithm + theoretical
    guarantees. The previous mock returned a fixed threshold; this one
    derives it from v7.4 outputs on N=30+ ambient windows.
    """
    del n_samples  # not used — real impl auto-sizes from audio length
    from .conformal import calibrate_conformal_real
    return calibrate_conformal_real(
        site_id=site_id,
        ambient_source=ambient_source,
        alpha=alpha,
    )


# ── 10. set_alert_policy ────────────────────────────────────────────────
def _set_alert_policy(site_id: str, sensitivity: str = "medium",
                      email: str | None = None) -> dict[str, Any]:
    if sensitivity not in {"high", "medium", "low"}:
        return {"ok": False, "error": f"sensitivity must be high|medium|low, got {sensitivity}"}
    threshold_adjust = {"high": -0.05, "medium": 0.0, "low": +0.05}[sensitivity]
    return {
        "ok": True,
        "site_id": site_id,
        "sensitivity": sensitivity,
        "threshold_adjust": threshold_adjust,
        "email": email or None,
        "summary": f"sensitivity={sensitivity}, email={email or '—'}",
    }


# ── 11. register_site ───────────────────────────────────────────────────
def _register_site(site_id: str, config: dict[str, Any]) -> dict[str, Any]:
    SITES_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = SITES_DIR / f"{site_id}.yaml"
    yaml_path.write_text(yaml.safe_dump({"site_id": site_id, **config}, sort_keys=False))
    return {
        "ok": True,
        "site_id": site_id,
        "path": str(yaml_path),
        "summary": f"wrote {yaml_path}",
    }


# ── 12. simulate_detection  [REAL — CNN v7.4 + conformal threshold] ─────
def _simulate_detection(site_id: str, clip: str) -> dict[str, Any]:
    """Real impl — see ocean_sentinel.gemma.cnn_inference.

    Loads v7.4 from data/models/cnn_v7_4.pt, applies the conformal threshold
    from data/calibration/conformal_v7_4.json, and assembles the decision tier.

    AIS lookup is still mocked here (ais_vessels_in_radius=0). Once the GFW
    adapter is wired into cnn_inference, the call signature stays the same."""
    from .cnn_inference import simulate_detection
    return simulate_detection(site_id=site_id, clip=clip)


# Legacy stub — kept for fallback when CNN/conformal files are missing.
def _simulate_detection_legacy(site_id: str, clip: str) -> dict[str, Any]:
    cnn_conf = round(random.uniform(0.65, 0.92), 2)
    threshold = 0.71
    p_value = round(random.uniform(threshold + 0.02, 0.95), 2)
    ais_count = random.choice([0, 0, 0, 1, 2])
    if cnn_conf > 0.8 and p_value > threshold and ais_count == 0:
        tier, severity = "DARK_VESSEL", "HIGH"
    elif cnn_conf > 0.6 and ais_count == 0:
        tier, severity = "ACOUSTIC_ONLY_LOW", "MEDIUM"
    elif ais_count > 0:
        tier, severity = "CONFIRMED_VESSEL", "LOW"
    else:
        tier, severity = "AMBIENT", "NONE"
    return {
        "ok": True,
        "decision_id": f"DRY-RUN-{random.randint(1000, 9999)}",
        "cnn_confidence": cnn_conf,
        "conformal_p": p_value,
        "conformal_pass": p_value > threshold,
        "ais_vessels_in_radius": ais_count,
        "decision_tier": tier,
        "severity": severity,
        "summary": f"{tier} ({severity}) · CNN {cnn_conf} · AIS {ais_count}",
    }


# ── 13. explain_decision  [REAL — Gemma multimodal on spectrogram] ─────
def _explain_decision(decision_id: str, modality: str = "spectrogram+text") -> dict[str, Any]:
    """Real impl — see ocean_sentinel.gemma.explanations.

    Reads the decision record persisted by simulate_detection, renders the
    clip's log-mel spectrogram, computes real spectral features, and asks
    Gemma 4 multimodal for a 1-2 sentence operator-friendly explanation
    grounded in the image. Falls back to a templated narration (still using
    real measured features) if Ollama is unreachable."""
    from .explanations import explain_decision_real, is_multimodal_disabled
    return explain_decision_real(
        decision_id=decision_id,
        modality=modality,
        use_multimodal=not is_multimodal_disabled(),
    )


# ── 14. flag_for_review ─────────────────────────────────────────────────
def _flag_for_review(decision_id: str, reason: str) -> dict[str, Any]:
    queue_path = Path("data/review_queue.jsonl")
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a") as f:
        f.write(f'{{"decision_id": "{decision_id}", "reason": "{reason}"}}\n')
    return {
        "ok": True,
        "decision_id": decision_id,
        "reason": reason,
        "queued_at": queue_path.stat().st_size,
        "summary": f"queued {decision_id} for review",
    }


# ── Tool registry ───────────────────────────────────────────────────────
TOOLS: list[Tool] = [
    Tool(
        name="validate_site_coords",
        description="Validate latitude/longitude — confirm ocean, return depth and nearest MPA.",
        parameters={
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude in degrees, -90..90"},
                "lon": {"type": "number", "description": "Longitude in degrees, -180..180"},
            },
            "required": ["lat", "lon"],
        },
        fn=_validate_site_coords,
    ),
    Tool(
        name="fetch_hydrophone_metadata",
        description="Resolve a hydrophone stream URL — sample rate, network, format.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP(S) stream URL or known hydrophone ID"},
            },
            "required": ["url"],
        },
        fn=_fetch_hydrophone_metadata,
    ),
    Tool(
        name="fetch_ais_baseline",
        description="Get 30-day AIS baseline (vessel density, traffic class, shipping lanes) from Global Fishing Watch around the site.",
        parameters={
            "type": "object",
            "properties": {
                "lat":        {"type": "number"},
                "lon":        {"type": "number"},
                "radius_km":  {"type": "integer", "default": 10},
                "days":       {"type": "integer", "default": 30},
            },
            "required": ["lat", "lon"],
        },
        fn=_fetch_ais_baseline,
    ),
    Tool(
        name="record_ambient",
        description="Load N minutes of ambient hydrophone audio from a file path or stream URL.",
        parameters={
            "type": "object",
            "properties": {
                "source":       {"type": "string", "description": "File path or stream URL"},
                "duration_min": {"type": "integer", "default": 5},
            },
            "required": ["source"],
        },
        fn=_record_ambient,
    ),
    Tool(
        name="compute_spectral_signature",
        description="Compute a 64-band spectral signature for ambient audio — used to fingerprint the site.",
        parameters={
            "type": "object",
            "properties": {
                "audio":   {"type": "string", "description": "Audio buffer reference or file path"},
                "n_bands": {"type": "integer", "default": 64},
            },
            "required": ["audio"],
        },
        fn=_compute_spectral_signature,
    ),
    Tool(
        name="compare_to_known_sites",
        description="Find the most acoustically similar known training sites — returns ranked list with cosine similarity and a recommended adapter strategy.",
        parameters={
            "type": "object",
            "properties": {
                "signature": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "64-element spectral signature from compute_spectral_signature.",
                },
            },
            "required": ["signature"],
        },
        fn=_compare_to_known_sites,
    ),
    Tool(
        name="select_adapter_strategy",
        description="Given the highest cosine similarity to a known site, choose: none / finetune_last2 / full_retrain.",
        parameters={
            "type": "object",
            "properties": {
                "similarity": {"type": "number", "description": "0..1 cosine similarity"},
            },
            "required": ["similarity"],
        },
        fn=_select_adapter_strategy,
    ),
    Tool(
        name="finetune_adapter",
        description="Per-site adapter validation — runs v7.4 on the user's ambient and reports the real recall (% correctly classified as not_ship). Label-free: doesn't update model weights but verifies the base model performs adequately on the user's site before threshold calibration in Step 5.",
        parameters={
            "type": "object",
            "properties": {
                "site_id":        {"type": "string"},
                "ambient_source": {"type": "string", "description": "Path to user's ambient .wav from Step 2"},
                "epochs":         {"type": "integer", "default": 10, "description": "(reserved; not used in label-free mode)"},
                "lr":             {"type": "number",  "default": 3e-4, "description": "(reserved; not used in label-free mode)"},
            },
            "required": ["site_id", "ambient_source"],
        },
        fn=_finetune_adapter,
    ),
    Tool(
        name="calibrate_conformal",
        description="Run split-conformal calibration on the user's ambient audio to set a per-site false-alarm threshold with provable guarantees. Writes data/sites/{site_id}/conformal.json.",
        parameters={
            "type": "object",
            "properties": {
                "site_id":        {"type": "string"},
                "ambient_source": {"type": "string", "description": "Path to user's ambient .wav from Step 2"},
                "alpha":          {"type": "number",  "default": 0.05, "description": "Target false-alarm rate; smaller = more conservative"},
            },
            "required": ["site_id", "ambient_source"],
        },
        fn=_calibrate_conformal,
    ),
    Tool(
        name="set_alert_policy",
        description="Configure alert sensitivity and notification channels for a site.",
        parameters={
            "type": "object",
            "properties": {
                "site_id":     {"type": "string"},
                "sensitivity": {"type": "string", "enum": ["high", "medium", "low"]},
                "email":       {"type": "string", "description": "optional"},
            },
            "required": ["site_id", "sensitivity"],
        },
        fn=_set_alert_policy,
    ),
    Tool(
        name="register_site",
        description="Persist the resolved site config to data/sites/{site_id}.yaml. Call this once everything else is configured.",
        parameters={
            "type": "object",
            "properties": {
                "site_id": {"type": "string", "description": "kebab-case identifier"},
                "config":  {"type": "object", "description": "all site config to persist"},
            },
            "required": ["site_id", "config"],
        },
        fn=_register_site,
    ),
    Tool(
        name="simulate_detection",
        description="Run the full pipeline (CNN + conformal + AIS) on a test audio clip — returns the decision tier and a trace.",
        parameters={
            "type": "object",
            "properties": {
                "site_id": {"type": "string"},
                "clip":    {"type": "string", "description": "path to test audio clip"},
            },
            "required": ["site_id", "clip"],
        },
        fn=_simulate_detection,
    ),
    Tool(
        name="explain_decision",
        description="Explain a past decision: render the clip's spectrogram and ask Gemma 4 multimodal for a short, grounded narration. Returns real spectral features (peak Hz, centroid, low-band energy) plus a 1-2 sentence explanation.",
        parameters={
            "type": "object",
            "properties": {
                "decision_id": {"type": "string", "description": "decision_id returned by simulate_detection"},
                "modality":    {"type": "string", "enum": ["text", "spectrogram", "spectrogram+text"], "default": "spectrogram+text"},
            },
            "required": ["decision_id"],
        },
        fn=_explain_decision,
    ),
    Tool(
        name="flag_for_review",
        description="Append a decision to the human-review queue with a reason (e.g. user-disputed false alarm).",
        parameters={
            "type": "object",
            "properties": {
                "decision_id": {"type": "string"},
                "reason":      {"type": "string"},
            },
            "required": ["decision_id", "reason"],
        },
        fn=_flag_for_review,
    ),
]


TOOLS_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def all_schemas() -> list[dict[str, Any]]:
    """Return every tool's JSON schema, ready to pass to ollama.chat(tools=...)."""
    return [t.schema() for t in TOOLS]


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up a tool by name and call it with the supplied arguments.
    Returns the tool's result dict, or an error dict if the name is unknown
    or arguments are malformed."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    try:
        return tool.fn(**arguments)
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"{name} failed: {type(e).__name__}: {e}"}
