"""Persist one detection event in both places: locally + remotely.

Every CLI path that produces a real vessel-detection event (monitor,
detect, future replays of a live deployment) should funnel through here
so dashboard state stays consistent and the gateway is hit from exactly
one code path.

Local append (`data/sites/<site>/events.jsonl`) is always attempted —
it's the user's source-of-truth log, independent of network. The
gateway push is best-effort: if Supabase is unreachable, the row is
still on disk and the next run can sync.

What does NOT go through here:
- `os test` (bundled deterministic samples → would pollute dashboard
  with identical hash-derived DET-XXX IDs across every user install)
- `os alert-demo` (scripted demo with mocked AIS feeder)
- `os onboard` (the step-7 "test detection" is onboarding theatre)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_event(
    det: dict[str, Any],
    site_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Map a simulate_detection result + site config into the row shape
    the dashboard consumes. Includes templated gemma_reasoning derived
    from real spectral features (no fabricated numbers).
    """
    severity_to_threat = {"HIGH": "HIGH", "MEDIUM": "MEDIUM",
                          "LOW": "LOW", "NONE": "LOW"}
    threat = severity_to_threat.get(det.get("severity", "LOW"), "LOW")
    cnn_conf = float(det.get("cnn_confidence", 0.0))
    ais_n = int(det.get("ais_vessels_in_radius", 0) or 0)

    features: dict[str, Any] = {}
    reasoning: str = det.get("summary", "—")
    narration_source = "summary"
    try:
        from ..config import Settings
        from .event_narration import compute_features, gemma_reasoning
        s = Settings()
        clip_path = Path(det.get("clip", ""))
        if clip_path.exists():
            features = compute_features(clip_path)
            reasoning, narration_source = gemma_reasoning(
                features, det, config,
                ollama_host=s.ollama_base_url,
                ollama_model=s.gemma_model,
            )
    except Exception:
        pass  # narration is nice-to-have; never block the row

    return {
        # pipeline fields
        "ts":                   datetime.now(timezone.utc).isoformat(),
        "site_id":              site_id,
        "decision_tier":        det.get("decision_tier"),
        "severity":             det.get("severity"),
        "cnn_confidence":       cnn_conf,
        "cnn_uncertainty":      det.get("cnn_uncertainty"),
        "conformal_threshold":  det.get("conformal_threshold"),
        "conformal_pass":       det.get("conformal_pass"),
        "ais_vessels_in_radius": ais_n,
        "clip":                 det.get("clip"),
        "site_adapter":         det.get("site_adapter"),
        "features":             features,
        # display fields (frontend contract)
        "id":                   det.get("decision_id", "DET-?"),
        "vessel":               f"contact-{det.get('decision_id', '?')}",
        "lat":                  float(config.get("lat", 0.0) or 0.0),
        "lng":                  float(config.get("lon", 0.0) or 0.0),
        "threat":               threat,
        "confidence":           round(cnn_conf * 100.0, 1),
        "hydrophone":           site_id,
        "mpa_name":             str(config.get("nearest_mpa", "—")),
        "mpa_distance_km":      float(config.get("mpa_distance_km", 0.0) or 0.0),
        "ais_status":           "DARK" if ais_n == 0 else "ONLINE",
        "ais_offline_since":    datetime.now(timezone.utc).isoformat() if ais_n == 0 else None,
        "gemma_reasoning":      reasoning,
        "narration_source":     narration_source,
        "vessel_window":        f"{ais_n} AIS vessels in radius",
        "cnn_analysis":         (
            f"CNN ship_prob={cnn_conf:.2f}, "
            f"threshold={float(det.get('conformal_threshold', 0)):.2f}, "
            f"uncertainty={float(det.get('cnn_uncertainty', 0)):.2f}"
        ),
    }


# ── persistence sinks ───────────────────────────────────────────────────

def append_local(site_id: str, event: dict[str, Any]) -> Path:
    """Append a row to data/sites/{site_id}/events.jsonl. Idempotent;
    the dashboard's local mode reads this file directly."""
    import json
    site_dir = Path("data/sites") / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    events_path = site_dir / "events.jsonl"
    with events_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
    return events_path


_supabase_logger_cache: list = []  # one-slot lazy singleton


def _get_supabase_logger():
    if not _supabase_logger_cache:
        try:
            from ..config import Settings
            from ..adapters.supabase_event_logger import SupabaseEventLogger
            s = Settings()
            if not s.ingest_url:
                _supabase_logger_cache.append(None)
            else:
                _supabase_logger_cache.append(SupabaseEventLogger(
                    ingest_url=s.ingest_url,
                    cnn_checkpoint=s.cnn_checkpoint_path,
                ))
        except Exception:
            _supabase_logger_cache.append(None)
    return _supabase_logger_cache[0]


def push_remote(
    event: dict[str, Any], clip: Path, site_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Best-effort push through the public ingest gateway. Returns the
    gateway's response (or None when the logger isn't configured).
    Never raises — failures are logged inside the adapter."""
    logger = _get_supabase_logger()
    if logger is None:
        return None
    try:
        return logger.push_event(event=event, clip=clip, site_config=site_config)
    except Exception:
        return None


# ── one-stop call for command handlers ──────────────────────────────────

def persist_detection(
    det: dict[str, Any],
    site_id: str,
    clip: Path,
    site_config: dict[str, Any] | None = None,
    *,
    append_local_jsonl: bool = True,
    push_to_gateway: bool = True,
) -> dict[str, Any]:
    """Single entry point for any CLI command that produces a real
    detection event. Composes:

        det  →  vessel-event row  →  local jsonl  →  remote gateway

    Returns the event row. Both sinks are best-effort: the row is
    returned even if local append or remote push fails (caller can
    decide whether to surface that in CLI output).
    """
    cfg = site_config or {}
    event = build_event(det, site_id, cfg)

    if append_local_jsonl:
        try:
            append_local(site_id, event)
        except Exception:
            pass

    if push_to_gateway:
        push_remote(event, clip, cfg)

    return event
