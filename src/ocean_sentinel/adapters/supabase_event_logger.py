"""Pushes one vessel-event detection through the public ingest gateway.

End-user installs do NOT carry a Supabase service-role key — that would
be a credential leak. Instead, the CLI POSTs to a public Edge Function
endpoint (`ingest-event`) which runs on Supabase's side with the service
role, validates the payload, rate-limits per IP, uploads PNGs to storage,
and inserts one row into `vessel_events`.

  CLI                       Edge Function                Supabase
   │                              │                          │
   │ POST /functions/v1/          │                          │
   │  ingest-event {event,        │                          │
   │   spectrogram_b64,           │                          │
   │   saliency_b64}              │                          │
   │ ─────────────────────────────►                          │
   │                              │ validate + rate-limit    │
   │                              │ upload PNGs (svc role)   │
   │                              │ ────────────────────────►│
   │                              │ insert vessel_events     │
   │                              │ ────────────────────────►│
   │ ◄─ {ok, id, spectrogram_url, │                          │
   │     saliency_url}            │                          │
   │                              │                          │

The CLI ships with the production endpoint as a default; operators who
self-host can override via OS_INGEST_URL. Failures are non-fatal — the
local events.jsonl is always written, so a network blip costs at most
one dashboard refresh, never a detection.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog

log = structlog.get_logger()


class SupabaseEventLogger:
    """Calls the public ingest-event Edge Function once per detection.

    No service-role secret on this side — the endpoint is intentionally
    callable without auth (rate-limited per IP server-side).
    """

    def __init__(
        self,
        ingest_url: str,
        cnn_checkpoint: str | Path = "data/models/cnn_v7_6.pt",
        timeout_s: float = 30.0,
    ) -> None:
        self._ingest_url = ingest_url.rstrip("/")
        self._cnn_checkpoint = str(cnn_checkpoint)
        self._timeout = timeout_s

    # ── artefact rendering ──────────────────────────────────────────────

    def _render_spectrogram(self, clip: Path) -> bytes | None:
        try:
            from ocean_sentinel.services.artifacts import render_spectrogram_png
            return render_spectrogram_png(clip)
        except Exception as e:
            log.warning("spectrogram_render_failed", clip=str(clip),
                        error=f"{type(e).__name__}: {e}")
            return None

    def _render_saliency(self, clip: Path, target_class: int) -> bytes | None:
        try:
            from ocean_sentinel.services.artifacts import render_saliency_png
            return render_saliency_png(
                clip, checkpoint=self._cnn_checkpoint,
                target_class=target_class,
            )
        except Exception as e:
            log.warning("saliency_render_failed", clip=str(clip),
                        error=f"{type(e).__name__}: {e}")
            return None

    # ── main entry ──────────────────────────────────────────────────────

    def push_event(
        self,
        *,
        event: dict[str, Any],
        clip: Path,
        site_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Render artefacts, POST one JSON payload to the gateway.

        Returns the gateway's response dict (or an error dict). Never
        raises — the caller's events.jsonl is the source of truth.
        """
        event_id = str(event.get("id") or event.get("decision_id") or "DET-?")
        site_id = str(event.get("site_id") or event.get("hydrophone") or "unknown")

        # Attribute Grad-CAM to whichever class the model picked, so the
        # overlay always explains the actual decision (not the ship class
        # unconditionally — which would be misleading for AMBIENT events).
        target_class = 1 if str(event.get("decision_tier")) in {
            "DARK_VESSEL", "CONFIRMED_VESSEL", "ACOUSTIC_ONLY_LOW",
        } else 0

        spec_png = self._render_spectrogram(clip)
        cam_png = self._render_saliency(clip, target_class=target_class)

        # Row payload — only the columns the gateway whitelists. Anything
        # extra is silently dropped server-side, but trim here for bandwidth.
        ais_offline_since = event.get("ais_offline_since")
        row = {
            "id":                    event_id,
            "ts":                    event.get("ts") or datetime.now(timezone.utc).isoformat(),
            "site_id":               site_id,
            "hydrophone":            event.get("hydrophone") or site_id,
            "vessel":                event.get("vessel"),
            "lat":                   _as_float(event.get("lat") or site_config.get("lat")),
            "lng":                   _as_float(event.get("lng") or site_config.get("lon")),
            "ais_status":            "DARK" if event.get("ais_vessels_in_radius", 0) == 0
                                     else "ONLINE",
            "ais_offline_since":     ais_offline_since,
            "ais_offline_seconds":   _seconds_offline(ais_offline_since),
            "ais_vessels_in_radius": int(event.get("ais_vessels_in_radius", 0) or 0),
            "mpa_name":              event.get("mpa_name") or site_config.get("nearest_mpa"),
            "mpa_distance_km":       _as_float(
                event.get("mpa_distance_km") or site_config.get("mpa_distance_km")
            ),
            "decision_tier":         event.get("decision_tier"),
            "severity":              event.get("severity"),
            "threat":                event.get("threat"),
            "confidence":            _as_float(event.get("confidence")),
            "cnn_confidence":        _as_float(event.get("cnn_confidence")),
            "cnn_uncertainty":       _as_float(event.get("cnn_uncertainty")),
            "conformal_threshold":   _as_float(event.get("conformal_threshold")),
            "conformal_pass":        event.get("conformal_pass"),
            "gemma_reasoning":       event.get("gemma_reasoning"),
            "narration_source":      event.get("narration_source"),
            "cnn_analysis":          event.get("cnn_analysis"),
            "raw":                   event,
        }

        payload: dict[str, Any] = {"event": row}
        if spec_png is not None:
            payload["spectrogram_b64"] = base64.b64encode(spec_png).decode("ascii")
        if cam_png is not None:
            payload["saliency_b64"] = base64.b64encode(cam_png).decode("ascii")

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._ingest_url,
                    content=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
            data = resp.json() if resp.headers.get("content-type", "").startswith(
                "application/json"
            ) else {"raw": resp.text}

            if resp.status_code == 200 and data.get("ok"):
                log.info("vessel_event_pushed", id=event_id, site=site_id,
                         spectrogram=bool(data.get("spectrogram_url")),
                         saliency=bool(data.get("saliency_url")))
                return data

            log.warning("vessel_event_push_rejected",
                        id=event_id, status=resp.status_code, detail=data)
            return {"ok": False, "status": resp.status_code, "error": data}
        except Exception as e:
            log.warning("vessel_event_push_failed", id=event_id,
                        error=f"{type(e).__name__}: {e}")
            return {"ok": False, "error": str(e)}


def _as_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _seconds_offline(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return int((datetime.now(timezone.utc) - t).total_seconds())
    except Exception:
        return None
