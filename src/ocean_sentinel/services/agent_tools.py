"""Agent tools for Gemma function-calling.

Each tool is a pure async function. The Toolbox wires runtime dependencies
(event store, settings) and dispatches by name. Schemas use OpenAI/Ollama-
compatible JSON Schema — the same shape Gemini accepts with light translation.

Three tools, each addressing one threat-assessment dimension:

  check_mpa_proximity         — spatial: is this point inside an MPA?
  lookup_vessel_registry      — identity: who is this MMSI, IUU history?
  query_recent_detections     — temporal: pattern-of-activity in region?
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from ocean_sentinel.adapters.persistence import SQLiteEventStore
from ocean_sentinel.domain.models import TimeWindow

log = structlog.get_logger()


# ──────────────────────────────────────────────────────────────────────────
# 1. MPA proximity — geo lookup against curated MPA list near our hydrophones.
#    Production: WDPA shapefile + shapely point-in-polygon. Demo: centroid +
#    radius approximation, sufficient for the MPAs we deploy in.
# ──────────────────────────────────────────────────────────────────────────

_MPA_REGISTRY: list[dict[str, Any]] = [
    {"name": "Cordell Bank NMS",         "centroid": (38.05, -123.42), "radius_km": 28, "protection": "no-take",    "iucn": "II"},
    {"name": "Greater Farallones NMS",   "centroid": (37.70, -123.20), "radius_km": 55, "protection": "no-take",    "iucn": "II"},
    {"name": "Monterey Bay NMS",         "centroid": (36.80, -122.05), "radius_km": 80, "protection": "multi-use",  "iucn": "VI"},
    {"name": "Channel Islands NMS",      "centroid": (33.95, -119.80), "radius_km": 45, "protection": "partial",    "iucn": "IV"},
    {"name": "Olympic Coast NMS",        "centroid": (47.95, -124.85), "radius_km": 65, "protection": "multi-use",  "iucn": "VI"},
    {"name": "Stellwagen Bank NMS",      "centroid": (42.40,  -70.40), "radius_km": 30, "protection": "multi-use",  "iucn": "VI"},
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. Earth radius 6371."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


async def check_mpa_proximity(lat: float, lon: float) -> dict:
    """Find nearest MPA and whether (lat, lon) falls inside its radius."""
    nearest = min(
        _MPA_REGISTRY,
        key=lambda m: _haversine_km(lat, lon, m["centroid"][0], m["centroid"][1]),
    )
    distance = _haversine_km(lat, lon, nearest["centroid"][0], nearest["centroid"][1])
    return {
        "nearest_mpa": nearest["name"],
        "distance_km": round(distance, 2),
        "inside": distance <= nearest["radius_km"],
        "protection_level": nearest["protection"],
        "iucn_category": nearest["iucn"],
    }


# ──────────────────────────────────────────────────────────────────────────
# 2. Vessel registry — curated mock for demo. Production: GFW
#    /v3/vessels?mmsis=<mmsi>. Includes recidivists so the demo can show
#    threat escalation based on history.
# ──────────────────────────────────────────────────────────────────────────

_VESSEL_REGISTRY: dict[str, dict[str, Any]] = {
    "224157000": {"name": "PESCA NORTE",       "flag": "Spain",   "vessel_type": "trawler",     "length_m": 47,  "owner": "Pesquera Atlantica SA",      "prior_iuu_events": 3, "last_iuu_date": "2025-09-12"},
    "311042900": {"name": "STARLIGHT VOYAGER", "flag": "Bahamas", "vessel_type": "cargo",       "length_m": 182, "owner": "Starlight Maritime Ltd",     "prior_iuu_events": 0, "last_iuu_date": None},
    "367719270": {"name": "PACIFIC HARVEST",   "flag": "USA",     "vessel_type": "purse_seine", "length_m": 31,  "owner": "Pacific Harvest Co-op",      "prior_iuu_events": 1, "last_iuu_date": "2024-06-03"},
    "412549870": {"name": "JOSCO HUIZHOU",     "flag": "China",   "vessel_type": "cargo",       "length_m": 199, "owner": "JOSCO Shipping Ltd",         "prior_iuu_events": 0, "last_iuu_date": None},
    "636092115": {"name": "SEA NOMAD",         "flag": "Liberia", "vessel_type": "trawler",     "length_m": 62,  "owner": "Anchor Holdings Ltd (FOC)",  "prior_iuu_events": 5, "last_iuu_date": "2026-01-22"},
}


async def lookup_vessel_registry(mmsi: str | int) -> dict:
    """Look up vessel by MMSI. Returns metadata + IUU history, or `found=False`.

    Accepts int or str — Gemma occasionally encodes the 9-digit MMSI as a
    JSON number, even though the schema requests a string.
    """
    mmsi = str(mmsi)
    record = _VESSEL_REGISTRY.get(mmsi)
    if record is None:
        return {
            "mmsi": mmsi,
            "found": False,
            "note": "MMSI not in registry — vessel may be unregistered or use a spoofed ID.",
        }
    return {"mmsi": mmsi, "found": True, **record}


# ──────────────────────────────────────────────────────────────────────────
# 3. Recent detections in region — real query against SQLite event store.
# ──────────────────────────────────────────────────────────────────────────

async def query_recent_detections(
    store: SQLiteEventStore,
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    hours: int = 24,
) -> dict:
    """List detections within radius_km of (lat, lon) in the last `hours` hours."""
    end = datetime.now(timezone.utc)
    window = TimeWindow(start=end - timedelta(hours=hours), end=end)

    events = await store.list_events(time_window=window)

    nearby = []
    for ev in events:
        d = _haversine_km(lat, lon, ev.location.lat, ev.location.lon)
        if d <= radius_km:
            nearby.append({
                "timestamp": ev.timestamp.isoformat(),
                "distance_km": round(d, 2),
                "threat_level": ev.threat_level.value,
                "confidence": round(ev.confidence, 2),
            })

    high_count = sum(1 for n in nearby if n["threat_level"] in ("HIGH", "CRITICAL"))

    return {
        "count": len(nearby),
        "detections": nearby[:10],
        "persistent_activity": high_count >= 3,
        "window_hours": hours,
        "radius_km": radius_km,
    }


# ──────────────────────────────────────────────────────────────────────────
# JSON Schemas — Ollama / OpenAI tool calling format.
# ──────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_mpa_proximity",
            "description": (
                "Find the nearest marine protected area (MPA) for a lat/lon. "
                "Returns nearest MPA name, distance in km, whether the point "
                "is inside the MPA, protection level (no-take/partial/multi-use), "
                "and IUCN category. Call this whenever a vessel is detected — "
                "MPA status is decisive for threat severity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude in degrees N"},
                    "lon": {"type": "number", "description": "Longitude in degrees E (negative for W)"},
                },
                "required": ["lat", "lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_vessel_registry",
            "description": (
                "Look up a vessel by its MMSI (9-digit AIS broadcast ID). "
                "Returns name, flag, vessel type, length, owner, and prior "
                "IUU (illegal/unreported/unregulated fishing) event history. "
                "Call this when an AIS gap event has a known vessel_id — "
                "vessel history can escalate or downgrade threat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mmsi": {"type": "string", "description": "9-digit MMSI string"},
                },
                "required": ["mmsi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_recent_detections",
            "description": (
                "Query past detection events near a location to identify "
                "pattern-of-activity. Returns count, list of nearby detections, "
                "and a `persistent_activity` flag (true when ≥3 HIGH/CRITICAL "
                "detections in the window). Use this to distinguish one-off "
                "transit from sustained operation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_km": {"type": "number", "description": "Search radius (default 10 km)"},
                    "hours": {"type": "integer", "description": "Look-back window in hours (default 24)"},
                },
                "required": ["lat", "lon"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────
# Toolbox — holds wired deps, dispatches by name.
# ──────────────────────────────────────────────────────────────────────────

class Toolbox:
    """Wires runtime dependencies for tool execution and dispatches by name."""

    def __init__(self, event_store: SQLiteEventStore | None = None) -> None:
        self._event_store = event_store

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name. Returns JSON string for the tool message."""
        try:
            if name == "check_mpa_proximity":
                result = await check_mpa_proximity(**args)
            elif name == "lookup_vessel_registry":
                result = await lookup_vessel_registry(**args)
            elif name == "query_recent_detections":
                if self._event_store is None:
                    result = {"error": "event_store not configured for this Toolbox"}
                else:
                    result = await query_recent_detections(self._event_store, **args)
            else:
                result = {"error": f"unknown tool: {name}"}
        except Exception as exc:  # pragma: no cover — defensive: never crash agent loop
            log.warning("tool_dispatch_error", tool=name, args=args, error=str(exc))
            result = {"error": f"tool execution failed: {exc}"}

        log.info("tool_dispatched", tool=name, args=args)
        return json.dumps(result)
