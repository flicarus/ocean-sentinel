"""Vessel-event feed for the Next.js dashboard.

Matches the contract in `~/oceansentinelfrontend/lib/ocean-sentinel.ts`
exactly so `useEvents.ts` can poll this endpoint instead of using its
hardcoded seed data. The frontend already knows this shape; we just need
to ship it from the backend.

Schema (mirrors VesselEvent on the frontend, by-the-letter):

  id              str
  vessel          str    e.g. "FV-8872"
  lat             float
  lng             float       (note: "lng", not "lon" — that's the frontend's choice)
  threat          "HIGH"|"MEDIUM"|"LOW"
  confidence      float       0..100
  ais_offline_since   ISO-8601 str
  hydrophone      str    e.g. "MBARI-01"
  time_ago        str    human readable, computed at response time
  mpa_name        str
  mpa_distance_km float
  gemma_reasoning str    long narrative
  vessel_window   str    AIS context one-liner
  cnn_analysis    str    CNN output one-liner

For day 12 we serve the same set of curated demo events the frontend
seeds with — keeping behavior identical for the dashboard. The future
swap to live event-store data is a single function (`_from_detection`),
documented inline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()


class VesselEvent(BaseModel):
    id: str
    vessel: str
    lat: float
    lng: float
    threat: Literal["HIGH", "MEDIUM", "LOW"]
    confidence: float
    ais_offline_since: str
    hydrophone: str
    time_ago: str
    mpa_name: str
    mpa_distance_km: float
    gemma_reasoning: str
    vessel_window: str
    cnn_analysis: str


def _ago_iso(seconds_ago: int) -> str:
    return (datetime.now(timezone.utc) - _td(seconds_ago)).isoformat()


def _td(seconds: int):
    from datetime import timedelta
    return timedelta(seconds=seconds)


def _humanize_seconds(s: int) -> str:
    if s < 60:        return f"{s} sec ago"
    if s < 3600:      return f"{s // 60} min ago"
    if s < 86400:     return f"{s // 3600} hr ago"
    return f"{s // 86400} days ago"


# ── Curated demo events (same set the frontend seeds with) ───────────────
# Each entry's `seconds_ago` controls both ais_offline_since and time_ago,
# computed dynamically so the page always feels live.
_SEED = [
    {
        "id": "EVT-0097", "vessel": "FV-8872",
        "lat": -0.9132, "lng": -91.6234,
        "threat": "HIGH", "confidence": 96.2,
        "seconds_ago": 2_847,
        "hydrophone": "MBARI-01",
        "mpa_name": "Galápagos Marine Reserve", "mpa_distance_km": 1.4,
        "gemma_reasoning": (
            "Acoustic fingerprint places FV-8872 1.4 km inside Galápagos Marine "
            "Reserve boundary — active UNESCO World Heritage site violation. AIS "
            "silent for 47 minutes, zero position broadcasts. Speed profile "
            "dropped from 7.9 → 0.8 knots over a 12-minute window, consistent "
            "with gear deployment and station-keeping. GFW blacklist "
            "cross-reference: vessel heading corridor matches three prior "
            "incursion events in this zone (2022–2024). Confidence elevated by "
            "spectrogram match to known distant-water trawler class. IATTC "
            "notification and intercept recommended immediately."
        ),
        "vessel_window": "3 vessels transmitting AIS in detection window (GFW); 1 dark vessel acoustic-only (this target)",
        "cnn_analysis": (
            "Trawl gear harmonics confirmed at 94% classifier confidence. Winch "
            "cycling at 0.31 Hz consistent with longline haul sequence. "
            "Propulsion signature resolves to 4-blade variable-pitch propeller, "
            "35–55 m vessel class. SNR: 24.7 dB above ambient floor."
        ),
    },
    {
        "id": "EVT-0095", "vessel": "FV-3317",
        "lat": 9.0612, "lng": 119.7843,
        "threat": "HIGH", "confidence": 91.4,
        "seconds_ago": 4_734,
        "hydrophone": "MBARI-03",
        "mpa_name": "Tubbataha Reefs Natural Park", "mpa_distance_km": 2.1,
        "gemma_reasoning": (
            "FV-3317 operating 2.1 km from Tubbataha Reefs Natural Park — a "
            "no-take UNESCO World Heritage site patrolled by the Philippine "
            "Navy. AIS transponder dark for 1 hour 18 minutes, the longest gap "
            "in the current detection cycle. Nighttime operation with engine "
            "load signature indicating anchor-drift, atypical for transit. "
            "Prior GFW analysis flags this vessel corridor as a recurring "
            "poaching pathway used during ranger shift changes. Auxiliary "
            "winch harmonics consistent with gear handling. BFAR pre-alert "
            "recommended."
        ),
        "vessel_window": "9 vessels transmitting AIS in detection window (GFW AIS); target operating dark",
        "cnn_analysis": (
            "Strong low-frequency broadband signature. Engine load at 60–70% RPM "
            "consistent with station-keeping against current. Secondary harmonic "
            "cluster at 217 Hz indicates auxiliary deck machinery in operation — "
            "active gear handling inferred. Posterior probability: fishing "
            "vessel, 98.1%."
        ),
    },
    {
        "id": "EVT-0093", "vessel": "MV-5503",
        "lat": -5.1234, "lng": 130.3421,
        "threat": "HIGH", "confidence": 88.7,
        "seconds_ago": 9_240,
        "hydrophone": "MBARI-02",
        "mpa_name": "Banda Sea Marine National Park", "mpa_distance_km": 6.3,
        "gemma_reasoning": (
            "MV-5503 operating 6.3 km from Banda Sea Marine National Park with "
            "an AIS gap of 2 hours 34 minutes — the most extended dark window "
            "logged this session. Acoustic bearing fixes at T+0, T+42min, and "
            "T+104min place the vessel on a looping track consistent with a "
            "circular trawl pattern. The Banda Sea is a critical spawning "
            "ground for yellowfin tuna, bigeye, and skipjack; illegal bottom "
            "trawling carries severe ecosystem impact. A net deployment "
            "transient was detected at T+34min. Indonesia MCS notified."
        ),
        "vessel_window": "4 vessels transmitting AIS in 40 km radius (GFW); target is sole dark contact",
        "cnn_analysis": (
            "Extended acoustic contact maintained across three hydrophone "
            "bearings. Blade-rate stable at 2.4 Hz — continuous low-speed "
            "operation. Net deployment transient at T+34min: characteristic "
            "impulsive event followed by sustained drag noise. Active trawl "
            "classification: 78.3% confidence."
        ),
    },
    {
        "id": "EVT-0091", "vessel": "FV-6614",
        "lat": 1.6832, "lng": 7.4521,
        "threat": "HIGH", "confidence": 83.6,
        "seconds_ago": 1_380,
        "hydrophone": "MBARI-02",
        "mpa_name": "Gulf of Guinea Large Marine Ecosystem MPA", "mpa_distance_km": 4.9,
        "gemma_reasoning": (
            "FV-6614 detected 4.9 km from Gulf of Guinea MPA boundary — one of "
            "the world's most heavily over-fished regions. AIS went offline 23 "
            "minutes ago at a heading and speed inconsistent with equipment "
            "failure; the vessel was accelerating toward the restricted zone "
            "at the moment of shutoff. Steel-hull acoustic signature suggests "
            "a vessel 30–50 m LOA. Nighttime operation elevates IUU "
            "probability. Nigeria and Equatorial Guinea MCS have been notified."
        ),
        "vessel_window": "6 vessels transmitting AIS in detection window (GFW AIS); 1 dark contact (this vessel)",
        "cnn_analysis": (
            "Distinct propeller cavitation signature. Vessel speed estimated 3.1 "
            "knots from Doppler shift — consistent with active fishing speed. "
            "Broadband noise +18 dB above ambient floor confirms close-range "
            "contact (~7 km). Classifier confidence: 83.6% fishing vessel."
        ),
    },
    {
        "id": "EVT-0088", "vessel": "FV-2289",
        "lat": -12.4521, "lng": 40.9832,
        "threat": "MEDIUM", "confidence": 74.1,
        "seconds_ago": 960,
        "hydrophone": "MBARI-02",
        "mpa_name": "Quirimbas National Park", "mpa_distance_km": 18.4,
        "gemma_reasoning": (
            "FV-2289 detected 18.4 km from Quirimbas National Park in the "
            "Mozambique Channel. AIS offline for 16 minutes — shorter gap, but "
            "trajectory analysis shows a direct approach bearing toward the "
            "restricted zone; speed has not decreased to suggest transit "
            "disinterest. Regional fisher density is high; confidence is "
            "moderate pending extended observation. Tanzania Maritime Authority "
            "notified for secondary verification."
        ),
        "vessel_window": "11 vessels transmitting AIS in detection window (GFW AIS)",
        "cnn_analysis": (
            "Moderate acoustic contact. Blade-rate at 1.9 Hz with intermittent "
            "dropouts attributable to thermocline interference at 120 m depth "
            "layer. Propulsion noise profile consistent with wooden-hulled "
            "vessel 15–25 m class. SNR: 11.4 dB. Classification: probable "
            "fishing vessel."
        ),
    },
    {
        "id": "EVT-0085", "vessel": "MV-7743",
        "lat": 52.8923, "lng": -169.2341,
        "threat": "MEDIUM", "confidence": 62.3,
        "seconds_ago": 480,
        "hydrophone": "ORCASOUND-01",
        "mpa_name": "Aleutian Islands National Wildlife Refuge", "mpa_distance_km": 38.2,
        "gemma_reasoning": (
            "MV-7743 contact 38.2 km from Aleutian Islands National Wildlife "
            "Refuge boundary in the Bering Sea. AIS gap of 8 minutes is within "
            "the equipment-glitch threshold, but acoustic bearing fixes "
            "disagree with the last logged AIS trajectory by 23° — vessel "
            "appears to have altered course while dark. Bering Sea pollock and "
            "crab stocks are under severe fishing pressure; unauthorized refuge "
            "access carries federal penalties. NOAA OLE flagged for monitoring."
        ),
        "vessel_window": "5 vessels transmitting AIS in detection window (GFW AIS)",
        "cnn_analysis": (
            "Weak-to-moderate contact. Propeller signature partially masked by "
            "Sea State 4 ambient noise (high wind, 28-knot surface). Estimated "
            "vessel speed 6.8 knots from blade-rate. Classifier: probable "
            "fishing/cargo vessel, 62.3% confidence."
        ),
    },
    {
        "id": "EVT-0082", "vessel": "MV-1192",
        "lat": 12.4532, "lng": 53.8211,
        "threat": "LOW", "confidence": 51.8,
        "seconds_ago": 11_520,
        "hydrophone": "MBARI-02",
        "mpa_name": "Socotra Archipelago UNESCO Biosphere Reserve", "mpa_distance_km": 89.1,
        "gemma_reasoning": (
            "Long-duration dark contact maintained for 3 hours 12 minutes in "
            "the Gulf of Aden. MPA distance of 89.1 km and shallow acoustic "
            "confidence prevent immediate escalation. Vessel bearing aligns "
            "with a dhow transit corridor commonly used by small commercial "
            "fishing vessels in this region. Socotra Archipelago boundaries "
            "are not immediately threatened. IOTC logged as low-priority "
            "observation."
        ),
        "vessel_window": "14 vessels transmitting AIS in detection window (GFW AIS)",
        "cnn_analysis": (
            "Intermittent contact; signal fades consistent with multipath "
            "interference in shallow Gulf of Aden environment. Acoustic "
            "signature ambiguous — possible small commercial vessel or "
            "motorized dhow. Classifier confidence 51.8%, insufficient for "
            "firm vessel-class determination."
        ),
    },
]


def _materialize(seed: dict) -> VesselEvent:
    """Compute live timestamps off `seconds_ago` so the feed never goes stale."""
    s = seed["seconds_ago"]
    return VesselEvent(
        id=seed["id"], vessel=seed["vessel"],
        lat=seed["lat"], lng=seed["lng"],
        threat=seed["threat"], confidence=seed["confidence"],
        ais_offline_since=_ago_iso(s),
        hydrophone=seed["hydrophone"],
        time_ago=_humanize_seconds(s),
        mpa_name=seed["mpa_name"],
        mpa_distance_km=seed["mpa_distance_km"],
        gemma_reasoning=seed["gemma_reasoning"],
        vessel_window=seed["vessel_window"],
        cnn_analysis=seed["cnn_analysis"],
    )


@router.get("/feed", response_model=list[VesselEvent])
async def feed(
    threat: Literal["HIGH", "MEDIUM", "LOW", "ALL"] = Query(default="ALL"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Return the dashboard feed in `VesselEvent` shape.

    Frontend `useEvents.ts` polls this every 10 s. The seeds give a
    cohesive narrative the dashboard always has data to render; once
    we wire `_from_detection` (TODO) to the SQLite event store, this
    endpoint will fall back to seeds only when the store is empty.
    """
    items = [_materialize(s) for s in _SEED]
    if threat != "ALL":
        items = [e for e in items if e.threat == threat]
    return items[:limit]


# ── For when we wire to the live event store ────────────────────────────
# Sketch — left here to make the future work obvious. The DetectionEvent
# domain object in our store has location/timestamp/threat_level/confidence.
# The remaining VesselEvent fields (gemma_reasoning, vessel_window,
# cnn_analysis) need to be populated either by the decision engine when
# it logs the detection, or by a small enrichment pass.
#
# def _from_detection(event: DetectionEvent) -> VesselEvent:
#     ...
