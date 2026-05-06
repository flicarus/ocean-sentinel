"""Generate AIS-correlated training pairs — v2 with audio-time-aware AIS query.

Critical fix vs v1 (`bootstrap_ais_correlated.py`): the original script took a
user-supplied --date as the AIS query window, but Orcasound's adapter ignores
the requested date and serves whatever HLS stream was newest at the time the
script was running. The audio capture timestamp could be years off from the
AIS query window, silently producing wrong labels.

This v2 inverts the flow:
  1. Fetch the Orcasound segment first.
  2. Use audio.time_window.start as the AIS query date.
  3. Label based on AIS that's actually contemporaneous with the audio.

Each Orcasound node currently exposes one viable stream from a specific date
(north_sjc 2023-06-18, mast_center 2023-07-09, sunset_bay 2022-06-08 etc.).
We accept whatever the node gives us, derive the date from the stream, and
build training pairs from real audio↔AIS pairs.

Usage:
    PYTHONPATH=src venv/bin/python3 scripts/bootstrap_orcasound_v2.py \\
        --hydrophone north-sjc --step 300 --radius-km 10
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import httpx
import numpy as np
import structlog

sys.path.insert(0, "src")

import json

from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.orcasound import OrcasoundAdapter
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import (
    AcousticFeatures, GeoPoint, NearbyVessel, TimeWindow,
)
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer

# Per-node JSONL files. Avoids concurrent-append races when we run multiple
# node bootstraps in parallel. Merge into v7 corpus afterwards.
OUTPUT_DIR = Path("data/training/v7_bulk")


def _log_pair(entry: dict, hydrophone_id: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"orcasound_{hydrophone_id}.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(entry) + "\n")


GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"


def _bbox(lat: float, lon: float, radius_km: float) -> dict:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * cos(radians(lat)))
    return {
        "geojson": {
            "type": "Polygon",
            "coordinates": [[
                [lon - dlon, lat - dlat],
                [lon + dlon, lat - dlat],
                [lon + dlon, lat + dlat],
                [lon - dlon, lat + dlat],
                [lon - dlon, lat - dlat],
            ]],
        }
    }


async def fetch_hourly_vessels(
    token: str,
    location: GeoPoint,
    day: datetime,
    radius_km: float,
    max_retries: int = 6,
) -> dict[int, list[NearbyVessel]]:
    """Returns map of UTC hour -> list of vessels present that hour.

    Retries with exponential backoff on 429 rate-limit responses, so a
    parallel bootstrap launch that bursts through GFW's per-second cap
    self-recovers instead of failing the run.
    """
    d0 = day.strftime("%Y-%m-%d")
    d1 = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"{GFW_BASE}/4wings/report"
        f"?datasets[0]=public-global-presence:latest"
        f"&date-range={d0},{d1}"
        f"&temporal-resolution=HOURLY"
        f"&spatial-resolution=HIGH"
        f"&group-by=VESSEL_ID"
        f"&format=JSON"
    )

    rows: list[dict] = []
    delay = 5.0
    last_err: Exception | None = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    url, json=_bbox(location.lat, location.lon, radius_km),
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code == 429:
                    log.warning("gfw_rate_limited", attempt=attempt, wait_s=delay)
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                for entry in resp.json().get("entries", []):
                    if isinstance(entry, dict):
                        for k, v in entry.items():
                            if k.startswith("public-") and isinstance(v, list):
                                rows.extend(v)
                break
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code == 429:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
        else:
            raise RuntimeError(
                f"GFW exhausted retries on rate limit; last={last_err}"
            )

    out: dict[int, list[NearbyVessel]] = {h: [] for h in range(24)}
    for r in rows:
        ts = r.get("entryTimestamp") or r.get("date")
        if not ts:
            continue
        try:
            h = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
        except Exception:
            continue
        try:
            length_m = float(r.get("vesselLength") or 0) or None
        except (TypeError, ValueError):
            length_m = None
        v = NearbyVessel(
            vessel_id=r.get("vesselId") or r.get("mmsi") or "",
            vessel_name=r.get("shipName"),
            vessel_class=(r.get("vesselType") or r.get("geartype") or "").lower() or None,
            flag_state=r.get("flag"),
            position=location,
            distance_km=0.0,
            length_m=length_m,
            present_start=None,
            present_end=None,
        )
        out[h].append(v)
    return out

log = structlog.get_logger()

HYDROPHONE_NODES: dict[str, str] = {
    "bush-point":     "rpi_bush_point",
    "sunset-bay":     "rpi_sunset_bay",
    "orcasound-lab":  "rpi_orcasound_lab",
    "north-sjc":      "rpi_north_sjc",
    "point-robinson": "rpi_point_robinson",
    "port-townsend":  "rpi_port_townsend",
    "andrews-bay":    "rpi_andrews_bay",
    "mast-center":    "rpi_mast_center",
}


def _severity_rank(vessel: NearbyVessel) -> int:
    vc = (vessel.vessel_class or "").lower()
    if vc in ("cargo", "tanker"):
        return 4
    if vessel.length_m and vessel.length_m >= 80:
        return 3
    if vc == "fishing":
        return 2
    if vessel.length_m and vessel.length_m >= 30:
        return 2
    if vc == "passenger":
        return 2
    return 0


def _label_from_vessels(vessels: list[NearbyVessel]) -> tuple[str, str] | None:
    """Return (threat_level, vessel_type) or None to skip ambiguous windows."""
    if not vessels:
        return None
    worst = max(vessels, key=_severity_rank)
    vc = (worst.vessel_class or "").lower()
    if vc in ("cargo", "tanker"):
        return ("HIGH", vc)
    if worst.length_m and worst.length_m >= 80:
        return ("HIGH", vc or "unknown")
    if vc == "fishing":
        return ("MEDIUM", "fishing_vessel")
    if worst.length_m and 30 <= worst.length_m < 80:
        return ("MEDIUM", vc or "unknown")
    if vc == "passenger":
        return ("MEDIUM", "passenger_vessel")
    return None


def _vessels_at_time(vessels: list[NearbyVessel], window: TimeWindow) -> list[NearbyVessel]:
    out = []
    for v in vessels:
        if v.present_start is None or v.present_end is None:
            out.append(v)
            continue
        if v.present_end >= window.start and v.present_start <= window.end:
            out.append(v)
    return out


async def label_for_window(
    gfw: GFWAdapter,
    location: GeoPoint,
    window: TimeWindow,
    radius_km: float,
) -> tuple[str, str] | None:
    all_day = await gfw.get_vessels_in_radius(location, radius_km, window)
    current = _vessels_at_time(all_day, window)
    if current:
        return _label_from_vessels(current)
    prior_window = TimeWindow(
        start=window.start - timedelta(hours=1),
        end=window.start,
    )
    all_prior = await gfw.get_vessels_in_radius(location, radius_km, prior_window)
    prior = _vessels_at_time(all_prior, prior_window)
    if prior:
        return None
    return ("NONE", "none")


def _vessel_type_label(vessel_type: str) -> str:
    mapping = {
        "cargo":            "cargo_ship",
        "tanker":           "tanker",
        "fishing":          "fishing_vessel",
        "fishing_vessel":   "fishing_vessel",
        "passenger":        "passenger_vessel",
        "passenger_vessel": "passenger_vessel",
    }
    return mapping.get(vessel_type.lower(), vessel_type)


async def main(hydrophone_id: str, radius_km: float, step: int, max_windows: int | None) -> None:
    node_name = HYDROPHONE_NODES[hydrophone_id]
    settings = Settings()

    orca = OrcasoundAdapter(node_name=node_name, settings=settings)
    analyzer = AudioAnalyzer(settings)

    spec_dir = Path("data/spectrograms")
    spec_dir.mkdir(parents=True, exist_ok=True)

    # Probe the stream once to know how much audio is actually available
    # and what date it covers. _find_viable_stream picks ONE stream per node;
    # all segments share that stream's start timestamp.
    await orca._ensure_stream()
    stream_start = datetime.fromtimestamp(orca._stream_ts, tz=timezone.utc)
    stream_total_seconds = len(orca._segments) * 10
    derived_date = stream_start.strftime("%Y-%m-%d")

    location = orca.location

    # Fetch HOURLY vessel presence at three radii: close (10km), medium (30km),
    # far (50km). Each window gets a per-hour distance bucket so v7 can learn
    # close/medium/far/none without us throwing away ambiguous windows.
    print(f"Pre-fetching hourly AIS for {derived_date} around {hydrophone_id} at 10/30/50 km...")
    day0 = stream_start.replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_10 = await fetch_hourly_vessels(settings.gfw_api_token, location, day0, 10.0)
    hourly_30 = await fetch_hourly_vessels(settings.gfw_api_token, location, day0, 30.0)
    hourly_50 = await fetch_hourly_vessels(settings.gfw_api_token, location, day0, 50.0)
    print(f"  hours with vessels in 10km: {sum(1 for v in hourly_10.values() if v):>2d}/24")
    print(f"  hours with vessels in 30km: {sum(1 for v in hourly_30.values() if v):>2d}/24")
    print(f"  hours with vessels in 50km: {sum(1 for v in hourly_50.values() if v):>2d}/24")

    counters = {"close": 0, "medium": 0, "far": 0, "none": 0, "FAILED": 0}

    print(f"Hydrophone : {hydrophone_id} ({node_name})")
    print(f"Stream date: {stream_start.isoformat()} ({stream_total_seconds // 3600}h available)")
    print(f"Radius     : {radius_km} km   step: {step}s")
    if max_windows:
        print(f"Cap        : {max_windows} windows max")
    print()

    # Iterate over offsets within the stream (not over a virtual 24h day).
    offsets = list(range(0, stream_total_seconds, step))
    if max_windows:
        offsets = offsets[:max_windows]

    total = len(offsets)
    for i, offset in enumerate(offsets):
        try:
            # 1. Fetch audio first — this gives us the real capture_start.
            segment = await orca.fetch_at_offset(
                stream_start, offset_seconds=offset, duration_seconds=60,
            )
            window = segment.time_window
            event_id = (
                f"ais_{hydrophone_id}_"
                f"{window.start.strftime('%Y-%m-%d')}_"
                f"{window.start.strftime('%H%M%S')}"
            )

            if i % 20 == 0:
                print(
                    f"  [{i}/{total}] {window.start.strftime('%H:%M:%S')} — "
                    f"close={counters['close']} med={counters['medium']} "
                    f"far={counters['far']} none={counters['none']} "
                    f"FAIL={counters['FAILED']}"
                )

            # 2. Distance bucket from cached hourly AIS at 10/30/50 km radii.
            window_hour = window.start.hour
            in_10 = hourly_10.get(window_hour, [])
            in_30 = hourly_30.get(window_hour, [])
            in_50 = hourly_50.get(window_hour, [])

            if in_10:
                distance_bucket = "close"
                vessels_for_type = in_10
            elif in_30:
                distance_bucket = "medium"
                vessels_for_type = in_30
            elif in_50:
                distance_bucket = "far"
                vessels_for_type = in_50
            else:
                # Acoustic-tail check: vessels in prior hour can still be heard.
                prior_in_50 = hourly_50.get((window_hour - 1) % 24, [])
                if prior_in_50:
                    counters["FAILED"] += 1  # tagging as ambiguous
                    continue
                distance_bucket = "none"
                vessels_for_type = []

            # Vessel type from worst-case vessel in the bucket
            if vessels_for_type:
                worst = max(vessels_for_type, key=_severity_rank)
                vessel_type = (worst.vessel_class or "unknown").lower()
            else:
                vessel_type = "none"

            # Binary label: anything in 10km is "ship", everything else "ambient"
            # (medium/far still keep their distance label as auxiliary target)
            binary_label = "ship" if distance_bucket == "close" else "not_ship"
            threat_level = (
                "HIGH" if distance_bucket == "close" and vessel_type in ("cargo", "tanker") else
                "MEDIUM" if distance_bucket == "close" else
                "LOW" if distance_bucket in ("medium", "far") else
                "NONE"
            )

            # 3. Spectrogram + features.
            analyzed, features = analyzer.analyze(segment)
            spec_path = spec_dir / f"{event_id}.npy"
            np.save(spec_path, analyzed.spectrogram)

            # 4. Log training pair to v7_bulk file.
            entry = {
                "event_id": event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "spectrogram_path": str(spec_path),
                "audio_capture_start": window.start.isoformat(),
                "audio_capture_end": window.end.isoformat(),
                "features": {
                    "engine_band_ratio": features["engine_band_ratio"],
                    "peak_frequency_hz": features["peak_frequency_hz"],
                    "spectral_flatness": features["spectral_flatness"],
                    "rms_energy": features["rms_energy"],
                    "engine_band_energy_db": features["engine_band_energy_db"],
                },
                "context_text": (
                    f"{node_name} | {window.start.isoformat()} | "
                    f"vessel={vessel_type} | radius={radius_km}km"
                ),
                "gemma_verdict": {
                    "threat_level": threat_level,
                    "vessel_type": _vessel_type_label(vessel_type),
                    "is_bootstrap": True,
                },
                "provenance": {
                    "source_id": f"ais-correlated-{hydrophone_id}",
                    "hydrophone": hydrophone_id,
                    "stream_ts": orca._stream_ts,
                    "stream_offset_seconds": offset,
                },
                "label": binary_label,
                "distance_bucket": distance_bucket,
                "ground_truth_label": threat_level,
            }
            _log_pair(entry, hydrophone_id)

            counters[distance_bucket] = counters.get(distance_bucket, 0) + 1

        except Exception as e:
            counters["FAILED"] += 1
            log.warning("window_failed", offset=offset, error=str(e))

    print()
    print("Final counters:", counters)
    await orca.close()


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hydrophone", required=True, choices=HYDROPHONE_NODES.keys())
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=300, help="seconds between windows")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap to N windows for smoke testing")
    args = ap.parse_args()
    asyncio.run(main(
        hydrophone_id=args.hydrophone,
        radius_km=args.radius_km,
        step=args.step,
        max_windows=args.max_windows,
    ))


if __name__ == "__main__":
    cli()
