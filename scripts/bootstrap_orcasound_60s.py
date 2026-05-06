"""Bootstrap Orcasound at 60s windows for v8 (longer-context training).

Differences vs bootstrap_orcasound_v2.py:
  - Fetches 6 consecutive 10s HLS segments per chunk and concats them into
    a single 60s audio clip before computing the spectrogram. The resulting
    spec is shape (128, ~1876), 6x more frames than the v2 bootstrap.
  - Output JSONL is `data/training/v8_bulk/orcasound_<node>.jsonl`, kept
    separate so v8 training can pick it up without colliding with the
    older 10s data.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import httpx
import librosa
import numpy as np
import structlog

sys.path.insert(0, "src")
from ocean_sentinel.adapters.orcasound import OrcasoundAdapter
from ocean_sentinel.adapters.training_logger import JSONLTrainingLogger
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import (
    AcousticFeatures, AudioSegment, GeoPoint, NearbyVessel, TimeWindow,
)
from ocean_sentinel.services.audio_analyzer import AudioAnalyzer

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

GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"

OUTPUT_DIR = Path("data/training/v8_bulk")
SPEC_DIR = Path("data/spectrograms_60s")


def _bbox(lat: float, lon: float, radius_km: float) -> dict:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * cos(radians(lat)))
    return {"geojson": {"type": "Polygon", "coordinates": [[
        [lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
        [lon - dlon, lat - dlat],
    ]]}}


async def fetch_hourly_vessels(
    token: str, location: GeoPoint, day: datetime, radius_km: float,
    max_retries: int = 6,
) -> dict[int, list[NearbyVessel]]:
    d0 = day.strftime("%Y-%m-%d")
    d1 = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (f"{GFW_BASE}/4wings/report"
           f"?datasets[0]=public-global-presence:latest"
           f"&date-range={d0},{d1}&temporal-resolution=HOURLY"
           f"&spatial-resolution=HIGH&group-by=VESSEL_ID&format=JSON")
    delay = 5.0
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(max_retries):
            resp = await client.post(
                url, json=_bbox(location.lat, location.lon, radius_km),
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 429:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            rows: list[dict] = []
            for entry in resp.json().get("entries", []):
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        if k.startswith("public-") and isinstance(v, list):
                            rows.extend(v)
            break
        else:
            raise RuntimeError("rate limit exhausted")

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
            position=location, distance_km=0.0, length_m=length_m,
            present_start=None, present_end=None,
        )
        out[h].append(v)
    return out


def _severity_rank(v: NearbyVessel) -> int:
    vc = (v.vessel_class or "").lower()
    if vc in ("cargo", "tanker"): return 4
    if v.length_m and v.length_m >= 80: return 3
    if vc == "fishing": return 2
    if v.length_m and v.length_m >= 30: return 2
    if vc == "passenger": return 2
    return 0


async def fetch_60s_chunk(orca: OrcasoundAdapter, offset_seconds: int) -> AudioSegment:
    """Concat 6 consecutive 10s HLS segments → 60s AudioSegment."""
    pieces: list[AudioSegment] = []
    for i in range(6):
        seg = await orca.fetch_at_offset(
            datetime.now(timezone.utc),
            offset_seconds=offset_seconds + i * 10,
            duration_seconds=10,
        )
        pieces.append(seg)
    samples = np.concatenate([p.samples for p in pieces])
    first = pieces[0]
    last = pieces[-1]
    return AudioSegment(
        source_file=first.source_file,
        location=first.location,
        time_window=TimeWindow(start=first.time_window.start,
                               end=last.time_window.end),
        sample_rate=first.sample_rate,
        samples=samples,
        source_id=first.source_id,
    )


async def main(hydrophone_id: str, radius_km: float, step: int,
               max_chunks: int | None) -> None:
    node_name = HYDROPHONE_NODES[hydrophone_id]
    settings = Settings()
    if not settings.gfw_api_token:
        sys.exit("OS_GFW_API_TOKEN missing")

    orca = OrcasoundAdapter(node_name=node_name, settings=settings)
    analyzer = AudioAnalyzer(settings)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUTPUT_DIR / f"orcasound_{hydrophone_id}.jsonl"

    await orca._ensure_stream()
    stream_start = datetime.fromtimestamp(orca._stream_ts, tz=timezone.utc)
    stream_total_seconds = len(orca._segments) * 10
    location = orca.location
    derived_date = stream_start.strftime("%Y-%m-%d")
    print(f"Hydrophone : {hydrophone_id}  stream_date={derived_date}  "
          f"available={stream_total_seconds//3600}h")

    day0 = stream_start.replace(hour=0, minute=0, second=0, microsecond=0)
    print("Fetching hourly AIS at 10/30/50 km ...")
    h10 = await fetch_hourly_vessels(settings.gfw_api_token, location, day0, 10.0)
    h30 = await fetch_hourly_vessels(settings.gfw_api_token, location, day0, 30.0)
    h50 = await fetch_hourly_vessels(settings.gfw_api_token, location, day0, 50.0)

    counters = {"close": 0, "medium": 0, "far": 0, "none": 0, "FAILED": 0}
    offsets = list(range(0, max(stream_total_seconds - 60, 0), step))
    if max_chunks:
        offsets = offsets[:max_chunks]

    for i, offset in enumerate(offsets):
        try:
            seg = await fetch_60s_chunk(orca, offset)
            analyzed, features = analyzer.analyze(seg)
            window = seg.time_window

            h = window.start.hour
            in_10 = h10.get(h, [])
            in_30 = h30.get(h, [])
            in_50 = h50.get(h, [])
            if in_10:
                bucket = "close"
                worst = max(in_10, key=_severity_rank)
                vessel_type = (worst.vessel_class or "unknown").lower()
            elif in_30:
                bucket = "medium"
                worst = max(in_30, key=_severity_rank)
                vessel_type = (worst.vessel_class or "unknown").lower()
            elif in_50:
                bucket = "far"
                worst = max(in_50, key=_severity_rank)
                vessel_type = (worst.vessel_class or "unknown").lower()
            else:
                prior_50 = h50.get((h - 1) % 24, [])
                if prior_50:
                    counters["FAILED"] += 1
                    continue
                bucket = "none"
                vessel_type = "none"

            event_id = (f"ais60s_{hydrophone_id}_"
                        f"{window.start.strftime('%Y-%m-%d_%H%M%S')}")
            spec_path = SPEC_DIR / f"{event_id}.npy"
            np.save(spec_path, analyzed.spectrogram.astype(np.float32))

            entry = {
                "event_id": event_id,
                "spectrogram_path": str(spec_path),
                "audio_capture_start": window.start.isoformat(),
                "audio_capture_end": window.end.isoformat(),
                "duration_s": 60.0,
                "features": {
                    "engine_band_ratio": features["engine_band_ratio"],
                    "peak_frequency_hz": features["peak_frequency_hz"],
                    "spectral_flatness": features["spectral_flatness"],
                    "rms_energy": features["rms_energy"],
                    "engine_band_energy_db": features["engine_band_energy_db"],
                },
                "context_text": (f"{node_name} | {window.start.isoformat()} | "
                                 f"vessel={vessel_type} | bucket={bucket}"),
                "gemma_verdict": {"threat_level": (
                    "HIGH" if bucket == "close" and vessel_type in ("cargo", "tanker") else
                    "MEDIUM" if bucket == "close" else
                    "LOW" if bucket in ("medium", "far") else "NONE"
                ), "vessel_type": vessel_type, "is_bootstrap": True},
                "provenance": {
                    "source_id": f"ais-correlated-60s-{hydrophone_id}",
                    "hydrophone": hydrophone_id,
                    "stream_ts": orca._stream_ts,
                    "stream_offset_seconds": offset,
                },
                "label": "ship" if bucket == "close" else "not_ship",
                "distance_bucket": bucket,
            }
            with out_jsonl.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            counters[bucket] += 1

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(offsets)}]  "
                      f"close={counters['close']} med={counters['medium']} "
                      f"far={counters['far']} none={counters['none']} "
                      f"fail={counters['FAILED']}")
        except Exception as e:
            counters["FAILED"] += 1
            log.warning("chunk_failed", offset=offset, error=str(e))

    print(f"\nFinal counters: {counters}")
    await orca.close()


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hydrophone", required=True, choices=HYDROPHONE_NODES.keys())
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=120,
                    help="seconds between 60s chunks (default 120 = 50%% overlap)")
    ap.add_argument("--max-chunks", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main(args.hydrophone, args.radius_km, args.step, args.max_chunks))


if __name__ == "__main__":
    cli()
