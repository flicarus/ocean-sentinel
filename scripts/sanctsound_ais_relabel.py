"""Re-label existing SanctSound chunks using AIS cross-validation.

The SanctSound dataset blanket-labels every chunk as 'ambient'. Our OC01 study
showed at least one chunk was actually a confirmed cargo at <=10 km — and the
broader pattern (Cape Flattery is a shipping corridor) suggests this isn't an
isolated case. This script systematically queries GFW for vessel presence at
each chunk's timestamp and emits new labels:

  - 'ship' — vessel within 10 km in the chunk's UTC hour (close-range)
  - 'ship_distant' — vessel within 50 km, not 10 (acoustically detectable but
                     not visually obvious in the spectrogram)
  - 'ambient' — no vessel in 50 km AND no vessel in prior hour (no acoustic
                tail); confidence label
  - SKIPPED — anything else (ambiguous; we don't trust either ship or ambient)

Existing rows in v7.jsonl are unchanged; new rows go to a separate manifest
that the v7 trainer picks up automatically.

Output: data/training/sanctsound_corrected.jsonl

Usage:
    PYTHONPATH=src venv/bin/python scripts/sanctsound_ais_relabel.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import httpx
import structlog

sys.path.insert(0, "src")
from ocean_sentinel.config import Settings

log = structlog.get_logger()

# Lat/lon for each SanctSound deployment we touch. From NOAA public metadata.
# OC01 confirmed via earlier scripts; SB01 from NOAA Stellwagen Bank NMS docs.
SITE_LOCATIONS: dict[str, tuple[float, float]] = {
    "oc01": (48.400, -124.700),    # Olympic Coast 01, Cape Flattery WA
    "sb01": (42.460, -70.500),     # Stellwagen Bank 01, MA
    # Add more sites here as we relabel them.
}

INPUT_JSONL = Path("data/training/gemma_labels.v7.jsonl")
OUTPUT_JSONL = Path("data/training/sanctsound_corrected.jsonl")

GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"

# Filename pattern: SanctSound_OC01_01_671399974_20190308T190000Z.flac
FILENAME_TS_RE = re.compile(r"_(\d{8}T\d{6})Z\.flac")


def _parse_capture_start(source_file: str) -> datetime | None:
    m = FILENAME_TS_RE.search(source_file)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


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


async def fetch_hourly(
    client: httpx.AsyncClient, token: str,
    lat: float, lon: float, day: datetime, radius_km: float,
    max_retries: int = 6,
) -> dict[int, list[dict]]:
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
    delay = 5.0
    for attempt in range(max_retries):
        resp = await client.post(
            url, json=_bbox(lat, lon, radius_km),
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 429:
            log.warning("gfw_rate_limited", attempt=attempt, wait_s=delay)
            await asyncio.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        out: dict[int, list[dict]] = {h: [] for h in range(24)}
        for entry in resp.json().get("entries", []):
            if isinstance(entry, dict):
                for k, v in entry.items():
                    if k.startswith("public-") and isinstance(v, list):
                        for r in v:
                            ts = r.get("entryTimestamp") or r.get("date")
                            if not ts:
                                continue
                            try:
                                h = datetime.fromisoformat(
                                    ts.replace("Z", "+00:00")
                                ).hour
                            except Exception:
                                continue
                            out[h].append(r)
        return out
    raise RuntimeError("rate limit exhausted")


async def main() -> None:
    settings = Settings()
    if not settings.gfw_api_token:
        sys.exit("OS_GFW_API_TOKEN missing")

    print(f"Reading {INPUT_JSONL} ...")
    rows: list[dict] = []
    with INPUT_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("provenance", {}).get("source_id") == "sanctsound":
                rows.append(r)
    print(f"  {len(rows)} sanctsound rows")

    # Group by (site, day) so we cache GFW responses per day instead of per chunk.
    grouped: dict[tuple[str, str], list[dict]] = {}
    skipped_no_ts = 0
    for r in rows:
        source_file = r.get("provenance", {}).get("source_file", "")
        site = source_file.split("/")[0] if "/" in source_file else "unknown"
        if site not in SITE_LOCATIONS:
            continue
        capture_start = _parse_capture_start(source_file)
        if capture_start is None:
            skipped_no_ts += 1
            continue
        # Chunk index from event_id suffix: e.g. "..._0", "..._1"
        chunk_idx_match = re.search(r"_(\d+)$", r["event_id"])
        if not chunk_idx_match:
            skipped_no_ts += 1
            continue
        chunk_idx = int(chunk_idx_match.group(1))
        # Each chunk is duration_s long
        duration_s = float(r.get("audio", {}).get("duration_s") or 5.0)
        chunk_ts = capture_start + timedelta(seconds=chunk_idx * duration_s)
        day_key = chunk_ts.strftime("%Y-%m-%d")
        r["_chunk_ts"] = chunk_ts.isoformat()
        r["_chunk_hour"] = chunk_ts.hour
        r["_site"] = site
        grouped.setdefault((site, day_key), []).append(r)

    print(f"  {sum(len(v) for v in grouped.values())} rows mappable to "
          f"{len(grouped)} (site, day) groups; skipped {skipped_no_ts}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    counters = {"ship": 0, "ship_distant": 0, "ambient": 0, "skipped": 0}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, ((site, day_str), site_rows) in enumerate(grouped.items()):
            lat, lon = SITE_LOCATIONS[site]
            day = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            print(f"\n[{i+1}/{len(grouped)}] {site} {day_str} — {len(site_rows)} chunks")

            try:
                hr10 = await fetch_hourly(client, settings.gfw_api_token, lat, lon, day, 10.0)
                # Polite spacing between calls — keeps us under per-second cap.
                await asyncio.sleep(1.0)
                hr50 = await fetch_hourly(client, settings.gfw_api_token, lat, lon, day, 50.0)
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"  GFW failed: {e} — skipping group")
                counters["skipped"] += len(site_rows)
                continue

            n_close_hours = sum(1 for v in hr10.values() if v)
            n_wide_hours = sum(1 for v in hr50.values() if v)
            print(f"  vessels in 10km: {n_close_hours}/24 hours, "
                  f"in 50km: {n_wide_hours}/24")

            for r in site_rows:
                h = r["_chunk_hour"]
                close = bool(hr10.get(h))
                wide = bool(hr50.get(h))
                prior_wide = bool(hr50.get((h - 1) % 24))

                if close:
                    new_label = "ship"
                    distance_bucket = "close"
                elif wide:
                    new_label = "ship_distant"
                    distance_bucket = "medium"
                elif prior_wide:
                    counters["skipped"] += 1
                    continue  # ambiguous acoustic tail
                else:
                    new_label = "ambient"
                    distance_bucket = "none"

                # Record the new corrected label as a fresh row so the trainer
                # picks it up via gemma_labels.v7.jsonl PLUS this file. We
                # keep the original spectrogram_path — same audio, new label.
                new_row = dict(r)
                new_row["original_label"] = r.get("label")
                new_row["label"] = "ship" if new_label == "ship" else "not_ship"
                new_row["sanctsound_ais_label"] = new_label
                new_row["distance_bucket"] = distance_bucket
                new_row.setdefault("provenance", {})["source_id"] = (
                    f"sanctsound-corrected-{site}"
                )
                # Strip our internal helper fields.
                new_row.pop("_chunk_ts", None)
                new_row.pop("_chunk_hour", None)
                new_row.pop("_site", None)

                with OUTPUT_JSONL.open("a") as f:
                    f.write(json.dumps(new_row) + "\n")

                counters[new_label] = counters.get(new_label, 0) + 1

    print()
    print("=" * 60)
    print("Counters:", counters)
    print(f"  total emitted: {sum(v for k, v in counters.items() if k != 'skipped')}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
