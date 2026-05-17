"""Phase 2: per-chunk AIS cross-reference for the OC01 corpus (4 recordings).

For each of the 800 oc01 chunks, derive its (file_start_utc, chunk_idx)
and AIS-vessel-presence at 50/10/5 km radii at HOURLY granularity.
Cross-tab against current_label (the post-v6-relabel label).

This is the GROUND-TRUTH check the case study needs:
  - Does the "ambient control" recording (03-09T05:59) actually have a
    vessel broadcasting AIS within 10 km during those hours?
  - Do the three "ship-time" recordings have continuous vessel presence
    or only in specific hours?
  - Quantify the bidirectional label noise IN AIS TERMS, not just acoustic.

Output: data/audit/oc01_ais_per_chunk.json
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import httpx

from ocean_sentinel.config import Settings


OC01_LAT, OC01_LON = 48.400, -124.700
GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"
RADII_KM = (50, 10, 5)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/training/sanctsound_corrected.jsonl"
OUT_JSON = ROOT / "data/audit/oc01_ais_per_chunk.json"


def bbox(radius_km: float) -> dict:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * cos(radians(OC01_LAT)))
    return {
        "geojson": {
            "type": "Polygon",
            "coordinates": [[
                [OC01_LON - dlon, OC01_LAT - dlat],
                [OC01_LON + dlon, OC01_LAT - dlat],
                [OC01_LON + dlon, OC01_LAT + dlat],
                [OC01_LON - dlon, OC01_LAT + dlat],
                [OC01_LON - dlon, OC01_LAT - dlat],
            ]],
        }
    }


async def query_hourly(
    client: httpx.AsyncClient, token: str, day: datetime, radius_km: float,
) -> list[dict]:
    """Return raw GFW vessel-presence rows for the day at hourly resolution."""
    day_str = day.strftime("%Y-%m-%d")
    next_day_str = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"{GFW_BASE}/4wings/report"
        f"?datasets[0]=public-global-presence:latest"
        f"&date-range={day_str},{next_day_str}"
        f"&temporal-resolution=HOURLY"
        f"&spatial-resolution=HIGH"
        f"&group-by=VESSEL_ID"
        f"&format=JSON"
    )
    resp = await client.post(
        url, json=bbox(radius_km),
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    data = resp.json()
    rows: list[dict] = []
    for entry in data.get("entries", []):
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.startswith("public-") and isinstance(v, list):
                    rows.extend(v)
    return rows


def parse_recording_start(source_file: str) -> datetime | None:
    """oc01/SanctSound_OC01_01_671399974_20190308T190000Z.flac → datetime."""
    name = source_file.split("/")[-1]
    parts = name.split("_")
    ts_part = next((p.split(".")[0] for p in parts if p.endswith("Z") or p.endswith("Z.flac")), None)
    if not ts_part:
        return None
    try:
        return datetime.strptime(ts_part.replace(".flac", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def chunk_idx_from_event(event_id: str) -> int:
    return int(event_id.rsplit("_", 1)[-1])


async def main() -> int:
    settings = Settings()
    token = settings.gfw_api_token
    if not token:
        print("OS_GFW_API_TOKEN missing")
        return 1

    # 1. Collect oc01 samples + derive chunk UTC timestamps
    samples = []
    with open(CORPUS) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            prov = d.get("provenance", {}).get("source_id", "")
            if "oc01" not in prov.lower():
                continue
            sf = d.get("provenance", {}).get("source_file", "")
            start = parse_recording_start(sf)
            if not start:
                continue
            idx = chunk_idx_from_event(d["event_id"])
            chunk_utc = start + timedelta(seconds=idx * 60)
            samples.append({
                "event_id": d["event_id"],
                "source_file": sf,
                "recording_start_utc": start.isoformat(),
                "chunk_idx": idx,
                "chunk_start_utc": chunk_utc.isoformat(),
                "chunk_hour_utc": chunk_utc.replace(minute=0, second=0, microsecond=0).isoformat(),
                "current_label": d["label"],
            })
    print(f"oc01 samples: {len(samples)}")

    # 2. Days we need to query (unique YYYY-MM-DD)
    days = sorted({datetime.fromisoformat(s["chunk_start_utc"]).strftime("%Y-%m-%d") for s in samples})
    print(f"days to query: {days}")

    # 3. For each (day × radius), pull hourly AIS rows. Build a
    # (radius, hour_utc_iso) → set of vessel ids map.
    presence: dict[tuple[float, str], set[str]] = defaultdict(set)
    async with httpx.AsyncClient(timeout=60.0) as client:
        for day_str in days:
            day = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            for radius in RADII_KM:
                print(f"  GFW query day={day_str} radius={radius}km ...", end=" ", flush=True)
                try:
                    rows = await query_hourly(client, token, day, radius)
                except httpx.HTTPStatusError as e:
                    print(f"FAILED: HTTP {e.response.status_code}: {e.response.text[:200]}")
                    continue
                except Exception as e:
                    print(f"FAILED: {type(e).__name__}: {e}")
                    continue
                print(f"got {len(rows)} rows")
                for row in rows:
                    # entryTimestamp typically: "2019-03-09T12:00:00.000Z"
                    ts_str = row.get("entryTimestamp") or row.get("date")
                    if not ts_str:
                        continue
                    try:
                        hour_iso = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(
                            minute=0, second=0, microsecond=0,
                        ).isoformat()
                    except Exception:
                        continue
                    vid = row.get("vesselId") or row.get("vessel_id") or row.get("ssvid") or "unknown"
                    presence[(radius, hour_iso)].add(str(vid))

    # 4. For each chunk, determine vessel presence at each radius
    for s in samples:
        s["ais"] = {}
        for radius in RADII_KM:
            vessels = presence.get((radius, s["chunk_hour_utc"]), set())
            s["ais"][f"r{int(radius)}km"] = {
                "n_vessels": len(vessels),
                "vessel_ids": sorted(vessels)[:10],
            }

    # 5. Cross-tab: current_label × AIS presence
    print()
    print("=== CROSS-TAB: current_label × AIS presence ===")
    for radius in RADII_KM:
        print(f"\nradius {radius} km:")
        ct: Counter = Counter()
        for s in samples:
            present = s["ais"][f"r{int(radius)}km"]["n_vessels"] > 0
            ct[(s["current_label"], present)] += 1
        for (lbl, present), n in sorted(ct.items()):
            print(f"  label={lbl:9s}  AIS_present={present!s:5s}  n={n}")

    # 6. Per-recording summary
    print()
    print("=== PER-RECORDING AIS ANALYSIS ===")
    by_file: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for s in samples:
        sf = s["source_file"]
        by_file[sf]["n_chunks"] += 1
        by_file[sf][f"label_{s['current_label']}"] += 1
        for radius in RADII_KM:
            present = s["ais"][f"r{int(radius)}km"]["n_vessels"] > 0
            if present:
                by_file[sf][f"ais_present_r{int(radius)}km"] += 1
    for sf, stats in by_file.items():
        print(f"\n{sf}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    # 7. Hour-level table for the "ambient" file specifically
    print()
    print("=== HOUR-BY-HOUR: 03-09T05:59 'ambient' recording ===")
    amb_chunks = [s for s in samples if "20190309T055952" in s["event_id"]]
    if amb_chunks:
        hours = sorted({s["chunk_hour_utc"] for s in amb_chunks})
        print(f"hours covered: {hours}")
        for h in hours:
            chunks_in_h = [s for s in amb_chunks if s["chunk_hour_utc"] == h]
            n_chunks = len(chunks_in_h)
            n5 = len(chunks_in_h[0]["ais"]["r5km"]["vessel_ids"]) if chunks_in_h else 0
            n10 = len(chunks_in_h[0]["ais"]["r10km"]["vessel_ids"]) if chunks_in_h else 0
            n50 = len(chunks_in_h[0]["ais"]["r50km"]["vessel_ids"]) if chunks_in_h else 0
            print(f"  {h}  chunks={n_chunks}  vessels@5km={n5}  @10km={n10}  @50km={n50}")

    # Save full per-chunk audit
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "site": "oc01",
        "lat": OC01_LAT,
        "lon": OC01_LON,
        "radii_km": list(RADII_KM),
        "n_chunks": len(samples),
        "per_chunk": samples,
        "by_file": dict(by_file),
    }, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
