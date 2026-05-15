"""Phase 2.5: per-chunk named vessel + exact CPA distance for OC01 corpus.

Pulls hourly AIS at 50 km (catch all vessels in the area) for the
2-day window, then for each row computes haversine distance to oc01.
For each of 800 chunks, identifies:
  - the closest named vessel active in that hour
  - exact distance (km) from oc01 hydrophone
  - vessel name, flag, class, mmsi, imo

This is the Part-I-style ground truth applied to all 800 chunks:
named vessels and exact CPA distances, not just "AIS within X km".

Output: data/audit/oc01_named_vessels.json + summary table
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import httpx

from ocean_sentinel.config import Settings


OC01_LAT, OC01_LON = 48.400, -124.700
GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"
QUERY_RADIUS_KM = 50.0  # wide enough to catch every vessel in the Strait

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/training/sanctsound_corrected.jsonl"
AIS_JSON = ROOT / "data/audit/oc01_ais_per_chunk.json"
OUT_JSON = ROOT / "data/audit/oc01_named_vessels.json"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    a, b, c, d = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = c - a, d - b
    h = sin(dlat / 2) ** 2 + cos(a) * cos(c) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(h))


def bbox(radius_km):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * cos(radians(OC01_LAT)))
    return {"geojson": {"type": "Polygon", "coordinates": [[
        [OC01_LON - dlon, OC01_LAT - dlat],
        [OC01_LON + dlon, OC01_LAT - dlat],
        [OC01_LON + dlon, OC01_LAT + dlat],
        [OC01_LON - dlon, OC01_LAT + dlat],
        [OC01_LON - dlon, OC01_LAT - dlat],
    ]]}}


async def fetch_day_hourly(client, token, day_str):
    next_day = (datetime.strptime(day_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"{GFW_BASE}/4wings/report"
        f"?datasets[0]=public-global-presence:latest"
        f"&date-range={day_str},{next_day}"
        f"&temporal-resolution=HOURLY"
        f"&spatial-resolution=HIGH"
        f"&group-by=VESSEL_ID"
        f"&format=JSON"
    )
    r = await client.post(url, json=bbox(QUERY_RADIUS_KM),
                          headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    data = r.json()
    rows = []
    for entry in data.get("entries", []):
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.startswith("public-") and isinstance(v, list):
                    rows.extend(v)
    return rows


async def main():
    settings = Settings()
    token = settings.gfw_api_token

    # 1. Pull all hourly rows for 03-08 and 03-09
    async with httpx.AsyncClient(timeout=60) as client:
        rows = []
        for d in ["2019-03-08", "2019-03-09"]:
            print(f"querying GFW {d} ...", end=" ", flush=True)
            try:
                day_rows = await fetch_day_hourly(client, token, d)
                print(f"got {len(day_rows)}")
                rows.extend(day_rows)
            except Exception as e:
                print(f"FAIL {e}")
                return 1

    # 2. For each row, compute distance to oc01 and bucket by (hour, vesselId)
    by_hour_vid: dict[tuple[str, str], dict] = {}
    for r in rows:
        ts = r.get("entryTimestamp") or r.get("date")
        if not ts:
            continue
        try:
            hour_iso = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(
                minute=0, second=0, microsecond=0
            ).isoformat()
        except Exception:
            continue
        vid = r.get("vesselId", "unknown")
        lat, lon = r.get("lat"), r.get("lon")
        if lat is None or lon is None:
            continue
        dist = haversine_km(OC01_LAT, OC01_LON, lat, lon)
        info = {
            "vessel_id": vid,
            "ship_name": r.get("shipName") or "(unnamed)",
            "vessel_type": r.get("vesselType") or r.get("geartype") or "OTHER",
            "flag": r.get("flag") or "?",
            "mmsi": r.get("mmsi") or "",
            "imo": r.get("imo") or "",
            "lat": lat,
            "lon": lon,
            "distance_km": dist,
            "hour_iso": hour_iso,
        }
        key = (hour_iso, vid)
        # If same vessel reported in multiple bands same hour, keep the closest
        if key not in by_hour_vid or dist < by_hour_vid[key]["distance_km"]:
            by_hour_vid[key] = info

    # 3. By hour: list of vessels with distances
    by_hour: dict[str, list[dict]] = defaultdict(list)
    for (hour, _vid), info in by_hour_vid.items():
        by_hour[hour].append(info)
    for h in by_hour:
        by_hour[h].sort(key=lambda v: v["distance_km"])

    # 4. Load per-chunk AIS file (or rebuild chunk timestamps from corpus)
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
            name = sf.split("/")[-1]
            ts_part = name.split("_")[-1].replace(".flac", "")
            try:
                start = datetime.strptime(ts_part, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            idx = int(d["event_id"].rsplit("_", 1)[-1])
            chunk_start = start + timedelta(seconds=idx * 60)
            hour_iso = chunk_start.replace(minute=0, second=0, microsecond=0).isoformat()
            vessels = by_hour.get(hour_iso, [])
            nearest = vessels[0] if vessels else None
            samples.append({
                "event_id": d["event_id"],
                "source_file": sf,
                "chunk_idx": idx,
                "chunk_start_utc": chunk_start.isoformat(),
                "chunk_hour_utc": hour_iso,
                "current_label": d["label"],
                "n_vessels_in_hour_within_50km": len(vessels),
                "nearest_vessel": nearest,
            })

    # 5. Cross-tab + per-recording-and-hour table
    print()
    print("=== CROSS-TAB by current_label × nearest vessel distance ===")
    bucket = Counter()
    for s in samples:
        d = s["nearest_vessel"]["distance_km"] if s["nearest_vessel"] else None
        if d is None:
            band = "no_AIS"
        elif d <= 5:
            band = "<=5km"
        elif d <= 10:
            band = "5-10km"
        elif d <= 20:
            band = "10-20km"
        else:
            band = "20-50km"
        bucket[(s["current_label"], band)] += 1
    for k, v in sorted(bucket.items()):
        print(f"  label={k[0]:9s}  nearest={k[1]:10s}  n={v}")

    # 6. Per-recording hour-by-hour with named vessels
    print()
    print("=== PER-RECORDING HOUR-BY-HOUR (named vessels @ exact distance) ===")
    by_file_hour: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        by_file_hour[(s["source_file"], s["chunk_hour_utc"])].append(s)
    files_order = sorted({s["source_file"] for s in samples})
    for sf in files_order:
        print(f"\n{sf}")
        hours = sorted({s["chunk_hour_utc"] for s in samples if s["source_file"] == sf})
        for h in hours:
            chunks = by_file_hour[(sf, h)]
            n = len(chunks)
            lbl = chunks[0]["current_label"]
            v_list = by_hour.get(h, [])
            v_within_10 = [v for v in v_list if v["distance_km"] <= 10]
            nearest = v_list[0] if v_list else None
            nearest_str = (
                f"{nearest['ship_name']} ({nearest['flag']}) @ {nearest['distance_km']:.1f}km"
                if nearest else "(none)"
            )
            print(f"  {h}  chunks={n:3d}  label={lbl:9s}  nearest={nearest_str:60s}  v@<=10km={len(v_within_10)}")
            for v in v_within_10[:5]:
                print(f"     · {v['ship_name']} ({v['vessel_type']}, {v['flag']}) @ {v['distance_km']:.2f}km")

    # 7. The 118 "mislabeled ambient" chunks — list named vessels
    print()
    print("=== THE 118 MISLABELED AMBIENT CHUNKS — named vessels @ ≤10km ===")
    miss_amb = [s for s in samples if s["current_label"] == "not_ship"
                and s["nearest_vessel"] and s["nearest_vessel"]["distance_km"] <= 10]
    print(f"count: {len(miss_amb)}")
    vessel_freq = Counter()
    for s in miss_amb:
        v = s["nearest_vessel"]
        vessel_freq[(v["ship_name"], v["flag"], v["vessel_type"])] += 1
    for (n, f, t), c in vessel_freq.most_common():
        print(f"  · {n} ({t}, {f}): {c} chunks")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "site": "oc01",
        "lat": OC01_LAT,
        "lon": OC01_LON,
        "query_radius_km": QUERY_RADIUS_KM,
        "by_hour": {h: by_hour[h] for h in sorted(by_hour)},
        "per_chunk": samples,
    }, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
