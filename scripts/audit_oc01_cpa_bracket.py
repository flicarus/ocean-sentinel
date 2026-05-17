"""Phase 2.6: bracket each vessel's closest-point-of-approach (CPA) per
hour using progressively smaller bbox queries.

A vessel reported in a 3km bbox got within ~4.2km of OC01 at SOME point
in that hour, even if its single reported lat/lon happened to be at the
bbox edge. By querying at radii [1, 2, 3, 5, 7, 10, 15, 25, 50] km we
bracket per-vessel-per-hour CPA tightly enough for chunk-level labels.

Output per chunk:
  - nearest vessel (name, flag, class)
  - CPA upper bound (smallest bbox the vessel appears in, in km)
  - CPA lower bound (largest bbox the vessel does NOT appear in)
  - hourly-bbox confidence: bracketed within {lower, upper}

Data: data/audit/oc01_cpa_bracket.json
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
RADII_KM = [1, 2, 3, 5, 7, 10, 15, 25, 50]  # ascending — small ones bracket CPA tighter

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/training/sanctsound_corrected.jsonl"
OUT_JSON = ROOT / "data/audit/oc01_cpa_bracket.json"


def bbox(radius_km: float) -> dict:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * cos(radians(OC01_LAT)))
    return {"geojson": {"type": "Polygon", "coordinates": [[
        [OC01_LON - dlon, OC01_LAT - dlat],
        [OC01_LON + dlon, OC01_LAT - dlat],
        [OC01_LON + dlon, OC01_LAT + dlat],
        [OC01_LON - dlon, OC01_LAT + dlat],
        [OC01_LON - dlon, OC01_LAT - dlat],
    ]]}}


async def fetch_day_radius(client, token, day_str, radius_km):
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
    r = await client.post(url, json=bbox(radius_km),
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

    # 1. Pull AIS at every radius × every day
    # presence[(radius_km, hour_iso, vessel_id)] = vessel_metadata
    presence: dict[tuple[float, str, str], dict] = {}
    async with httpx.AsyncClient(timeout=60) as client:
        for d in ["2019-03-08", "2019-03-09"]:
            for radius in RADII_KM:
                print(f"  GFW r={radius:>2}km day={d} ...", end=" ", flush=True)
                try:
                    rows = await fetch_day_radius(client, token, d, radius)
                except Exception as e:
                    print(f"FAIL {type(e).__name__}: {e}")
                    continue
                print(f"got {len(rows)} rows")
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
                    presence[(radius, hour_iso, vid)] = {
                        "ship_name": r.get("shipName") or "(unnamed)",
                        "vessel_type": r.get("vesselType") or r.get("geartype") or "OTHER",
                        "flag": r.get("flag") or "?",
                        "mmsi": r.get("mmsi") or "",
                        "lat": r.get("lat"),
                        "lon": r.get("lon"),
                    }

    # 2. For each (hour, vid): smallest radius where vessel appears = CPA upper bound
    hour_vid_cpa: dict[tuple[str, str], dict] = {}
    for (radius, hour_iso, vid), meta in presence.items():
        key = (hour_iso, vid)
        if key not in hour_vid_cpa:
            hour_vid_cpa[key] = {
                **meta,
                "cpa_upper_km": radius,
                "cpa_lower_km": 0,
                "radii_present": {radius},
                "radii_absent": set(),
            }
        else:
            entry = hour_vid_cpa[key]
            entry["radii_present"].add(radius)
            if radius < entry["cpa_upper_km"]:
                entry["cpa_upper_km"] = radius

    # Now compute lower bound: largest radius where vessel is ABSENT
    all_hours = {h for h, _ in hour_vid_cpa}
    all_vids = {v for _, v in hour_vid_cpa}
    for key, entry in hour_vid_cpa.items():
        absent = [r for r in RADII_KM if r not in entry["radii_present"] and r < entry["cpa_upper_km"]]
        entry["cpa_lower_km"] = max(absent) if absent else 0
        entry["radii_present"] = sorted(entry["radii_present"])
        entry["radii_absent"] = sorted(absent)

    # 3. For each chunk, find vessel with smallest CPA upper bound in its hour
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
            name = sf.split("/")[-1].replace(".flac", "")
            ts_part = name.split("_")[-1]
            try:
                start = datetime.strptime(ts_part, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            idx = int(d["event_id"].rsplit("_", 1)[-1])
            chunk_start = start + timedelta(seconds=idx * 60)
            hour_iso = chunk_start.replace(minute=0, second=0, microsecond=0).isoformat()
            # vessels in this hour, sorted by CPA upper bound
            in_hour = [(vid, hour_vid_cpa[(hour_iso, vid)])
                       for (h, vid) in hour_vid_cpa if h == hour_iso]
            in_hour.sort(key=lambda x: x[1]["cpa_upper_km"])
            nearest = in_hour[0][1] if in_hour else None
            samples.append({
                "event_id": d["event_id"],
                "source_file": sf,
                "chunk_idx": idx,
                "chunk_hour_utc": hour_iso,
                "current_label": d["label"],
                "n_vessels_in_hour": len(in_hour),
                "nearest_vessel": nearest,
            })

    # 4. Cross-tab by current_label × CPA upper bound bucket
    print()
    print("=== CROSS-TAB: current_label × nearest vessel CPA bracket ===")
    bucket = Counter()
    for s in samples:
        v = s["nearest_vessel"]
        if not v:
            band = "no_AIS"
        elif v["cpa_upper_km"] <= 1:
            band = "≤1km"
        elif v["cpa_upper_km"] <= 2:
            band = "≤2km"
        elif v["cpa_upper_km"] <= 3:
            band = "≤3km"
        elif v["cpa_upper_km"] <= 5:
            band = "≤5km"
        elif v["cpa_upper_km"] <= 7:
            band = "≤7km"
        elif v["cpa_upper_km"] <= 10:
            band = "≤10km"
        elif v["cpa_upper_km"] <= 15:
            band = "≤15km"
        elif v["cpa_upper_km"] <= 25:
            band = "≤25km"
        else:
            band = "≤50km"
        bucket[(s["current_label"], band)] += 1
    bands_order = ["≤1km", "≤2km", "≤3km", "≤5km", "≤7km", "≤10km", "≤15km", "≤25km", "≤50km", "no_AIS"]
    print(f"  {'label':<12} {'CPA bracket':<10} {'n':>5}")
    for lbl in ["ship", "not_ship"]:
        for b in bands_order:
            n = bucket.get((lbl, b), 0)
            if n:
                print(f"  {lbl:<12} {b:<10} {n:>5}")

    # 5. Per-recording hour table
    print()
    print("=== PER-RECORDING HOUR TABLE (CPA-bracketed) ===")
    by_file_hour: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        by_file_hour[(s["source_file"], s["chunk_hour_utc"])].append(s)
    for sf in sorted({s["source_file"] for s in samples}):
        print(f"\n{sf}")
        hours = sorted({s["chunk_hour_utc"] for s in samples if s["source_file"] == sf})
        for h in hours:
            chunks = by_file_hour[(sf, h)]
            lbl = chunks[0]["current_label"]
            in_hour = [(vid, hour_vid_cpa[(h, vid)])
                       for (hh, vid) in hour_vid_cpa if hh == h]
            in_hour.sort(key=lambda x: x[1]["cpa_upper_km"])
            print(f"  {h}  n={len(chunks):3d}  label={lbl:9s}  vessels={len(in_hour)}")
            for vid, v in in_hour[:5]:
                bracket = f"{v['cpa_lower_km']}-{v['cpa_upper_km']}km" if v["cpa_lower_km"] else f"≤{v['cpa_upper_km']}km"
                print(f"     · {v['ship_name']:30s} ({v['vessel_type']:7s}, {v['flag']:3s}) CPA {bracket}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "site": "oc01",
        "lat": OC01_LAT,
        "lon": OC01_LON,
        "radii_km": RADII_KM,
        "per_chunk": samples,
        "by_hour": {
            h: sorted(
                [{**hour_vid_cpa[(h, vid)], "vessel_id": vid} for (hh, vid) in hour_vid_cpa if hh == h],
                key=lambda x: x["cpa_upper_km"],
            )
            for h in sorted({h for h, _ in hour_vid_cpa})
        },
    }, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
