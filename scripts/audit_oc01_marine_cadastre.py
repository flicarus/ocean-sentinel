"""Phase 3: ground truth from MarineCadastre.gov raw AIS archive.

NOAA-published, second-level vessel positions, US waters. This is the
GOVERNMENT SOURCE for AIS in US waters — what NOAA themselves use.

Workflow:
  1. Filter daily AIS CSVs to bbox around OC01 (lat 47.5-49.5, lon -126 -> -123)
  2. Compute per-message distance to OC01 hydrophone
  3. For each named vessel in the area, find:
     - CPA (closest distance, with exact UTC timestamp)
     - track (distance over time)
  4. Cross-tab vs 800 chunks: chunk timestamp → which vessel was nearest
     at that exact second, at what exact distance.

Output: data/audit/oc01_mc_track.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_FILES = [
    ROOT / "data/audit/marine_cadastre/AIS_2019_03_08.csv",
    ROOT / "data/audit/marine_cadastre/AIS_2019_03_09.csv",
]
CORPUS = ROOT / "data/training/sanctsound_corrected.jsonl"
OUT_JSON = ROOT / "data/audit/oc01_mc_track.json"

OC01_LAT, OC01_LON = 48.400, -124.700
# Wide bbox around OC01 (within ~100 km)
LAT_LO, LAT_HI = 47.5, 49.5
LON_LO, LON_HI = -126.0, -123.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    a, b, c, d = map(radians, (lat1, lon1, lat2, lon2))
    h = sin((c - a) / 2) ** 2 + cos(a) * cos(c) * sin((d - b) / 2) ** 2
    return 2 * R * asin(sqrt(h))


def parse_csv(path: Path):
    """Yield rows filtered to oc01 area."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["LAT"])
                lon = float(row["LON"])
            except (KeyError, ValueError):
                continue
            if not (LAT_LO <= lat <= LAT_HI and LON_LO <= lon <= LON_HI):
                continue
            yield row


def main():
    # 1. Build per-vessel position track from both days (filtered to oc01 area)
    tracks: dict[str, list[dict]] = defaultdict(list)
    total = filtered = 0
    for csv_path in CSV_FILES:
        if not csv_path.exists():
            print(f"missing {csv_path}")
            continue
        print(f"parsing {csv_path.name} (filter to bbox around oc01)...")
        n = 0
        for row in parse_csv(csv_path):
            n += 1
            mmsi = row.get("MMSI", "")
            try:
                ts = datetime.strptime(row["BaseDateTime"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            lat = float(row["LAT"])
            lon = float(row["LON"])
            dist = haversine_km(OC01_LAT, OC01_LON, lat, lon)
            tracks[mmsi].append({
                "ts": ts,
                "lat": lat,
                "lon": lon,
                "distance_km": dist,
                "vessel_name": row.get("VesselName", "").strip(),
                "vessel_type": row.get("VesselType", ""),
                "imo": row.get("IMO", "").strip(),
                "sog": row.get("SOG", ""),
                "cog": row.get("COG", ""),
                "length": row.get("Length", ""),
            })
        filtered += n
        print(f"  {n} rows in bbox")
    print(f"total rows in bbox over 2 days: {filtered}")
    print(f"unique MMSIs in bbox: {len(tracks)}")

    # 2. For each MMSI, find CPA + timestamp
    cpa_per_mmsi: list[dict] = []
    for mmsi, points in tracks.items():
        if not points:
            continue
        points.sort(key=lambda p: p["ts"])
        cpa_pt = min(points, key=lambda p: p["distance_km"])
        cpa_per_mmsi.append({
            "mmsi": mmsi,
            "vessel_name": cpa_pt["vessel_name"] or "(unknown)",
            "vessel_type": cpa_pt["vessel_type"],
            "cpa_km": cpa_pt["distance_km"],
            "cpa_ts": cpa_pt["ts"].isoformat(),
            "n_pings": len(points),
            "first_ping": points[0]["ts"].isoformat(),
            "last_ping": points[-1]["ts"].isoformat(),
        })

    cpa_per_mmsi.sort(key=lambda v: v["cpa_km"])

    # 3. Print top 30 closest vessels
    print()
    print("=== TOP 30 CLOSEST VESSELS TO OC01 (2019-03-08 + 2019-03-09) ===")
    print(f"  {'CPA km':>7}  {'name':<32s}  {'mmsi':<10s}  {'when (UTC)':<20s}  pings")
    for v in cpa_per_mmsi[:30]:
        print(f"  {v['cpa_km']:7.2f}  {v['vessel_name']:<32s}  {v['mmsi']:<10s}  {v['cpa_ts']:<20s}  {v['n_pings']}")

    # 4. WIND SONG + BALOS specific check — track distance over hour 07
    print()
    print("=== WIND SONG and BALOS — track during hour 07 UTC on 2019-03-09 ===")
    target_names = ["WIND SONG", "BALOS"]
    for target in target_names:
        # find MMSI by name
        candidates = [(m, pts) for m, pts in tracks.items()
                      if any(p["vessel_name"].upper() == target for p in pts)]
        if not candidates:
            print(f"  {target}: NO POSITIONS FOUND in bbox")
            continue
        for mmsi, points in candidates:
            hour07 = [p for p in points
                      if datetime(2019, 3, 9, 7, tzinfo=timezone.utc) <= p["ts"]
                      < datetime(2019, 3, 9, 8, tzinfo=timezone.utc)]
            hour05 = [p for p in points
                      if datetime(2019, 3, 9, 5, tzinfo=timezone.utc) <= p["ts"]
                      < datetime(2019, 3, 9, 6, tzinfo=timezone.utc)]
            hour06 = [p for p in points
                      if datetime(2019, 3, 9, 6, tzinfo=timezone.utc) <= p["ts"]
                      < datetime(2019, 3, 9, 7, tzinfo=timezone.utc)]
            print(f"\n  {target} (MMSI {mmsi})")
            for hr, pts, label in [(5, hour05, "hour05"), (6, hour06, "hour06"), (7, hour07, "hour07")]:
                if not pts:
                    print(f"    {label}: 0 pings")
                    continue
                dists = [p["distance_km"] for p in pts]
                cpa = min(pts, key=lambda p: p["distance_km"])
                print(f"    {label}: {len(pts)} pings  min={min(dists):.2f}km  mean={sum(dists)/len(dists):.2f}km  max={max(dists):.2f}km  CPA@{cpa['ts']}")

    # 5. Cross-tab: per-chunk → nearest vessel @ chunk start time
    # Load chunks
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
            samples.append({
                "event_id": d["event_id"],
                "source_file": sf,
                "chunk_idx": idx,
                "chunk_start_utc": chunk_start.isoformat(),
                "current_label": d["label"],
            })

    # For each chunk, find vessel with smallest distance within ±2 min window
    print()
    print("=== PER-CHUNK NEAREST VESSEL FROM RAW MC AIS (±2 min window) ===")
    by_label_band: dict[tuple[str, str], int] = defaultdict(int)
    for s in samples:
        chunk_ts = datetime.fromisoformat(s["chunk_start_utc"])
        lo = chunk_ts - timedelta(minutes=2)
        hi = chunk_ts + timedelta(seconds=60 + 2 * 60)  # chunk + 2 min after
        nearest_dist = None
        nearest = None
        for mmsi, points in tracks.items():
            for p in points:
                if lo <= p["ts"] <= hi:
                    if nearest_dist is None or p["distance_km"] < nearest_dist:
                        nearest_dist = p["distance_km"]
                        nearest = {
                            "ts": p["ts"].isoformat(),
                            "vessel_name": p["vessel_name"] or "(unknown)",
                            "vessel_type": p["vessel_type"],
                            "mmsi": mmsi,
                            "distance_km": p["distance_km"],
                        }
        s["nearest_in_window"] = nearest
        d = nearest_dist
        if d is None:
            band = "no_AIS"
        elif d <= 3:
            band = "≤3km"
        elif d <= 5:
            band = "≤5km"
        elif d <= 7:
            band = "≤7km"
        elif d <= 10:
            band = "≤10km"
        elif d <= 15:
            band = "≤15km"
        else:
            band = ">15km"
        by_label_band[(s["current_label"], band)] += 1
        s["cpa_band"] = band

    print(f"  {'label':<12} {'CPA':<8} {'n':>5}")
    for (lbl, b), n in sorted(by_label_band.items()):
        print(f"  {lbl:<12} {b:<8} {n:>5}")

    # 6. Save full per-chunk results + per-vessel CPA
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "source": "MarineCadastre.gov AIS archive (NOAA Office for Coastal Management)",
        "days": ["2019-03-08", "2019-03-09"],
        "bbox": {"lat_lo": LAT_LO, "lat_hi": LAT_HI, "lon_lo": LON_LO, "lon_hi": LON_HI},
        "oc01": {"lat": OC01_LAT, "lon": OC01_LON},
        "n_chunks": len(samples),
        "vessel_cpa": cpa_per_mmsi[:50],
        "per_chunk": samples,
        "by_label_band": {f"{k[0]}|{k[1]}": v for k, v in by_label_band.items()},
    }, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
