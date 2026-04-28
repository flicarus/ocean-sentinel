"""Verify the 10 km AIS-cross-reference radius is robust, not knife-edge.

Two checks:

1. RADIUS SWEEP
   For radii 5, 10, 20 km — recompute flip count on the sanctsound
   subset. If 10 km is robust, flip rates change gradually. If 5 km
   shows zero flips and 20 km shows 95%, 10 km is on a steep cliff and
   we need different validation.

2. CNN AGREEMENT ON VAL (independent verification)
   v6 was trained on the OLD sanctsound labels (everything = not_ship).
   For the 60 sanctsound val clips, what does v6 predict? Cross with
   what relabel_sanctsound.py decided at 10 km. Strong evidence:
     - CNN said `ship` AND AIS-relabel said `ship` → both agree, despite
       CNN being trained to call it `not_ship`. This is the smoking gun.
     - CNN said `not_ship` AND AIS-relabel said `not_ship` → both agree,
       trivial.
     - CNN said `ship` AND AIS-relabel said `not_ship` → AIS missed?
     - CNN said `not_ship` AND AIS-relabel said `ship` → AIS over-eager?
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train_cnn import LABEL_MAP, SpecDataset, _split_session_indices
from ocean_sentinel.config import Settings
from ocean_sentinel.services.cnn_classifier import CNNClassifier


SITE_COORDS: dict[str, tuple[float, float]] = {
    "oc01": (48.400, -124.700),
    "sb01": (42.43668, -70.546655),
}
RADII_KM = (5.0, 10.0, 20.0)
GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"


def parse_event_id(event_id: str) -> tuple[str, str, int] | None:
    parts = event_id.split("_")
    if len(parts) < 4 or parts[0] != "sanctsound":
        return None
    site = parts[1]
    try:
        chunk = int(parts[-1])
    except ValueError:
        return None
    timestamp_part = next(
        (p for p in parts if p.endswith("Z") and len(p) == 16 and "T" in p),
        None,
    )
    if not timestamp_part:
        return None
    return site, timestamp_part, chunk


def derive_capture_time(timestamp_str: str, chunk_idx: int) -> datetime:
    ts = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc,
    )
    return ts + timedelta(seconds=chunk_idx * 60)


def bbox(lat: float, lon: float, radius_km: float) -> dict:
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


async def query_day(
    client: httpx.AsyncClient, token: str,
    site: str, day: datetime, radius_km: float,
) -> set[int]:
    lat, lon = SITE_COORDS[site]
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
        url, json=bbox(lat, lon, radius_km),
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    rows: list[dict] = []
    for entry in resp.json().get("entries", []):
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.startswith("public-") and isinstance(v, list):
                    rows.extend(v)
    hours: set[int] = set()
    for r in rows:
        ts_str = r.get("entryTimestamp") or r.get("date")
        if not ts_str:
            continue
        try:
            t = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if t.date() == day.date():
                hours.add(t.hour)
        except Exception:
            continue
    return hours


async def main() -> None:
    settings = Settings()
    if not settings.gfw_api_token:
        sys.exit("OS_GFW_API_TOKEN missing")

    # Load all sanctsound entries.
    sanctsound: list[dict] = []
    with open("data/training/gemma_labels.v5.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e.get("provenance", {}).get("source_id") == "sanctsound":
                sanctsound.append(e)

    print(f"Sanctsound clips: {len(sanctsound)}")

    # Identify unique (site, day) pairs once.
    plan: list[tuple[str, datetime]] = []
    seen = set()
    for e in sanctsound:
        parsed = parse_event_id(e["event_id"])
        if not parsed:
            continue
        site, ts_str, chunk = parsed
        capture = derive_capture_time(ts_str, chunk)
        day = capture.replace(hour=0, minute=0, second=0, microsecond=0)
        key = (site, day)
        if key not in seen:
            seen.add(key)
            plan.append(key)

    print(f"Unique (site, day) pairs: {len(plan)}")

    # === CHECK 1: radius sweep ===
    print(f"\n=== CHECK 1: radius sweep ===")
    radius_results: dict[float, dict[tuple[str, datetime], set[int]]] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        for radius in RADII_KM:
            print(f"\nRadius {radius} km:")
            cache: dict[tuple[str, datetime], set[int]] = {}
            for site, day in plan:
                hours = await query_day(
                    client, settings.gfw_api_token, site, day, radius,
                )
                cache[(site, day)] = hours
                print(f"  {site} {day.date()}: vessel-active hours = "
                      f"{sorted(hours)} ({len(hours)} hours)")
            radius_results[radius] = cache

    # Compute flip count per radius.
    print(f"\n{'radius km':>10s} {'flips→ship':>12s} {'flip%':>8s}")
    print("-" * 35)
    flips_by_radius: dict[float, list[dict]] = {r: [] for r in RADII_KM}
    for radius in RADII_KM:
        cache = radius_results[radius]
        for e in sanctsound:
            parsed = parse_event_id(e["event_id"])
            if not parsed:
                continue
            site, ts_str, chunk = parsed
            capture = derive_capture_time(ts_str, chunk)
            day = capture.replace(hour=0, minute=0, second=0, microsecond=0)
            active = cache.get((site, day), set())
            new_label = "ship" if capture.hour in active else "not_ship"
            if new_label != e["label"]:
                flips_by_radius[radius].append(e)
        n = len(flips_by_radius[radius])
        print(f"{radius:>10.0f} {n:>12d} {n / len(sanctsound):>7.1%}")

    # === CHECK 2: CNN agreement on val subset ===
    print(f"\n=== CHECK 2: CNN-AIS agreement on val (60 sanctsound) ===")
    print("v6 was trained on OLD labels (all sanctsound = not_ship). If CNN")
    print("now says `ship` for clips AIS-relabel also flips → both methods")
    print("independently agree the original label was wrong.\n")

    ds = SpecDataset(
        "data/training/gemma_labels.v5.jsonl",
        representation_version="abs_db_v1",
        augment=False,
    )
    _, val_idx = _split_session_indices(ds.entries)
    sanctsound_val = [
        ds.entries[i] for i in val_idx
        if ds.entries[i]["provenance"]["source_id"] == "sanctsound"
    ]
    print(f"Sanctsound val clips: {len(sanctsound_val)}")

    clf = CNNClassifier("data/models/cnn_v6.pt")

    cache_10km = radius_results[10.0]
    confusion = Counter()  # (cnn_pred, ais_label) → count
    disagreements: list[dict] = []

    for e in sanctsound_val:
        spec = np.load(e["spectrogram_path"]).astype(np.float32)
        cnn_out = clf.predict(spec, source_id="sanctsound")
        cnn_pred = cnn_out["label"]
        cnn_conf = cnn_out["confidence"]

        parsed = parse_event_id(e["event_id"])
        if not parsed:
            continue
        site, ts_str, chunk = parsed
        capture = derive_capture_time(ts_str, chunk)
        day = capture.replace(hour=0, minute=0, second=0, microsecond=0)
        active = cache_10km.get((site, day), set())
        ais_label = "ship" if capture.hour in active else "not_ship"

        confusion[(cnn_pred, ais_label)] += 1
        if cnn_pred != ais_label:
            disagreements.append({
                "event_id": e["event_id"],
                "cnn_pred": cnn_pred,
                "cnn_conf": cnn_conf,
                "ais_label": ais_label,
                "site": site,
                "hour": capture.hour,
                "vessel_active_hours": sorted(active),
            })

    print(f"\nConfusion matrix (CNN pred × AIS label):")
    print(f"{'':<14s} {'AIS=not_ship':>13s} {'AIS=ship':>10s}")
    for cnn in ("not_ship", "ship"):
        print(
            f"{'CNN='+cnn:<14s} "
            f"{confusion.get((cnn, 'not_ship'), 0):>13d} "
            f"{confusion.get((cnn, 'ship'), 0):>10d}"
        )

    n = len(sanctsound_val)
    agree = confusion.get(("not_ship", "not_ship"), 0) + confusion.get(("ship", "ship"), 0)
    print(f"\nAgreement: {agree}/{n} = {agree / n:.1%}")
    print()
    print("Interpretation:")
    print("  CNN=ship,     AIS=ship      → CONVERGENT (both flag despite original 'not_ship') ⭐")
    print("  CNN=not_ship, AIS=not_ship  → CONVERGENT trivial (both keep original)")
    print("  CNN=ship,     AIS=not_ship  → CNN sees vessel AIS missed (wrong direction for AIS)")
    print("  CNN=not_ship, AIS=ship      → AIS over-eager? (CNN doesn't hear vessel)")

    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for d in disagreements[:20]:
            print(
                f"  {d['event_id']:<60s} "
                f"CNN={d['cnn_pred']}({d['cnn_conf']:.2f}) "
                f"AIS={d['ais_label']} "
                f"site={d['site']} hour={d['hour']:02d}"
            )


if __name__ == "__main__":
    asyncio.run(main())
