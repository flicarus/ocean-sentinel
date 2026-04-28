"""OC01 proximity audit — shrink the radius until we know who was there.

We have "38 vessels within 50 km on 2019-03-09" — daily resolution, too
coarse to claim audit-grade audibility. This script re-queries GFW
4Wings/report with TWO upgrades:
  - temporal-resolution=HOURLY (vs DAILY)  ⇒ per-hour vessel presence
  - bbox sweep at 50 / 20 / 10 / 5 km radii ⇒ tighter spatial bound

For each radius, prints which named vessels were present during the
suspect hours (12:00-13:00 UTC). Audit-grade outcome: a named vessel at
≤ 10 km exactly during the chunks our CNN flagged.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from math import cos, radians

import httpx

from ocean_sentinel.config import Settings


OC01_LAT, OC01_LON = 48.400, -124.700
DAY = datetime(2019, 3, 9, tzinfo=timezone.utc)
SUSPECT_HOURS = {11, 12, 13, 14}        # 12:14-12:44 UTC ± padding
RADII_KM = (50, 20, 10, 5)
GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"


def _bbox(radius_km: float) -> dict:
    """Square geojson polygon ±radius_km around OC01."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * cos(radians(OC01_LAT)))
    coords = [
        [OC01_LON - dlon, OC01_LAT - dlat],
        [OC01_LON + dlon, OC01_LAT - dlat],
        [OC01_LON + dlon, OC01_LAT + dlat],
        [OC01_LON - dlon, OC01_LAT + dlat],
        [OC01_LON - dlon, OC01_LAT - dlat],
    ]
    return {"geojson": {"type": "Polygon", "coordinates": [coords]}}


async def query_radius(
    client: httpx.AsyncClient, token: str, radius_km: float,
) -> list[dict]:
    day = DAY.strftime("%Y-%m-%d")
    next_day = (DAY + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"{GFW_BASE}/4wings/report"
        f"?datasets[0]=public-global-presence:latest"
        f"&date-range={day},{next_day}"
        f"&temporal-resolution=HOURLY"
        f"&spatial-resolution=HIGH"
        f"&group-by=VESSEL_ID"
        f"&format=JSON"
    )
    resp = await client.post(
        url, json=_bbox(radius_km),
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    data = resp.json()

    # Entries are keyed by dataset id; flatten the list of vessel rows.
    rows: list[dict] = []
    for entry in data.get("entries", []):
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.startswith("public-") and isinstance(v, list):
                    rows.extend(v)
    return rows


def _parse_hour(row: dict) -> int | None:
    ts = row.get("entryTimestamp") or row.get("date")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
    except Exception:
        return None


async def main() -> None:
    settings = Settings()
    if not settings.gfw_api_token:
        print("OS_GFW_API_TOKEN missing — can't query GFW")
        return

    print(f"OC01: {OC01_LAT}°N, {OC01_LON}°W")
    print(f"Date: 2019-03-09")
    print(f"Suspect window: hours {sorted(SUSPECT_HOURS)} UTC\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        for radius in RADII_KM:
            print(f"=== Radius ≤ {radius} km, hourly resolution ===")
            try:
                rows = await query_radius(client, settings.gfw_api_token, radius)
            except httpx.HTTPStatusError as e:
                print(f"  HTTP {e.response.status_code}: {e.response.text[:200]}")
                continue
            except Exception as e:
                print(f"  query failed: {e}")
                continue

            # Filter to suspect-hour rows
            in_window = [r for r in rows if _parse_hour(r) in SUSPECT_HOURS]
            print(f"  total hourly rows: {len(rows)}, "
                  f"in suspect window: {len(in_window)}")

            seen: set[str] = set()
            for r in in_window:
                vid = r.get("vesselId") or r.get("mmsi") or ""
                if vid in seen:
                    continue
                seen.add(vid)
                name = r.get("shipName") or vid or "?"
                klass = (r.get("vesselType") or r.get("geartype") or "?").lower()
                ts = r.get("entryTimestamp") or r.get("date") or "?"
                print(f"    {name:<30s} class={klass:<12s} ts={ts}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
