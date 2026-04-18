"""Dry-run the AIS labeling logic WITHOUT fetching any audio.

Use this before running bootstrap_ais_correlated.py to verify:
  - GFW credentials work
  - The chosen hydrophone/date has a sensible label distribution
  - You're not about to waste 25 min on a broken API call

Expected for Bush Point (Puget Sound ferry corridor) on a weekday:
  NONE: 50-100, LOW: 80-150, MEDIUM: 30-80, HIGH: 10-30, SKIPPED: 20-60
If you get 288 NONE, GFW is returning nothing — check your API token.

Usage:
    PYTHONPATH=src venv/bin/python3 scripts/ais_label_preview.py \\
        --hydrophone bush-point \\
        --date 2024-06-15 \\
        --radius-km 10 \\
        --step 300
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from collections import Counter

from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.orcasound import OrcasoundAdapter
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import TimeWindow

# Reuse the same node map and label logic from the main script
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


async def label_for_window(gfw, location, window, radius_km):
    nearby = await gfw.get_vessels_in_radius(location, radius_km, window)

    if not nearby:
        extended = TimeWindow(
            start=window.start - timedelta(hours=1),
            end=window.end,
        )
        historical = await gfw.get_vessels_in_radius(location, radius_km, extended)
        if historical:
            return None
        return "NONE"

    closest = nearby[0]
    if closest.distance_km <= 2.0:
        if closest.length_m and closest.length_m >= 80:
            return "HIGH"
        return "MEDIUM"
    if closest.distance_km <= radius_km:
        return "LOW"
    return None


async def main(hydrophone_id: str, date: str, radius_km: float, step: int) -> None:
    node_name = HYDROPHONE_NODES[hydrophone_id]
    settings = Settings()

    gfw = GFWAdapter(settings)
    orca = OrcasoundAdapter(node_name=node_name, settings=settings)
    location = orca.location

    base_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    total_windows = 86400 // step

    print(f"Preview: {hydrophone_id} / {date} — {total_windows} windows, no audio fetch")
    print(f"Location: lat={location.lat}, lon={location.lon}, radius={radius_km}km")
    print()

    counts: Counter = Counter()

    for i, offset in enumerate(range(0, 86400, step)):
        window_start = base_date + timedelta(seconds=offset)
        window = TimeWindow(
            start=window_start,
            end=window_start + timedelta(seconds=60),
        )

        try:
            label = await label_for_window(gfw, location, window, radius_km)
            key = label if label is not None else "SKIPPED"
            counts[key] += 1
        except Exception as e:
            counts["ERROR"] += 1
            print(f"  ERROR at offset {offset}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total_windows}] running totals: {dict(counts)}")

    await gfw.close()

    print()
    print("═" * 40)
    print(f"Label distribution — {hydrophone_id} / {date}")
    for label in ["NONE", "LOW", "MEDIUM", "HIGH", "SKIPPED", "ERROR"]:
        n = counts[label]
        bar = "█" * (n // 5)
        print(f"  {label:10s}: {n:4d}  {bar}")
    print("═" * 40)

    if counts["NONE"] == total_windows:
        print()
        print("⚠  WARNING: all windows returned NONE.")
        print("   This likely means GFW returned no data — check OS_GFW_API_TOKEN in .env")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Preview AIS label distribution without fetching audio"
    )
    ap.add_argument("--hydrophone", required=True, choices=list(HYDROPHONE_NODES.keys()))
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=300)
    args = ap.parse_args()

    asyncio.run(main(args.hydrophone, args.date, args.radius_km, args.step))
