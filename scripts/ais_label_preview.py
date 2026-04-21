"""Dry-run the AIS labeling logic WITHOUT fetching any audio.

Use this before running bootstrap_ais_correlated.py to verify:
  - GFW credentials work and the API returns data
  - The chosen hydrophone/date has a sensible label distribution
  - You're not about to waste 25 min on a broken API response

Expected for Bush Point (Puget Sound ferry corridor) on a weekday:
  NONE=40-60, MEDIUM=80-120, HIGH=20-40, SKIPPED=40-80
If you see ERROR for every window, check OS_GFW_API_TOKEN in .env.

Usage:
    PYTHONPATH=src venv/bin/python3 scripts/ais_label_preview.py \\
        --hydrophone bush-point \\
        --date 2024-06-15 \\
        --radius-km 10 \\
        --step 300
"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone

from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.adapters.orcasound import OrcasoundAdapter
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import GeoPoint, NearbyVessel, TimeWindow

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


def _label_from_vessels(vessels: list[NearbyVessel]) -> str | None:
    if not vessels:
        return None

    worst = max(vessels, key=_severity_rank)
    vc = (worst.vessel_class or "").lower()

    if vc in ("cargo", "tanker"):
        return "HIGH"
    if worst.length_m and worst.length_m >= 80:
        return "HIGH"
    if vc == "fishing":
        return "MEDIUM"
    if worst.length_m and 30 <= worst.length_m < 80:
        return "MEDIUM"
    if vc == "passenger":
        return "MEDIUM"

    return None  # SKIP — too ambiguous


def _vessels_at_time(vessels: list[NearbyVessel], window: TimeWindow) -> list[NearbyVessel]:
    """Filter to vessels whose presence window overlaps with the given time window."""
    result = []
    for v in vessels:
        if v.present_start is None or v.present_end is None:
            result.append(v)  # no timestamp info — assume present
            continue
        # overlap: vessel was present if it hadn't left before window starts
        # and hadn't arrived after window ends
        if v.present_end >= window.start and v.present_start <= window.end:
            result.append(v)
    return result


async def label_for_window(
    gfw: GFWAdapter,
    location: GeoPoint,
    window: TimeWindow,
    radius_km: float,
) -> str:
    """Returns NONE, MEDIUM, HIGH, or SKIPPED."""
    all_day = await gfw.get_vessels_in_radius(location, radius_km, window)
    current = _vessels_at_time(all_day, window)

    if current:
        result = _label_from_vessels(current)
        return result if result is not None else "SKIPPED"

    prior_window = TimeWindow(
        start=window.start - timedelta(hours=1),
        end=window.start,
    )
    all_prior = await gfw.get_vessels_in_radius(location, radius_km, prior_window)
    prior = _vessels_at_time(all_prior, prior_window)
    if prior:
        return "SKIPPED"  # acoustic tail risk

    return "NONE"


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
            counts[label] += 1
        except Exception as e:
            counts["ERROR"] += 1
            print(f"  ERROR at offset {offset}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total_windows}] {dict(counts)}")

    await gfw.close()

    print()
    print("═" * 40)
    print(f"Label distribution — {hydrophone_id} / {date}")
    for label in ["NONE", "MEDIUM", "HIGH", "SKIPPED", "ERROR"]:
        n = counts[label]
        bar = "█" * (n // 5)
        print(f"  {label:10s}: {n:4d}  {bar}")
    print("═" * 40)
    print("Expected Bush Point weekday: NONE=40-60, MEDIUM=80-120, HIGH=20-40, SKIPPED=40-80")

    if counts["ERROR"] == total_windows:
        print()
        print("WARNING: all windows errored — check OS_GFW_API_TOKEN in .env")


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
