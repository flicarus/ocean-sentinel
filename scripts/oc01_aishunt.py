"""OC01 detective hunt — verify the SanctSound mislabel hypothesis.

Our v6 CNN flagged 10 chunks from `SanctSound_OC01_01_671399974_20190309T115949Z.flac`
as `ship` while SanctSound metadata says ambient. Misclassified chunks
cluster at 12:14, 12:33-37, and 12:41-44 UTC on 2019-03-09. Chunk
acoustics are tonal + low-frequency-peaked + +3dB louder in engine band
than correct sanctsound samples — exactly what a vessel sounds like.

Hypothesis: there was a vessel passing the OC01 hydrophone (48.40°N,
-124.70°W) and SanctSound's ambient labeling didn't account for it.

Method:
  1. Hit GFW 4Wings vessel-presence for a 50km box around OC01 on
     2019-03-09 (daily resolution, the API's finest)
  2. Hit GFW gaps API for the same area / day in case any vessel went
     dark instead of broadcasting
  3. Print every vessel found with class, flag, name

If we find ANY broadcasting vessel within 50km that day, the hypothesis
is well-supported — the model was right, the ambient label was wrong.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.config import Settings
from ocean_sentinel.domain.models import GeoPoint, TimeWindow


OC01 = GeoPoint(lat=48.400, lon=-124.700)
DAY_START = datetime(2019, 3, 9, 0, 0, tzinfo=timezone.utc)
DAY_END = datetime(2019, 3, 9, 23, 59, tzinfo=timezone.utc)


async def main() -> None:
    settings = Settings()
    if not settings.gfw_api_token:
        print("OS_GFW_API_TOKEN not set — cannot query GFW")
        return

    gfw = GFWAdapter(settings)

    print(f"OC01 hydrophone:  {OC01.lat}°N, {OC01.lon}°W")
    print(f"Date:             2019-03-09")
    print(f"Suspect window:   12:14 — 12:44 UTC (10 misclassified chunks)")
    print()

    # 1. AIS gaps in the area on the day
    print("--- AIS gap events (vessels that went dark) ---")
    try:
        gaps = await gfw.get_ais_gaps(
            location=OC01,
            radius_km=50,
            time_window=TimeWindow(start=DAY_START, end=DAY_END),
        )
        if not gaps:
            print("  None found within 50km on 2019-03-09.")
        for g in gaps:
            print(
                f"  {g.vessel_name or g.vessel_id} "
                f"flag={g.flag_state or '?'} "
                f"start={g.gap_start.isoformat()} "
                f"duration={g.gap_duration_hours:.1f}h "
                f"intentional={g.intentional_disabling} "
                f"in_mpa={g.in_mpa}"
            )
    except Exception as e:
        print(f"  GFW gaps query failed: {e}")
    print()

    # 2. Broadcasting vessels in the area on the day
    print("--- Vessels broadcasting AIS within 50km on 2019-03-09 ---")
    try:
        vessels = await gfw.get_vessels_in_radius(
            location=OC01,
            radius_km=50,
            time_window=TimeWindow(start=DAY_START, end=DAY_END),
        )
        if not vessels:
            print("  None found.")
        for v in vessels:
            entry = v.present_start.isoformat() if v.present_start else "?"
            exit_ = v.present_end.isoformat() if v.present_end else "?"
            print(
                f"  {v.vessel_name or v.vessel_id:<30s} "
                f"class={v.vessel_class or '?':<15s} "
                f"flag={v.flag_state or '?':<5s} "
                f"present={entry} -> {exit_}"
            )
    except Exception as e:
        print(f"  GFW vessel-presence query failed: {e}")

    await gfw.close()


if __name__ == "__main__":
    asyncio.run(main())
