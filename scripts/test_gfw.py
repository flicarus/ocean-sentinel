import asyncio
from datetime import datetime
from ocean_sentinel.config import Settings
from ocean_sentinel.adapters.gfw import GFWAdapter
from ocean_sentinel.domain.models import GeoPoint, TimeWindow

async def test():
    adapter = GFWAdapter(Settings())
    gaps = await adapter.get_ais_gaps(
        location=GeoPoint(lat=48.0, lon=-5.0),
        radius_km=50.0,
        time_window=TimeWindow(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31)
        )
    )
    print(f"Got {len(gaps)} AIS gap events")
    for g in gaps[:3]:
        print(g)

asyncio.run(test())