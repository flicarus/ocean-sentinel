import asyncio
from datetime import datetime
from ocean_sentinel.config import Settings
from ocean_sentinel.adapters.copernicus import CopernicusAdapter
from ocean_sentinel.domain.models import GeoPoint

async def test():
    settings = Settings()
    adapter = CopernicusAdapter(settings)
    location = GeoPoint(lat=48.0, lon=-5.0)  # Bay of Biscay, safe test spot
    conditions = await adapter.get_conditions(location, datetime(2024, 2, 1))
    print(f"Current speed: {conditions.current_speed_ms:.4f} m/s")
    print(f"Current direction: {conditions.current_direction_deg:.1f}°")
    print(f"SST: {conditions.sea_surface_temp_c:.2f}°C")

asyncio.run(test())