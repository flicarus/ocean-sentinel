import asyncio
import copernicusmarine
import numpy as np
from datetime import datetime
from ocean_sentinel.domain.models import OceanConditions, GeoPoint
from ocean_sentinel.config import Settings


class CopernicusAdapter:
    DATASET_CURRENTS = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
    DATASET_TEMP = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"

    def __init__(self, settings: Settings):
        self.settings = settings

    async def get_conditions(
        self,
        location: GeoPoint,
        timestamp: datetime
    ) -> OceanConditions:
        date_str = timestamp.strftime("%Y-%m-%d")
        margin = 0.25

        bbox = dict(
            minimum_longitude=location.lon - margin,
            maximum_longitude=location.lon + margin,
            minimum_latitude=location.lat - margin,
            maximum_latitude=location.lat + margin,
            start_datetime=date_str,
            end_datetime=date_str,
            minimum_depth=0.49,
            maximum_depth=2.0,
        )

        def _fetch_currents():
            return copernicusmarine.open_dataset(
                dataset_id=self.DATASET_CURRENTS,
                variables=["uo", "vo"],
                username=self.settings.copernicus_username,
                password=self.settings.copernicus_password,
                **bbox,
            )

        def _fetch_temp():
            return copernicusmarine.open_dataset(
                dataset_id=self.DATASET_TEMP,
                variables=["thetao"],
                username=self.settings.copernicus_username,
                password=self.settings.copernicus_password,
                **bbox,
            )

        ds_cur, ds_temp = await asyncio.gather(
            asyncio.to_thread(_fetch_currents),
            asyncio.to_thread(_fetch_temp),
        )

        uo = float(ds_cur["uo"].isel(time=0, depth=0).mean().values)
        vo = float(ds_cur["vo"].isel(time=0, depth=0).mean().values)
        sst = float(ds_temp["thetao"].isel(time=0, depth=0).mean().values)

        speed = float(np.sqrt(uo**2 + vo**2))
        direction = float(np.degrees(np.arctan2(vo, uo)) % 360)

        return OceanConditions(
            location=location,
            timestamp=timestamp,
            current_speed_ms=speed,
            current_direction_deg=direction,
            sea_surface_temp_c=sst,
        )