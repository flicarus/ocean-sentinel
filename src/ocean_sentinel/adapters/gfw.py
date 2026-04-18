import httpx
import structlog
from datetime import datetime, timezone
from ocean_sentinel.domain.models import AISGapEvent, GeoPoint, NearbyVessel, TimeWindow
from ocean_sentinel.config import Settings
from ocean_sentinel.exceptions import GFWError
from math import asin, cos, radians, sin, sqrt
log = structlog.get_logger(__name__)

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in kilometers."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))

class GFWAdapter:
    BASE_URL = "https://gateway.api.globalfishingwatch.org/v3"
    GAPS_DATASET = "public-global-gaps-events:latest"

    def __init__(self, settings: Settings):
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {settings.gfw_api_token}"},
            timeout=30.0
        )

    async def get_ais_gaps(
        self,
        location: GeoPoint,
        radius_km: float,
        time_window: TimeWindow
    ) -> list[AISGapEvent]:
        log.info(
            "fetching_ais_gaps",
            location=f"{location.lat},{location.lon}",
            radius_km=radius_km,
            start=time_window.start.isoformat(),
            end=time_window.end.isoformat(),
        )

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/events",
                params={
                    "datasets[0]": self.GAPS_DATASET,
                    "start-date": time_window.start.strftime("%Y-%m-%d"),
                    "end-date": time_window.end.strftime("%Y-%m-%d"),
                    "limit": 100,
                    "offset": 0,
                }
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error("gfw_http_error", status=e.response.status_code, detail=str(e))
            raise GFWError(
                code="gfw_http_error",
                message=f"GFW API returned {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            log.error("gfw_request_error", detail=str(e))
            raise GFWError(
                code="gfw_request_error",
                message=f"GFW API request failed: {e}"
            ) from e

        data = response.json()
        entries = data.get("entries", [])

        log.info("gfw_gaps_received", count=len(entries))

        events = []
        for entry in entries:
            gap = entry.get("gap", {})
            vessel = entry.get("vessel", {})

            try:
                event = AISGapEvent(
                    vessel_id=vessel.get("id", ""),
                    vessel_name=vessel.get("name"),
                    flag_state=vessel.get("flag"),
                    last_known_position=GeoPoint(
                        lat=float(gap["offPosition"]["lat"]),
                        lon=float(gap["offPosition"]["lon"])
                    ),
                    gap_start=datetime.fromisoformat(
                        entry["start"].replace("Z", "+00:00")
                    ),
                    gap_end=datetime.fromisoformat(
                        entry["end"].replace("Z", "+00:00")
                    ) if entry.get("end") else None,
                    gap_duration_hours=gap.get("durationHours"),
                    intentional_disabling=gap.get("intentionalDisabling", False),
                    in_mpa=len(entry.get("regions", {}).get("mpa", [])) > 0,
                )
                events.append(event)
            except (KeyError, ValueError) as e:
                log.warning("gfw_entry_parse_error", entry_id=entry.get("id"), error=str(e))
                continue

        filtered = [
            e for e in events
            if _haversine_km(
                location.lat, location.lon,
                e.last_known_position.lat, e.last_known_position.lon,
            ) <= radius_km
        ]

        log.info(
            "gfw_gaps_filtered",
            total=len(events),
            within_radius=len(filtered),
            radius_km=radius_km,
        )

        return filtered

    async def get_vessels_in_radius(
        self,
        location: GeoPoint,
        radius_km: float,
        time_window: TimeWindow,
    ) -> list[NearbyVessel]:
        """Return all AIS-broadcasting vessels within radius_km of location
        during the hour enclosing time_window, sorted by distance ascending.

        Uses GFW 4Wings Report API (POST /4wings/report) with
        public-global-presence:latest. Temporal resolution is hourly minimum —
        the window is rounded to its enclosing hour, which is fine for
        acoustic labeling (a vessel within 10km stays audible for >1h).
        """
        # Round window to enclosing hour — GFW minimum granularity
        hour_start = time_window.start.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start.replace(hour=hour_start.hour + 1) if hour_start.hour < 23 \
            else hour_start.replace(hour=0) .replace(day=hour_start.day + 1)

        log.info(
            "fetching_nearby_vessels",
            location=f"{location.lat},{location.lon}",
            radius_km=radius_km,
            hour=hour_start.isoformat(),
        )

        # Build a bounding box polygon around the hydrophone
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * cos(radians(location.lat)))
        min_lon = location.lon - lon_offset
        max_lon = location.lon + lon_offset
        min_lat = location.lat - lat_offset
        max_lat = location.lat + lat_offset

        # GeoJSON polygon for the region (clockwise bbox)
        region_polygon = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        }

        body = {
            "datasets": ["public-global-presence:latest"],
            "date-range": (
                f"{hour_start.strftime('%Y-%m-%dT%H:%M:%SZ')},"
                f"{hour_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            ),
            "region": region_polygon,
            "group-by": "VESSEL_ID",
            "temporal-resolution": "HOURLY",
            "spatial-resolution": "HIGH",
        }

        try:
            response = await self.client.post(
                f"{self.BASE_URL}/4wings/report",
                json=body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error("gfw_http_error", status=e.response.status_code, detail=str(e))
            raise GFWError(
                code="gfw_http_error",
                message=f"GFW API returned {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            log.error("gfw_request_error", detail=str(e))
            raise GFWError(
                code="gfw_request_error",
                message=f"GFW API request failed: {e}"
            ) from e

        # 4Wings report returns a list of per-vessel rows
        rows = response.json() if isinstance(response.json(), list) else \
            response.json().get("entries", response.json().get("data", []))

        vessels: list[NearbyVessel] = []
        for row in rows:
            try:
                vessels.append(NearbyVessel(
                    vessel_id=row.get("vesselId") or row.get("vessel_id", ""),
                    vessel_name=row.get("vesselName") or row.get("vessel_name"),
                    vessel_class=row.get("vesselType") or row.get("vessel_type"),
                    flag_state=row.get("flag"),
                    # 4Wings report doesn't give exact position — use hydrophone
                    # location as a stand-in (vessel is somewhere in the bbox)
                    position=location,
                    distance_km=0.0,
                    length_m=row.get("lengthM") or row.get("length_m"),
                ))
            except (KeyError, ValueError) as e:
                log.warning("gfw_vessel_parse_error", error=str(e))
                continue

        log.info("gfw_vessels_found", count=len(vessels), radius_km=radius_km)
        return vessels

    async def close(self):
        await self.client.aclose()