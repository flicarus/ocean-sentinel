import httpx
import structlog
from datetime import datetime, timedelta, timezone
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
        self._vessels_cache: dict[str, list[NearbyVessel]] = {}

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
        on the day of time_window.

        Uses GFW 4Wings Report API (POST /4wings/report) with daily resolution —
        the minimum the API supports. For acoustic labeling this is fine: a
        vessel within 10km on the same day is audible during any 60s clip.
        """
        day = time_window.start.strftime("%Y-%m-%d")
        next_day = (time_window.start + timedelta(days=1)).strftime("%Y-%m-%d")

        cache_key = f"{location.lat},{location.lon},{radius_km},{day}"
        if cache_key in self._vessels_cache:
            return self._vessels_cache[cache_key]

        log.info(
            "fetching_nearby_vessels",
            location=f"{location.lat},{location.lon}",
            radius_km=radius_km,
            day=day,
        )

        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * cos(radians(location.lat)))

        body = {
            "geojson": {
                "type": "Polygon",
                "coordinates": [[
                    [location.lon - lon_offset, location.lat - lat_offset],
                    [location.lon + lon_offset, location.lat - lat_offset],
                    [location.lon + lon_offset, location.lat + lat_offset],
                    [location.lon - lon_offset, location.lat + lat_offset],
                    [location.lon - lon_offset, location.lat - lat_offset],
                ]],
            }
        }

        # Build URL manually to avoid httpx encoding brackets in datasets[0]
        url = (
            f"{self.BASE_URL}/4wings/report"
            f"?datasets[0]=public-global-presence:latest"
            f"&date-range={day},{next_day}"
            f"&temporal-resolution=DAILY"
            f"&spatial-resolution=LOW"
            f"&group-by=VESSEL_ID"
            f"&format=JSON"
        )

        try:
            response = await self.client.post(url, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error(
                "gfw_http_error",
                status=e.response.status_code,
                body=e.response.text,
            )
            raise GFWError(
                code="gfw_http_error",
                message=f"GFW API returned {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            log.error("gfw_request_error", detail=str(e))
            raise GFWError(
                code="gfw_request_error",
                message=f"GFW API request failed: {e}"
            ) from e

        data = response.json()
        # GFW wraps vessel rows inside entries[0]["public-global-presence:..."]
        entries = data.get("entries", []) if isinstance(data, dict) else data
        rows: list = []
        for entry in entries:
            if isinstance(entry, dict):
                dataset_key = next((k for k in entry if k.startswith("public-")), None)
                if dataset_key:
                    rows.extend(entry[dataset_key])

        vessels: list[NearbyVessel] = []
        seen_vessel_ids: set[str] = set()
        for row in rows:
            try:
                vid = row.get("vesselId") or row.get("mmsi") or ""
                if vid in seen_vessel_ids:
                    continue
                seen_vessel_ids.add(vid)
                vessel_type = (row.get("vesselType") or row.get("geartype") or "").lower()

                entry_str = row.get("entryTimestamp")
                exit_str = row.get("exitTimestamp")
                present_start = (
                    datetime.fromisoformat(entry_str.replace("Z", "+00:00"))
                    if entry_str else None
                )
                present_end = (
                    datetime.fromisoformat(exit_str.replace("Z", "+00:00"))
                    if exit_str else None
                )

                vessels.append(NearbyVessel(
                    vessel_id=vid,
                    vessel_name=row.get("shipName"),
                    vessel_class=vessel_type or None,
                    flag_state=row.get("flag"),
                    position=location,
                    distance_km=0.0,
                    length_m=None,
                    present_start=present_start,
                    present_end=present_end,
                ))
            except (KeyError, ValueError) as e:
                log.warning("gfw_vessel_parse_error", error=str(e))
                continue

        log.info("gfw_vessels_found", count=len(vessels), day=day)
        self._vessels_cache[cache_key] = vessels
        return vessels

    async def close(self):
        await self.client.aclose()