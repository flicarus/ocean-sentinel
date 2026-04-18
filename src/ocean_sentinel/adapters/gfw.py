import httpx
import structlog
from datetime import datetime
from ocean_sentinel.domain.models import AISGapEvent, GeoPoint, TimeWindow
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

    async def close(self):
        await self.client.aclose()